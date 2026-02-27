const express = require('express');
const fs = require('fs');
const fsPromises = require('fs').promises;
const path = require('path');
const { spawn } = require('child_process');

const { createProxyMiddleware } = require('http-proxy-middleware');

const app = express();
app.use(express.json());

const PORT = process.env.PORT || 8084;
const TILE_SERVER = process.env.TILE_SERVER || 'http://localhost:3007';
const RUNS_DIR = path.join(__dirname, '../runs');
const ARCHIVE_DIR = path.join(RUNS_DIR, '_archive');
const TEST_IMAGES_DIR = path.join(__dirname, '../test-images');
const ORIGAMI_BIN = path.join(__dirname, '../../server/target2/release/origami');

// Track running encode processes
const runningEncodes = {};

// Serve static frontend
app.use(express.static(path.join(__dirname, 'public')));

// ─── Run name generation (shared convention) ───

/**
 * Check if a baseq value represents lossless mode.
 */
function isLosslessBaseQ(val) {
  if (typeof val === 'string') {
    const v = val.trim().toLowerCase();
    return v === 'l' || v === 'lossless';
  }
  return false;
}

/**
 * Generate a standardized run directory name from parameters.
 * Format: b{baseq}_l1q{N}_l0q{N}[_optl2][_l1s{N}][_l0s{N}]
 * JPEG baseline: jpeg_q{N}
 */
function generateRunName(params) {
  if (params.type === 'jpeg_baseline') {
    return `jpeg_q${params.baseq || params.quality}`;
  }
  // V2 fused pipeline: no L1 residuals
  const bq = isLosslessBaseQ(params.baseq) ? 'L' : params.baseq;
  let name = `v2_b${bq}_l0q${params.l0q}`;
  if (params.optl2) name += '_optl2';
  if (params.optl2 && params.max_delta && params.max_delta !== 20) name += `_d${params.max_delta}`;
  if (params.sharpen) name += `_sh${Math.round(params.sharpen * 10)}`;
  if (params.l0_scale && params.l0_scale < 100) name += `_l0s${params.l0_scale}`;
  if (params.l0_sharpen) name += `_l0sh${Math.round(params.l0_sharpen * 10)}`;
  return name;
}

// ─── Run scanning ───

/**
 * Flatten decompression_phase from nested { L1: { tile_0_0: {...}, ... }, L0: { ... } }
 * into a flat array of tile metric objects.
 */
function flattenDecompPhase(dp) {
  const tiles = [];
  if (!dp || typeof dp !== 'object') return tiles;
  for (const level of ['L1', 'L0']) {
    if (dp[level] && typeof dp[level] === 'object') {
      for (const tileKey of Object.keys(dp[level])) {
        tiles.push(dp[level][tileKey]);
      }
    }
  }
  return tiles;
}

/**
 * Compute distribution stats from an array of numbers.
 * Returns { min, max, avg, values: sorted[] } or null if empty.
 */
function computeMetricDist(values) {
  if (!values || values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const sum = sorted.reduce((a, b) => a + b, 0);
  return {
    min: sorted[0],
    max: sorted[sorted.length - 1],
    avg: sum / sorted.length,
    values: sorted,
  };
}

/**
 * Scan runs directory. For each dir:
 * 1. Try manifest.json for authoritative metadata
 * 2. Fall back to new naming regex: b{N}_l1q{N}_l0q{N}[_optl2][_l1s{N}][_l0s{N}]
 * 3. Fall back to legacy scanCaptures() patterns
 * 4. Unrecognized dirs with content get dirname as display name
 */
async function scanRuns() {
  const runs = [];

  let entries;
  try {
    entries = await fsPromises.readdir(RUNS_DIR, { withFileTypes: true });
  } catch (e) {
    console.error(`Runs directory not found: ${RUNS_DIR}`);
    return runs;
  }

  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    const dirName = entry.name;
    if (dirName === '_archive') continue;
    const dirPath = path.join(RUNS_DIR, dirName);

    const hasCompress = fs.existsSync(path.join(dirPath, 'compress'));
    const hasDecompress = fs.existsSync(path.join(dirPath, 'decompress'));
    const hasImages = fs.existsSync(path.join(dirPath, 'images'));
    const hasTiles = fs.existsSync(path.join(dirPath, 'tiles'));
    const manifestPath = path.join(dirPath, 'manifest.json');
    const hasManifest = fs.existsSync(manifestPath);

    const isRunning = !!runningEncodes[dirName];
    if (!hasCompress && !hasDecompress && !hasImages && !hasTiles && !hasManifest && !isRunning) continue;

    let run = {
      name: dirName,
      dirName,
      type: 'origami',
      has_compress: hasCompress,
      has_decompress: hasDecompress,
      has_images: hasImages,
      has_tiles: hasTiles,
    };

    // Try reading manifest.json for authoritative metadata
    let manifest = null;
    if (hasManifest) {
      try {
        manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf-8'));
        run.baseq = manifest.baseq;
        run.l1q = manifest.l1q;
        run.l0q = manifest.l0q;
        run.optl2 = manifest.optl2 || false;
        run.l1_scale = manifest.l1_scale || 100;
        run.l0_scale = manifest.l0_scale || 100;
        run.sharpen = manifest.sharpen || null;
        run.l1_sharpen = manifest.l1_sharpen || null;
        run.l0_sharpen = manifest.l0_sharpen || null;
        run.encoder = manifest.encoder || 'turbojpeg';
        run.subsamp = manifest.subsamp || '444';
        run.total_bytes = manifest.total_bytes;
        run.source = manifest.source;

        // Compute metrics from tiles and decompression_phase
        if (manifest.tiles && manifest.tiles.length > 0) {
          const psnrs = manifest.tiles
            .filter(t => t.y_psnr_db != null)
            .map(t => t.y_psnr_db);
          if (psnrs.length > 0) {
            run.avg_psnr = psnrs.reduce((a, b) => a + b, 0) / psnrs.length;
            run.avg_psnr = Math.round(run.avg_psnr * 100) / 100;
          }

          // Flatten decompression_phase: { L1: { tile_0_0: {metrics}, ... }, L0: { ... } }
          const dp = manifest.decompression_phase;
          const tileMetrics = flattenDecompPhase(dp);

          const extract = (arr, key) => arr.filter(t => t[key] != null).map(t => t[key]);

          run.metrics = {
            psnr:       computeMetricDist(psnrs),
            ssim:       computeMetricDist(extract(tileMetrics, 'final_ssim')),
            ms_ssim:    computeMetricDist(extract(tileMetrics, 'final_ms_ssim')),
            delta_e:    computeMetricDist(extract(tileMetrics, 'final_delta_e')),
            lpips:      computeMetricDist(extract(tileMetrics, 'final_lpips')),
            blockiness: computeMetricDist(extract(tileMetrics, 'blockiness_delta')),
          };
        }
      } catch (e) { /* ignore parse errors */ }
    }

    // Generate display name
    // V2 fused pipeline: v2_b{baseq}_l0q{N}[_optl2[_d{N}]][_l0s{N}][_l0sh{N}]
    const v2Match = dirName.match(/^v2_b(\d+)_l0q(\d+)(?:_(optl2))?(?:_d(\d+))?(?:_l0s(\d+))?(?:_l0sh(\d+))?$/);
    // V1 naming: b{baseq}_l1q{N}_l0q{N}[_optl2]...
    const newNameMatch = dirName.match(/^b(\d+)_l1q(\d+)_l0q(\d+)(?:_(optl2))?(?:_sh(\d+))?(?:_l1s(\d+))?(?:_l0s(\d+))?(?:_l1sh(\d+))?(?:_l0sh(\d+))?$/);
    const jpegBaselineMatch = dirName.match(/^jpeg_q(\d+)$/);

    if (v2Match) {
      run.baseq = run.baseq || parseInt(v2Match[1]);
      run.l0q = run.l0q || parseInt(v2Match[2]);
      run.optl2 = run.optl2 || !!v2Match[3];
      const delta = v2Match[4] ? parseInt(v2Match[4]) : null;
      run.l0_scale = run.l0_scale || (v2Match[5] ? parseInt(v2Match[5]) : 100);
      run.l0_sharpen = run.l0_sharpen || (v2Match[6] ? parseInt(v2Match[6]) / 10 : null);
      run.subsamp = run.subsamp || '444';

      let label = `ORIGAMI v2 B${run.baseq} L0=${run.l0q}`;
      if (run.optl2) label += delta ? ` optL2 \u00b1${delta}` : ' optL2';
      if (run.l0_scale < 100) label += ` L0@${run.l0_scale}%`;
      if (run.l0_sharpen) label += ` L0sh=${Number(run.l0_sharpen).toFixed(1)}`;
      run.displayName = label;
    } else if (newNameMatch) {
      run.baseq = run.baseq || parseInt(newNameMatch[1]);
      run.l1q = run.l1q || parseInt(newNameMatch[2]);
      run.l0q = run.l0q || parseInt(newNameMatch[3]);
      run.optl2 = run.optl2 || !!newNameMatch[4];
      run.sharpen = run.sharpen || (newNameMatch[5] ? parseInt(newNameMatch[5]) / 10 : null);
      run.l1_scale = run.l1_scale || (newNameMatch[6] ? parseInt(newNameMatch[6]) : 100);
      run.l0_scale = run.l0_scale || (newNameMatch[7] ? parseInt(newNameMatch[7]) : 100);
      run.l1_sharpen = run.l1_sharpen || (newNameMatch[8] ? parseInt(newNameMatch[8]) / 10 : null);
      run.l0_sharpen = run.l0_sharpen || (newNameMatch[9] ? parseInt(newNameMatch[9]) / 10 : null);

      let label = `ORIGAMI turbo B${run.baseq} L1=${run.l1q} L0=${run.l0q}`;
      if (run.optl2) label += ' optL2';
      if (run.sharpen) label += ` sh=${Number(run.sharpen).toFixed(1)}`;
      if (run.l1_scale < 100) label += ` L1@${run.l1_scale}%`;
      if (run.l0_scale < 100) label += ` L0@${run.l0_scale}%`;
      if (run.l1_sharpen) label += ` L1sh=${Number(run.l1_sharpen).toFixed(1)}`;
      if (run.l0_sharpen) label += ` L0sh=${Number(run.l0_sharpen).toFixed(1)}`;
      run.displayName = label;
    } else if (jpegBaselineMatch) {
      const q = parseInt(jpegBaselineMatch[1]);
      run.type = 'jpeg_baseline';
      run.baseq = q;
      run.displayName = `JPEG ${q}`;
    } else {
      // Try legacy patterns for display name and metadata
      const legacy = getLegacyInfo(dirName);
      if (legacy) {
        run.displayName = legacy.displayName;
        if (legacy.type) run.type = legacy.type;
        if (legacy.baseq != null) run.baseq = run.baseq || legacy.baseq;
        if (legacy.l1q != null) run.l1q = run.l1q || legacy.l1q;
        if (legacy.l0q != null) run.l0q = run.l0q || legacy.l0q;
        if (legacy.optl2 != null) run.optl2 = run.optl2 || legacy.optl2;
        if (legacy.encoder) run.encoder = run.encoder || legacy.encoder;
        if (legacy.subsamp) run.subsamp = run.subsamp || legacy.subsamp;
      } else {
        run.displayName = dirName;
      }
    }

    // Check if encode is currently running
    if (runningEncodes[dirName]) {
      run.status = 'running';
    } else {
      run.status = 'complete';
    }

    runs.push(run);
  }

  runs.sort((a, b) => a.dirName.localeCompare(b.dirName));
  console.log(`Found ${runs.length} run(s) in ${RUNS_DIR}`);
  return runs;
}

/**
 * Legacy info extraction for old directory naming conventions.
 * Returns { displayName, type?, baseq?, l1q?, l0q?, optl2?, encoder?, subsamp? } or null.
 */
function getLegacyInfo(dirName) {
  const encoderDisplayName = {
    'libjpeg-turbo': 'turbo', 'jpegli': 'jpegli', 'mozjpeg': 'mozjpeg',
    'jpegxl': 'jpegxl', 'webp': 'webp',
  };

  let m;

  // JPEG baseline: {encoder}_jpeg_baseline_q{N} or jpeg_baseline_q{N}
  m = dirName.match(/^(?:(jpegli|mozjpeg|jpegxl|webp)_)?jpeg_baseline_q(\d+)$/);
  if (m) {
    const encoder = m[1] || 'libjpeg-turbo';
    const q = parseInt(m[2]);
    return {
      displayName: `JPEG ${encoderDisplayName[encoder] || encoder} ${q}`,
      type: 'jpeg_baseline', baseq: q, encoder,
    };
  }

  // JP2
  m = dirName.match(/^jp2_baseline_q(\d+)$/);
  if (m) return {
    displayName: `JP2 ${m[1]}`,
    type: 'jpeg_baseline', baseq: parseInt(m[1]), encoder: 'jpeg2000',
  };

  // rs_{subsamp}_b{N}[_optl2[_d{N}]]_l1q{N}_l0q{N}[_l0s{N}]
  m = dirName.match(/^rs_(444|420opt|420)_b(\d+)(?:_(optl2))?(?:_d(\d+))?_l1q(\d+)_l0q(\d+)(?:_l0s(\d+))?$/);
  if (m) {
    let label = `RS ORIGAMI turbo B${m[2]} L1=${m[5]} L0=${m[6]} ${m[1]}`;
    if (m[3]) label += m[4] ? ` optL2 \u00b1${m[4]}` : ' optL2';
    if (m[7]) label += ` L0@${m[7]}%`;
    return {
      displayName: label, type: 'origami', encoder: 'libjpeg-turbo',
      baseq: parseInt(m[2]), l1q: parseInt(m[5]), l0q: parseInt(m[6]),
      optl2: !!m[3], subsamp: m[1],
    };
  }

  // rs_{subsamp}_optl2_d{N}_l1q{N}_l0q{N}[_l0s{N}]
  m = dirName.match(/^rs_(444|420opt|420)_optl2_d(\d+)_l1q(\d+)_l0q(\d+)(?:_l0s(\d+))?$/);
  if (m) {
    let label = `RS ORIGAMI turbo L1=${m[3]} L0=${m[4]} ${m[1]} optL2 \u00b1${m[2]}`;
    if (m[5]) label += ` L0@${m[5]}%`;
    return {
      displayName: label, type: 'origami', encoder: 'libjpeg-turbo',
      l1q: parseInt(m[3]), l0q: parseInt(m[4]), optl2: true, subsamp: m[1],
    };
  }

  // rs_{subsamp}[_optl2]_l1q{N}_l0q{N}
  m = dirName.match(/^rs_(444|420opt|420)(?:_(optl2))?_l1q(\d+)_l0q(\d+)$/);
  if (m) {
    let label = `RS ORIGAMI turbo L1=${m[3]} L0=${m[4]} ${m[1]}`;
    if (m[2]) label += ' optL2';
    return {
      displayName: label, type: 'origami', encoder: 'libjpeg-turbo',
      l1q: parseInt(m[3]), l0q: parseInt(m[4]), optl2: !!m[2], subsamp: m[1],
    };
  }

  // rs_{subsamp}[_optl2]_j{N}
  m = dirName.match(/^rs_(444|420opt|420)(?:_(optl2))?_j(\d+)$/);
  if (m) {
    let label = `RS ORIGAMI turbo ${m[3]} ${m[1]}`;
    if (m[2]) label += ' optL2';
    return {
      displayName: label, type: 'origami', encoder: 'libjpeg-turbo',
      l1q: parseInt(m[3]), l0q: parseInt(m[3]), optl2: !!m[2], subsamp: m[1],
    };
  }

  // bc_rs_*
  m = dirName.match(/^bc_rs_(444|420opt|420)(?:_(optl2))?(?:_d(\d+))?_l1q(\d+)_l0q(\d+)(?:_l0s(\d+))?$/);
  if (m) {
    let label = `BC ORIGAMI turbo L1=${m[4]} L0=${m[5]} ${m[1]}`;
    if (m[2]) label += m[3] ? ` optL2 \u00b1${m[3]}` : ' optL2';
    if (m[6]) label += ` L0@${m[6]}%`;
    return {
      displayName: label, type: 'origami', encoder: 'libjpeg-turbo',
      l1q: parseInt(m[4]), l0q: parseInt(m[5]), optl2: !!m[2], subsamp: m[1],
    };
  }

  // uf_{filter}_{subsamp}[_optl2]_l1q{N}_l0q{N} — upsample filter sweep
  // filter: bl (bilinear), bc (bicubic), l3 (lanczos3)
  m = dirName.match(/^uf_(bl|bc|l3)_(444|420opt|420)(?:_(optl2))?_l1q(\d+)_l0q(\d+)$/);
  if (m) {
    const filterNames = { bl: 'bilinear', bc: 'bicubic', l3: 'lanczos3' };
    let label = `UF ${filterNames[m[1]]} L1=${m[4]} L0=${m[5]} ${m[2]}`;
    if (m[3]) label += ' optL2';
    return {
      displayName: label, type: 'origami', encoder: 'libjpeg-turbo',
      l1q: parseInt(m[4]), l0q: parseInt(m[5]), optl2: !!m[3], subsamp: m[2],
    };
  }

  // GPU
  m = dirName.match(/^gpu_(444|420opt|420)_b(\d+)(?:_(optl2))?(?:_d(\d+))?_l1q(\d+)_l0q(\d+)$/);
  if (m) {
    let label = `GPU nvjpeg B${m[2]} L1=${m[5]} L0=${m[6]} ${m[1]}`;
    if (m[3]) label += m[4] ? ` optL2 \u00b1${m[4]}` : ' optL2';
    return {
      displayName: label, type: 'origami', encoder: 'nvjpeg',
      baseq: parseInt(m[2]), l1q: parseInt(m[5]), l0q: parseInt(m[6]),
      optl2: !!m[3], subsamp: m[1],
    };
  }

  // optl2_debug_*
  m = dirName.match(/^optl2_(?:debug_)?(?:l1q(\d+)_l0q(\d+)|j(\d+))(?:_pac)?$/);
  if (m) {
    if (m[1]) return {
      displayName: `OPTL2 turbo L1=${m[1]} L0=${m[2]}`,
      type: 'origami', encoder: 'libjpeg-turbo', l1q: parseInt(m[1]), l0q: parseInt(m[2]), optl2: true,
    };
    return {
      displayName: `OPTL2 turbo ${m[3]}`,
      type: 'origami', encoder: 'libjpeg-turbo', l1q: parseInt(m[3]), l0q: parseInt(m[3]), optl2: true,
    };
  }

  // Generic debug_*
  m = dirName.match(/^(?:debug_)?(?:l1q(\d+)_l0q(\d+)|j(\d+))(?:_pac)?$/);
  if (m) {
    if (m[1]) return {
      displayName: `ORIGAMI turbo L1=${m[1]} L0=${m[2]}`,
      type: 'origami', encoder: 'libjpeg-turbo', l1q: parseInt(m[1]), l0q: parseInt(m[2]),
    };
    return {
      displayName: `ORIGAMI turbo ${m[3]}`,
      type: 'origami', encoder: 'libjpeg-turbo', l1q: parseInt(m[3]), l0q: parseInt(m[3]),
    };
  }

  // Sharpen patterns
  m = dirName.match(/^rs_(444|420opt|420)_b(\d+)_(?:sharp|sdec)(\d+)_l1q(\d+)_l0q(\d+)$/);
  if (m) {
    const strengthStr = (parseInt(m[3]) / 10).toFixed(1);
    return {
      displayName: `RS ORIGAMI turbo B${m[2]} L1=${m[4]} L0=${m[5]} ${m[1]} s=${strengthStr}`,
      type: 'origami', encoder: 'libjpeg-turbo',
      baseq: parseInt(m[2]), l1q: parseInt(m[4]), l0q: parseInt(m[5]), subsamp: m[1],
    };
  }

  // JXL residuals: jxl_{subsamp}[_optl2[_d{N}]]_l1q{N}_l0q{N}
  m = dirName.match(/^jxl_(444|420opt|420)(?:_(optl2)(?:_d(\d+))?)?_l1q(\d+)_l0q(\d+)$/);
  if (m) {
    let label = `JXL ORIGAMI L1=${m[4]} L0=${m[5]} ${m[1]}`;
    if (m[2]) label += m[3] ? ` optL2 \u00b1${m[3]}` : ' optL2';
    return {
      displayName: label, type: 'origami', encoder: 'jpegxl',
      l1q: parseInt(m[4]), l0q: parseInt(m[5]), optl2: !!m[2], subsamp: m[1],
    };
  }

  // JXL L2 storage: jxl_l2_{mode}_{subsamp}[_optl2[_d{N}]]_l1q{N}_l0q{N}
  m = dirName.match(/^jxl_l2_(\w+)_(444|420opt|420)(?:_(optl2)(?:_d(\d+))?)?_l1q(\d+)_l0q(\d+)$/);
  if (m) {
    const l2mode = m[1] === 'lossless' ? 'L2:JXL-LL' : `L2:JXL-Q${m[1]}`;
    let label = `ORIGAMI turbo ${l2mode} L1=${m[5]} L0=${m[6]} ${m[2]}`;
    if (m[3]) label += m[4] ? ` optL2 \u00b1${m[4]}` : ' optL2';
    return {
      displayName: label, type: 'origami', encoder: 'libjpeg-turbo',
      l1q: parseInt(m[5]), l0q: parseInt(m[6]), optl2: !!m[3], subsamp: m[2],
    };
  }

  return null;
}

// ─── Legacy scanCaptures() for /captures.json backward compat ───

/**
 * Build the old captures format from scanRuns() data.
 */
async function buildCapturesFromRuns(runs) {
  const captures = {};
  for (const run of runs) {
    const key = run.displayName;
    captures[key] = {
      q: run.l0q || run.baseq || 0,
      j: run.l0q || run.baseq || 0,
      name: run.dirName,
      type: run.type,
      encoder: run.encoder || 'libjpeg-turbo',
      has_images: run.has_images,
      has_compress: run.has_compress,
      has_decompress: run.has_decompress,
      has_tiles: run.has_tiles,
      baseq: run.baseq,
      l1q: run.l1q,
      l0q: run.l0q,
      subsamp: run.subsamp,
      optl2: run.optl2,
      l0_scale: run.l0_scale,
      l1_scale: run.l1_scale,
      sharpen: run.sharpen,
      l1_sharpen: run.l1_sharpen,
      l0_sharpen: run.l0_sharpen,
      metrics: run.metrics,
    };
  }
  return captures;
}

// Cache
let capturesCache = null;

async function getCaptures() {
  if (!capturesCache) {
    const runs = await scanRuns();
    capturesCache = await buildCapturesFromRuns(runs);
  }
  return capturesCache;
}

// ─── Build encode command ───

function buildEncodeCommand(params, outDir) {
  const baseqArg = isLosslessBaseQ(params.baseq) ? 'lossless' : String(params.baseq);
  const args = [
    'encode',
    '--image', params.image,
    '--out', outDir,
    '--baseq', baseqArg,
    '--l0q', String(params.l0q),
    '--subsamp', '444',
    '--manifest',
    '--debug-images',
    '--pack',
  ];
  if (params.optl2) {
    args.push('--optl2', '--max-delta', String(params.max_delta || 20));
  }
  if (params.sharpen) {
    args.push('--sharpen', Number(params.sharpen).toFixed(1));
  }
  if (params.l0_scale && params.l0_scale < 100) {
    args.push('--l0-scale', String(params.l0_scale));
  }
  if (params.l0_sharpen) {
    args.push('--l0-sharpen', Number(params.l0_sharpen).toFixed(1));
  }
  return args;
}

// ─── API endpoints ───

// GET /api/runs — List all runs with metadata + running status
app.get('/api/runs', async (req, res) => {
  try {
    const runs = await scanRuns();
    capturesCache = await buildCapturesFromRuns(runs);
    res.json(runs);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// POST /api/runs — Create & launch a new encode
app.post('/api/runs', async (req, res) => {
  try {
    const { image, baseq, l0q, optl2, max_delta, sharpen, l0_scale, l0_sharpen } = req.body;

    if (!image) {
      return res.status(400).json({ error: 'image is required' });
    }

    const imagePath = path.join(TEST_IMAGES_DIR, image);
    if (!fs.existsSync(imagePath)) {
      return res.status(400).json({ error: `Image not found: ${image}` });
    }

    const params = {
      image: imagePath,
      baseq: isLosslessBaseQ(baseq) ? 'lossless' : (baseq || 95),
      l0q: l0q || 60,
      optl2: optl2 !== false,
      max_delta: max_delta || 20,
      sharpen: sharpen || null,
      l0_scale: l0_scale || 100,
      l0_sharpen: l0_sharpen || null,
    };

    const runName = generateRunName(params);
    const outDir = path.join(RUNS_DIR, runName);

    if (fs.existsSync(outDir)) {
      return res.status(409).json({ error: `Run already exists: ${runName}` });
    }

    if (runningEncodes[runName]) {
      return res.status(409).json({ error: `Encode already running: ${runName}` });
    }

    // Check binary exists
    if (!fs.existsSync(ORIGAMI_BIN)) {
      return res.status(500).json({ error: `Encoder binary not found: ${ORIGAMI_BIN}` });
    }

    fs.mkdirSync(outDir, { recursive: true });

    const cmdArgs = buildEncodeCommand(params, outDir);
    const logPath = path.join(outDir, 'encode.log');
    const logStream = fs.createWriteStream(logPath);

    console.log(`Starting encode: ${ORIGAMI_BIN} ${cmdArgs.join(' ')}`);
    const child = spawn(ORIGAMI_BIN, cmdArgs, {
      stdio: ['ignore', 'pipe', 'pipe'],
      env: { ...process.env, RUST_LOG: 'info' },
    });

    child.stdout.pipe(logStream);
    child.stderr.pipe(logStream);

    runningEncodes[runName] = {
      pid: child.pid,
      child,
      startTime: Date.now(),
      params,
    };

    child.on('close', (code) => {
      console.log(`Encode ${runName} finished with code ${code}`);
      delete runningEncodes[runName];
      logStream.end();

      // Auto-run compute_metrics.py for full visual metrics (SSIM, VIF, Delta E, LPIPS)
      if (code === 0) {
        const metricsScript = path.join(__dirname, '../scripts/compute_metrics.py');
        if (fs.existsSync(metricsScript)) {
          console.log(`Running compute_metrics.py for ${runName}...`);
          const metrics = spawn('uv', ['run', 'python', metricsScript, outDir], {
            stdio: ['ignore', 'pipe', 'pipe'],
            cwd: path.join(__dirname, '../..'),
          });
          const metricsLog = fs.createWriteStream(path.join(outDir, 'metrics.log'));
          metrics.stdout.pipe(metricsLog);
          metrics.stderr.pipe(metricsLog);
          metrics.on('close', (mCode) => {
            console.log(`compute_metrics.py for ${runName} finished with code ${mCode}`);
            metricsLog.end();
          });
        }
      }
    });

    child.on('error', (err) => {
      console.error(`Encode ${runName} error:`, err);
      delete runningEncodes[runName];
      logStream.end();
    });

    res.json({ name: runName, status: 'started', pid: child.pid });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// POST /api/runs/:name/archive — Move run to _archive/
app.post('/api/runs/:name/archive', async (req, res) => {
  try {
    const { name } = req.params;
    const srcDir = path.join(RUNS_DIR, name);

    if (!fs.existsSync(srcDir)) {
      return res.status(404).json({ error: `Run not found: ${name}` });
    }

    if (runningEncodes[name]) {
      return res.status(409).json({ error: `Cannot archive running encode: ${name}` });
    }

    fs.mkdirSync(ARCHIVE_DIR, { recursive: true });
    const destDir = path.join(ARCHIVE_DIR, name);
    fs.renameSync(srcDir, destDir);
    capturesCache = null;
    res.json({ archived: name });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// GET /api/runs/:name/status — Check encode progress + log tail
app.get('/api/runs/:name/status', async (req, res) => {
  try {
    const { name } = req.params;
    const dirPath = path.join(RUNS_DIR, name);
    const logPath = path.join(dirPath, 'encode.log');

    const running = !!runningEncodes[name];
    let logTail = '';
    if (fs.existsSync(logPath)) {
      const content = fs.readFileSync(logPath, 'utf-8');
      const lines = content.split('\n');
      logTail = lines.slice(-20).join('\n');
    }

    let manifest = null;
    const manifestPath = path.join(dirPath, 'manifest.json');
    if (fs.existsSync(manifestPath)) {
      try { manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf-8')); } catch (e) {}
    }

    res.json({
      name,
      running,
      pid: runningEncodes[name]?.pid,
      elapsed_ms: runningEncodes[name] ? Date.now() - runningEncodes[name].startTime : null,
      logTail,
      hasManifest: !!manifest,
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// GET /api/test-images — List images in test-images/
app.get('/api/test-images', async (req, res) => {
  try {
    let entries;
    try {
      entries = await fsPromises.readdir(TEST_IMAGES_DIR, { withFileTypes: true });
    } catch (e) {
      return res.json([]);
    }
    const images = entries
      .filter(e => !e.isDirectory() && /\.(jpg|jpeg|png|tif|tiff)$/i.test(e.name))
      .map(e => e.name)
      .sort();
    res.json(images);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// ─── Existing endpoints (backward compat) ───

// GET /captures.json - List all runs in legacy format
app.get('/captures.json', async (req, res) => {
  try {
    const runs = await scanRuns();
    capturesCache = await buildCapturesFromRuns(runs);
    const result = {};
    for (const [key, capture] of Object.entries(capturesCache)) {
      result[key] = {
        q: capture.q,
        j: capture.j,
        name: capture.name,
        type: capture.type,
        encoder: capture.encoder,
      };
    }
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// GET /manifest?capture=KEY - Return manifest JSON from run dir
app.get('/manifest', async (req, res) => {
  try {
    const captureKey = req.query.capture;
    const captures = await getCaptures();

    if (!captureKey || !captures[captureKey]) {
      return res.status(404).json({ error: `Capture not found: ${captureKey}` });
    }

    const capture = captures[captureKey];
    const runDir = path.join(RUNS_DIR, capture.name);

    // Try analysis_manifest.json first, then manifest.json
    let manifestPath = path.join(runDir, 'analysis_manifest.json');
    if (!fs.existsSync(manifestPath)) {
      manifestPath = path.join(runDir, 'manifest.json');
    }
    if (!fs.existsSync(manifestPath)) {
      return res.status(404).json({ error: 'Manifest not found' });
    }

    const data = await fsPromises.readFile(manifestPath, 'utf-8');
    res.type('application/json').send(data);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// GET /image/:path?capture=KEY - Serve images from compress/decompress/tiles dirs
app.get('/image/*', async (req, res) => {
  try {
    const imagePath = req.params[0]; // everything after /image/
    const captureKey = req.query.capture;
    const captures = await getCaptures();

    if (!captureKey || !captures[captureKey]) {
      return res.status(404).send('Capture not found');
    }

    const capture = captures[captureKey];
    const runDir = path.join(RUNS_DIR, capture.name);

    // Handle JPEG baseline captures
    if (capture.type === 'jpeg_baseline') {
      const tileIdMatch = imagePath.match(/^(L[012]_\d+_\d+)/);
      if (tileIdMatch && (imagePath.includes('original') || imagePath.includes('reconstructed'))) {
        const tileId = tileIdMatch[1];
        for (const ext of ['.jpg', '.webp', '.png']) {
          const tilePath = path.join(runDir, 'tiles', `${tileId}${ext}`);
          if (fs.existsSync(tilePath)) {
            return res.sendFile(tilePath);
          }
        }
      }
      return res.status(404).send('Not available for baseline');
    }

    // Handle single_origami and single_ycbcr captures
    if (capture.type === 'single_origami' || capture.type === 'single_ycbcr') {
      let manifestData = null;
      const manifestPath = path.join(runDir, 'manifest.json');
      if (fs.existsSync(manifestPath)) {
        try {
          manifestData = JSON.parse(fs.readFileSync(manifestPath, 'utf-8'));
        } catch (e) { /* ignore parse errors */ }
      }

      if (manifestData && manifestData.images) {
        const baseName = imagePath.replace(/\.(png|jpg|jxl|webp)$/, '');
        const cleanKey = baseName.replace(/^L[012]_\d+_\d+_/, '');

        const relPath = manifestData.images[cleanKey];
        if (relPath) {
          const absPath = path.join(runDir, relPath);
          if (fs.existsSync(absPath)) {
            return res.sendFile(absPath);
          }
        }

        const compDir = path.join(runDir, 'compress');
        if (fs.existsSync(compDir)) {
          const files = fs.readdirSync(compDir);
          for (const ext of ['.png', '.jpg', '.webp']) {
            const match = files.find(f => f.endsWith(`_${cleanKey}${ext}`) || f.endsWith(`_${cleanKey}_rgb${ext}`));
            if (match) return res.sendFile(path.join(compDir, match));
          }
        }
      }
      return res.status(404).send('Image not found');
    }

    // Handle ORIGAMI captures
    const compressDir = path.join(runDir, 'compress');
    const decompressDir = path.join(runDir, 'decompress');
    const imagesDir = path.join(runDir, 'images');

    const baseName = imagePath.replace('.png', '').replace('.jpg', '').replace('.webp', '');

    let l2BaseName = null;
    if (baseName.startsWith('L2_')) {
      const parts = baseName.split('_');
      if (parts.length >= 4) {
        l2BaseName = parts[0] + '_' + parts.slice(3).join('_');
      }
    }

    function findNumberedFile(dir, base) {
      if (!fs.existsSync(dir)) return null;
      const files = fs.readdirSync(dir);
      for (const ext of ['.png', '.jpg', '.webp']) {
        const match = files.find(f => f.endsWith(`_${base}${ext}`));
        if (match) return path.join(dir, match);
      }
      return null;
    }

    let filePath = null;

    if (capture.has_images) {
      const directPath = path.join(imagesDir, imagePath);
      if (fs.existsSync(directPath)) {
        filePath = directPath;
      }
    }

    if (!filePath) {
      if ((imagePath.includes('reconstructed') || imagePath.includes('decode')) && fs.existsSync(decompressDir)) {
        filePath = findNumberedFile(decompressDir, baseName);
        if (!filePath && l2BaseName) {
          filePath = findNumberedFile(decompressDir, l2BaseName);
        }
      }

      if (!filePath && fs.existsSync(compressDir)) {
        filePath = findNumberedFile(compressDir, baseName);
        if (!filePath && l2BaseName) {
          filePath = findNumberedFile(compressDir, l2BaseName);
        }
      }
    }

    if (filePath && fs.existsSync(filePath)) {
      return res.sendFile(filePath);
    }

    res.status(404).send('Image not found');
  } catch (error) {
    console.error('Error serving image:', error);
    res.status(500).send('Error serving image');
  }
});

// Serve chart images from evals/charts/
const CHARTS_DIR = path.join(__dirname, '../charts');

async function findPngsRecursive(dir, base) {
  let results = [];
  let entries;
  try {
    entries = await fsPromises.readdir(dir, { withFileTypes: true });
  } catch (e) {
    return results;
  }
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      const sub = await findPngsRecursive(fullPath, base);
      results = results.concat(sub);
    } else if (entry.name.endsWith('.png')) {
      results.push(path.relative(base, fullPath));
    }
  }
  return results;
}

app.get('/charts.json', async (req, res) => {
  try {
    const charts = await findPngsRecursive(CHARTS_DIR, CHARTS_DIR);
    charts.sort();
    res.json(charts);
  } catch (e) {
    res.json([]);
  }
});

app.use('/charts', express.static(CHARTS_DIR));
app.use('/figures', express.static(path.join(__dirname, '../../paper/figures')));
app.use('/paper', express.static(path.join(__dirname, '../../paper')));
app.use('/docs', express.static(path.join(__dirname, '../../docs')));

// Serve run data files directly
app.use('/static', express.static(RUNS_DIR));

// GET /pipeline-data - Return pre-computed comparison data for the pipeline walkthrough
const PIPELINE_RUNS = [
  'rs_444_optl2_l1q60_l0q40',
  'rs_444_l1q60_l0q40',
  'rs_420_optl2_l1q60_l0q40',
  'rs_420opt_optl2_l1q60_l0q40',
  'rs_444_optl2_j40',
  'jpeg_baseline_q40',
];

app.get('/pipeline-data', async (req, res) => {
  try {
    const result = {};
    for (const runName of PIPELINE_RUNS) {
      const runDir = path.join(RUNS_DIR, runName);
      const data = { name: runName, manifest: null, summary: null, compressFiles: [], decompressFiles: [] };

      for (const mf of ['manifest.json']) {
        const mp = path.join(runDir, mf);
        if (fs.existsSync(mp)) {
          try { data.manifest = JSON.parse(fs.readFileSync(mp, 'utf-8')); } catch (e) {}
        }
      }

      const sp = path.join(runDir, 'summary.json');
      if (fs.existsSync(sp)) {
        try { data.summary = JSON.parse(fs.readFileSync(sp, 'utf-8')); } catch (e) {}
      }

      const compDir = path.join(runDir, 'compress');
      if (fs.existsSync(compDir)) {
        try { data.compressFiles = fs.readdirSync(compDir).sort(); } catch (e) {}
      }
      const decompDir = path.join(runDir, 'decompress');
      if (fs.existsSync(decompDir)) {
        try { data.decompressFiles = fs.readdirSync(decompDir).sort(); } catch (e) {}
      }

      if (runName.includes('jpeg_baseline')) {
        const st = path.join(runDir, 'summary.txt');
        if (fs.existsSync(st)) {
          try { data.summaryText = fs.readFileSync(st, 'utf-8'); } catch (e) {}
        }
      }

      result[runName] = data;
    }
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Serve raw run files for pipeline page image access
app.use('/run-image', express.static(RUNS_DIR));

// Proxy tile server routes to Rust tile server (same origin for browser)
const tileProxy = createProxyMiddleware({
  target: TILE_SERVER,
  changeOrigin: true,
  ws: false,
});
app.use('/dzi', tileProxy);
app.use('/tiles', tileProxy);
app.use('/viewer', tileProxy);
app.use('/compare2', tileProxy);
app.use('/compare', tileProxy);
app.use('/compare4', tileProxy);
app.use('/slides.json', tileProxy);

// Start server (HTTPS for HTTP/2 tile multiplexing)
const https = require('https');
const { execSync } = require('child_process');

// Generate self-signed cert via openssl (reuse if exists)
const certDir = path.join(__dirname, '.certs');
const certPath = path.join(certDir, 'cert.pem');
const keyPath = path.join(certDir, 'key.pem');

if (!fs.existsSync(certPath)) {
  fs.mkdirSync(certDir, { recursive: true });
  execSync(`openssl req -x509 -newkey rsa:2048 -keyout ${keyPath} -out ${certPath} -days 365 -nodes -subj "/CN=localhost"`, { stdio: 'pipe' });
  console.log('Generated self-signed cert for HTTPS');
}

const httpsServer = https.createServer({
  key: fs.readFileSync(keyPath),
  cert: fs.readFileSync(certPath),
}, app);

httpsServer.listen(PORT, () => {
  console.log(`ORIGAMI Comparison Viewer`);
  console.log(`Runs directory: ${RUNS_DIR}`);
  console.log(`Server running at https://localhost:${PORT}`);
  console.log('Press Ctrl+C to stop');
  console.log('');

  // Initial scan
  scanRuns();
});
