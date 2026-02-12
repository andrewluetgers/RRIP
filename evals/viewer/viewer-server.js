const express = require('express');
const fs = require('fs');
const fsPromises = require('fs').promises;
const path = require('path');

const app = express();
const PORT = process.env.PORT || 8084;
const RUNS_DIR = path.join(__dirname, '../runs');

// Serve static frontend
app.use(express.static(path.join(__dirname, 'public')));

/**
 * Scan the runs directory for all capture directories and classify them.
 */
async function scanCaptures() {
  const captures = {};

  let entries;
  try {
    entries = await fsPromises.readdir(RUNS_DIR, { withFileTypes: true });
  } catch (e) {
    console.error(`Runs directory not found: ${RUNS_DIR}`);
    return captures;
  }

  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    const dirName = entry.name;
    const dirPath = path.join(RUNS_DIR, dirName);

    // Encoder short names for display keys
    const encoderDisplayName = {
      'libjpeg-turbo': 'turbo',
      'jpegli': 'jpegli',
      'mozjpeg': 'mozjpeg',
      'jpegxl': 'jpegxl',
      'webp': 'webp',
    };

    // --- JPEG Baseline patterns: {encoder}_jpeg_baseline_q{N} or jpeg_baseline_q{N} ---
    const baselineMatch = dirName.match(/^(?:(jpegli|mozjpeg|jpegxl|webp)_)?jpeg_baseline_q(\d+)$/);
    if (baselineMatch) {
      const encoder = baselineMatch[1] || 'libjpeg-turbo';
      const quality = parseInt(baselineMatch[2]);
      const tilesDir = path.join(dirPath, 'tiles');
      if (fs.existsSync(tilesDir)) {
        const displayEncoder = encoderDisplayName[encoder] || encoder;
        captures[`JPEG ${displayEncoder} ${quality}`] = {
          type: 'jpeg_baseline',
          encoder,
          quality,
          q: quality,
          j: quality,
          name: dirName,
          has_tiles: true
        };
      }
      continue;
    }

    // --- JPEG2000 baseline patterns: jp2_baseline_q{N} ---
    const jp2Match = dirName.match(/^jp2_baseline_q(\d+)$/);
    if (jp2Match) {
      const quality = parseInt(jp2Match[1]);
      const tilesDir = path.join(dirPath, 'tiles');
      if (fs.existsSync(tilesDir)) {
        captures[`JP2 ${quality}`] = {
          type: 'jpeg_baseline',
          encoder: 'jpeg2000',
          quality,
          q: quality,
          j: quality,
          name: dirName,
          has_tiles: true
        };
      }
      continue;
    }

    // --- Single ORIGAMI: single_origami_p{N}_r{N} ---
    const singleMatch = dirName.match(/^(?:(jpegli|mozjpeg|jpegxl|webp)_)?single_origami_p(\d+)_r(\d+)$/);
    if (singleMatch) {
      const encoder = singleMatch[1] || 'libjpeg-turbo';
      const priorQ = parseInt(singleMatch[2]);
      const residualQ = parseInt(singleMatch[3]);
      const displayEncoder = encoderDisplayName[encoder] || encoder;
      captures[`SINGLE ${displayEncoder} p${priorQ} r${residualQ}`] = {
        type: 'single_origami', encoder, q: residualQ, j: residualQ,
        prior_quality: priorQ, residual_quality: residualQ,
        name: dirName, has_compress: fs.existsSync(path.join(dirPath, 'compress'))
      };
      continue;
    }

    // --- Single YCbCr subsample: single_ycbcr_y{N}_c{N} ---
    const ycbcrMatch = dirName.match(/^(?:(jpegli|mozjpeg|jpegxl|webp)_)?single_ycbcr_y(\d+)_c(\d+)$/);
    if (ycbcrMatch) {
      const encoder = ycbcrMatch[1] || 'libjpeg-turbo';
      const lumaQ = parseInt(ycbcrMatch[2]);
      const chromaQ = parseInt(ycbcrMatch[3]);
      const displayEncoder = encoderDisplayName[encoder] || encoder;
      captures[`YCbCr ${displayEncoder} y${lumaQ} c${chromaQ}`] = {
        type: 'single_ycbcr', encoder, q: lumaQ, j: lumaQ,
        luma_quality: lumaQ, chroma_quality: chromaQ,
        name: dirName, has_compress: fs.existsSync(path.join(dirPath, 'compress'))
      };
      continue;
    }

    // --- OptL2 split-quality ORIGAMI: optl2_{encoder}_debug_l1q{N}_l0q{N}_pac ---
    const optl2SplitMatch = dirName.match(/^optl2_(?:(jpegli|mozjpeg|jpegxl|webp)_)?debug_l1q(\d+)_l0q(\d+)(?:_pac)?$/);
    if (optl2SplitMatch) {
      const encoder = optl2SplitMatch[1] || 'libjpeg-turbo';
      const l1q = parseInt(optl2SplitMatch[2]);
      const l0q = parseInt(optl2SplitMatch[3]);
      const displayEncoder = encoderDisplayName[encoder] || encoder;

      const hasImages = fs.existsSync(path.join(dirPath, 'images'));
      const hasCompress = fs.existsSync(path.join(dirPath, 'compress'));
      const hasDecompress = fs.existsSync(path.join(dirPath, 'decompress'));

      if (hasImages || hasCompress || hasDecompress) {
        captures[`OPTL2 ${displayEncoder} L1=${l1q} L0=${l0q}`] = {
          type: 'origami',
          encoder,
          q: l0q,
          j: l0q,
          l1q,
          l0q,
          optl2: true,
          name: dirName,
          has_images: hasImages,
          has_compress: hasCompress,
          has_decompress: hasDecompress
        };
      }
      continue;
    }

    // --- OptL2 flat-quality ORIGAMI: optl2_{encoder}_debug_j{N}_pac ---
    const optl2FlatMatch = dirName.match(/^optl2_(?:(jpegli|mozjpeg|jpegxl|webp)_)?(?:debug_)?j(\d+)(?:_pac)?$/);
    if (optl2FlatMatch) {
      const encoder = optl2FlatMatch[1] || 'libjpeg-turbo';
      const quality = parseInt(optl2FlatMatch[2]);
      const displayEncoder = encoderDisplayName[encoder] || encoder;

      const hasImages = fs.existsSync(path.join(dirPath, 'images'));
      const hasCompress = fs.existsSync(path.join(dirPath, 'compress'));
      const hasDecompress = fs.existsSync(path.join(dirPath, 'decompress'));

      if (hasImages || hasCompress || hasDecompress) {
        captures[`OPTL2 ${displayEncoder} ${quality}`] = {
          type: 'origami',
          encoder,
          q: quality,
          j: quality,
          optl2: true,
          name: dirName,
          has_images: hasImages,
          has_compress: hasCompress,
          has_decompress: hasDecompress
        };
      }
      continue;
    }

    // --- Split-quality ORIGAMI: {encoder}_debug_l1q{N}_l0q{N}_pac ---
    const splitMatch = dirName.match(/^(?:(jpegli|mozjpeg|jpegxl|webp)_)?debug_l1q(\d+)_l0q(\d+)(?:_pac)?$/);
    if (splitMatch) {
      const encoder = splitMatch[1] || 'libjpeg-turbo';
      const l1q = parseInt(splitMatch[2]);
      const l0q = parseInt(splitMatch[3]);
      const displayEncoder = encoderDisplayName[encoder] || encoder;

      const hasImages = fs.existsSync(path.join(dirPath, 'images'));
      const hasCompress = fs.existsSync(path.join(dirPath, 'compress'));
      const hasDecompress = fs.existsSync(path.join(dirPath, 'decompress'));

      if (hasImages || hasCompress || hasDecompress) {
        captures[`ORIGAMI ${displayEncoder} L1=${l1q} L0=${l0q}`] = {
          type: 'origami',
          encoder,
          q: l0q,
          j: l0q,
          l1q,
          l0q,
          name: dirName,
          has_images: hasImages,
          has_compress: hasCompress,
          has_decompress: hasDecompress
        };
      }
      continue;
    }

    // --- ORIGAMI patterns: {encoder}_debug_j{N}_pac or debug_j{N}_pac ---
    const origamiMatch = dirName.match(/^(?:(jpegli|mozjpeg|jpegxl|webp)_)?(?:debug_)?j(\d+)(?:_pac)?$/);
    if (origamiMatch) {
      const encoder = origamiMatch[1] || 'libjpeg-turbo';
      const quality = parseInt(origamiMatch[2]);
      const displayEncoder = encoderDisplayName[encoder] || encoder;

      const hasImages = fs.existsSync(path.join(dirPath, 'images'));
      const hasCompress = fs.existsSync(path.join(dirPath, 'compress'));
      const hasDecompress = fs.existsSync(path.join(dirPath, 'decompress'));

      if (hasImages || hasCompress || hasDecompress) {
        captures[`ORIGAMI ${displayEncoder} ${quality}`] = {
          type: 'origami',
          encoder,
          q: quality,
          j: quality,
          name: dirName,
          has_images: hasImages,
          has_compress: hasCompress,
          has_decompress: hasDecompress
        };
      }
      continue;
    }

    // Fallback: unrecognized directories with content
    const hasImages = fs.existsSync(path.join(dirPath, 'images'));
    const hasCompress = fs.existsSync(path.join(dirPath, 'compress'));
    const hasDecompress = fs.existsSync(path.join(dirPath, 'decompress'));

    if (hasImages || hasCompress || hasDecompress) {
      captures[dirName] = {
        type: 'origami',
        encoder: 'unknown',
        q: 0,
        j: 0,
        name: dirName,
        has_images: hasImages,
        has_compress: hasCompress,
        has_decompress: hasDecompress
      };
    }
  }

  console.log(`Found ${Object.keys(captures).length} capture(s) in ${RUNS_DIR}`);
  for (const key of Object.keys(captures).sort()) {
    console.log(`  - ${key}: ${captures[key].name}`);
  }
  return captures;
}

// Cache captures on startup, rescan on request
let capturesCache = null;

async function getCaptures() {
  if (!capturesCache) {
    capturesCache = await scanCaptures();
  }
  return capturesCache;
}

// GET /captures.json - List all runs with metadata
app.get('/captures.json', async (req, res) => {
  try {
    // Rescan each time to pick up new runs
    capturesCache = await scanCaptures();
    const captures = capturesCache;

    const result = {};
    for (const [key, capture] of Object.entries(captures)) {
      result[key] = {
        q: capture.q,
        j: capture.j,
        name: capture.name,
        type: capture.type,
        encoder: capture.encoder
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
    // Baselines only have tile images (no pipeline stages), so map any request
    // for original/reconstructed to the tile file, and 404 everything else.
    if (capture.type === 'jpeg_baseline') {
      // Extract tile ID from paths like "L0_0_0_original.png" or "L0_0_0.jpg"
      const tileIdMatch = imagePath.match(/^(L[012]_\d+_\d+)/);
      if (tileIdMatch && (imagePath.includes('original') || imagePath.includes('reconstructed'))) {
        const tileId = tileIdMatch[1];
        // Try .jpg first, then .webp, then .png (JXL runs use PNG display copies)
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
      // Read manifest to get image key -> file path mapping
      let manifestData = null;
      const manifestPath = path.join(runDir, 'manifest.json');
      if (fs.existsSync(manifestPath)) {
        try {
          manifestData = JSON.parse(fs.readFileSync(manifestPath, 'utf-8'));
        } catch (e) { /* ignore parse errors */ }
      }

      if (manifestData && manifestData.images) {
        // Extract the image key from the request path
        // e.g. "original.png" -> key "original", "reconstructed.png" -> "reconstructed"
        const baseName = imagePath.replace(/\.(png|jpg|jxl|webp)$/, '');
        // Also try removing tile prefix like "L0_0_0_" since viewer may send it
        const cleanKey = baseName.replace(/^L[012]_\d+_\d+_/, '');

        const relPath = manifestData.images[cleanKey];
        if (relPath) {
          const absPath = path.join(runDir, relPath);
          if (fs.existsSync(absPath)) {
            return res.sendFile(absPath);
          }
        }

        // Fallback: try finding numbered file in compress/
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

    // L2 files don't have coordinates: L2_0_0_original -> L2_original
    let l2BaseName = null;
    if (baseName.startsWith('L2_')) {
      const parts = baseName.split('_');
      if (parts.length >= 4) {
        l2BaseName = parts[0] + '_' + parts.slice(3).join('_');
      }
    }

    // Helper to glob for numbered-prefix files like 017_L1_0_0_residual_centered.png
    function findNumberedFile(dir, base) {
      if (!fs.existsSync(dir)) return null;
      const files = fs.readdirSync(dir);
      // Try matching *_{baseName}.png then *_{baseName}.jpg then *_{baseName}.webp
      for (const ext of ['.png', '.jpg', '.webp']) {
        const match = files.find(f => f.endsWith(`_${base}${ext}`));
        if (match) return path.join(dir, match);
      }
      return null;
    }

    let filePath = null;

    // Old format: images/ directory
    if (capture.has_images) {
      const directPath = path.join(imagesDir, imagePath);
      if (fs.existsSync(directPath)) {
        filePath = directPath;
      }
    }

    // New format: compress/ and decompress/ with numbered prefixes
    if (!filePath) {
      // Try decompress directory first for reconstructed files
      if (imagePath.includes('reconstructed') && fs.existsSync(decompressDir)) {
        filePath = findNumberedFile(decompressDir, baseName);
      }

      // Try compress directory
      if (!filePath && fs.existsSync(compressDir)) {
        filePath = findNumberedFile(compressDir, baseName);
        // Try L2 name without coordinates
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

// GET /charts.json - List available chart files (must be before static middleware)
app.get('/charts.json', async (req, res) => {
  try {
    const files = await fsPromises.readdir(CHARTS_DIR);
    const charts = files
      .filter(f => f.endsWith('.png'))
      .sort();
    res.json(charts);
  } catch (e) {
    res.json([]);
  }
});

app.use('/charts', express.static(CHARTS_DIR));

// Serve run data files directly
app.use('/static', express.static(RUNS_DIR));

// Start server
app.listen(PORT, () => {
  console.log(`ORIGAMI Comparison Viewer`);
  console.log(`Runs directory: ${RUNS_DIR}`);
  console.log(`Server running at http://localhost:${PORT}`);
  console.log('Press Ctrl+C to stop');
  console.log('');

  // Initial scan
  scanCaptures();
});
