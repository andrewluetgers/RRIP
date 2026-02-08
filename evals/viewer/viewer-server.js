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

    // Jpegli JPEG baseline: jpegli_jpeg_baseline_q70
    const jpegliBaselineMatch = dirName.match(/^jpegli_jpeg_baseline_q(\d+)$/);
    if (jpegliBaselineMatch) {
      const quality = parseInt(jpegliBaselineMatch[1]);
      const tilesDir = path.join(dirPath, 'tiles');
      if (fs.existsSync(tilesDir)) {
        captures[`JPEGLI_JPEG_Q${quality}`] = {
          type: 'jpeg_baseline',
          encoder: 'jpegli',
          quality,
          q: quality,
          j: quality,
          name: dirName,
          has_tiles: true
        };
      }
      continue;
    }

    // libjpeg-turbo JPEG baseline: jpeg_baseline_q70
    const jpegBaselineMatch = dirName.match(/^jpeg_baseline_q(\d+)$/);
    if (jpegBaselineMatch) {
      const quality = parseInt(jpegBaselineMatch[1]);
      const tilesDir = path.join(dirPath, 'tiles');
      if (fs.existsSync(tilesDir)) {
        captures[`JPEG_Q${quality}`] = {
          type: 'jpeg_baseline',
          encoder: 'libjpeg-turbo',
          quality,
          q: quality,
          j: quality,
          name: dirName,
          has_tiles: true
        };
      }
      continue;
    }

    // ORIGAMI patterns
    let key, qVal = 0, jVal = 0, detectedEncoder = 'libjpeg-turbo';

    // Jpegli ORIGAMI: jpegli_debug_j50_pac or jpegli_j50_pac
    const jpegliPattern = /^jpegli_(?:debug_)?j(\d+)(?:_pac)?$/;
    const jpegliMatch = dirName.match(jpegliPattern);

    // Standard ORIGAMI: debug_j50_pac or j50_pac
    const newPattern = /^(?:debug_)?j(\d+)(?:_pac)?$/;
    const newMatch = dirName.match(newPattern);

    // Legacy ORIGAMI: debug_q32_j50_pac
    const legacyPattern = /q(\d+)_j(\d+)/i;
    const legacyMatch = dirName.match(legacyPattern);

    if (jpegliMatch) {
      jVal = parseInt(jpegliMatch[1]);
      key = `JPEGLI_J${jVal}`;
      detectedEncoder = 'jpegli';
    } else if (newMatch) {
      jVal = parseInt(newMatch[1]);
      key = `J${jVal}`;
    } else if (legacyMatch) {
      qVal = parseInt(legacyMatch[1]);
      jVal = parseInt(legacyMatch[2]);
      key = `Q${qVal}_J${jVal}`;
    } else {
      key = dirName;
    }

    // Check if this capture has necessary files
    const hasImages = fs.existsSync(path.join(dirPath, 'images'));
    const hasCompress = fs.existsSync(path.join(dirPath, 'compress'));
    const hasDecompress = fs.existsSync(path.join(dirPath, 'decompress'));

    if (hasImages || hasCompress || hasDecompress) {
      captures[key] = {
        type: 'origami',
        encoder: detectedEncoder,
        q: qVal,
        j: jVal,
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
    if (capture.type === 'jpeg_baseline') {
      if (imagePath.includes('original') || imagePath.includes('reconstructed') || imagePath.endsWith('.jpg')) {
        let jpegPath;
        if (imagePath.endsWith('.jpg')) {
          jpegPath = path.join(runDir, 'tiles', imagePath);
        } else {
          const tileId = imagePath.replace('_original.png', '').replace('_reconstructed.png', '');
          jpegPath = path.join(runDir, 'tiles', `${tileId}.jpg`);
        }

        if (fs.existsSync(jpegPath)) {
          return res.sendFile(jpegPath);
        }
      }
      return res.status(404).send('JPEG baseline does not have this image type');
    }

    // Handle ORIGAMI captures
    const compressDir = path.join(runDir, 'compress');
    const decompressDir = path.join(runDir, 'decompress');
    const imagesDir = path.join(runDir, 'images');

    const baseName = imagePath.replace('.png', '').replace('.jpg', '');

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
      // Try matching *_{baseName}.png then *_{baseName}.jpg
      for (const ext of ['.png', '.jpg']) {
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
