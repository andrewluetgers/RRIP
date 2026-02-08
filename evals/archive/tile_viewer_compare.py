#!/usr/bin/env python3
"""
Multi-column comparison tile viewer for ORIGAMI debug tiles.
Shows image stages in rows with different Q/J runs in columns.
"""

import argparse
import pathlib
import json
import numpy as np
from PIL import Image
from io import BytesIO
from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.parse
import os
import re


class ComparisonViewerHandler(SimpleHTTPRequestHandler):
    """Handler for serving multiple capture comparisons."""

    def __init__(self, *args, paper_dir=None, **kwargs):
        self.paper_dir = pathlib.Path(paper_dir) if paper_dir else pathlib.Path.cwd()
        self.captures = self.scan_captures()
        super().__init__(*args, **kwargs)

    def scan_captures(self):
        """Scan for all capture directories in paper/runs."""
        captures = {}

        # Simply scan all directories in paper/runs
        runs_dir = self.paper_dir / 'runs'
        if not runs_dir.exists():
            print(f"Creating runs directory: {runs_dir}")
            runs_dir.mkdir(exist_ok=True)
            return captures

        for dir_path in runs_dir.iterdir():
            if dir_path.is_dir():
                # Check if it's a jpegli JPEG baseline capture
                if dir_path.name.startswith('jpegli_jpeg_baseline_q'):
                    jpeg_match = re.search(r'jpegli_jpeg_baseline_q(\d+)', dir_path.name)
                    if jpeg_match:
                        quality = int(jpeg_match.group(1))
                        key = f"JPEGLI_JPEG_Q{quality}"

                        if (dir_path / 'tiles').exists():
                            captures[key] = {
                                'type': 'jpeg_baseline',
                                'encoder': 'jpegli',
                                'quality': quality,
                                'q': quality,
                                'j': quality,
                                'path': dir_path,
                                'name': dir_path.name,
                                'has_tiles': True
                            }

                # Check if it's a libjpeg-turbo JPEG baseline capture
                elif dir_path.name.startswith('jpeg_baseline_q'):
                    # Extract quality from name (e.g., jpeg_baseline_q70)
                    jpeg_match = re.search(r'jpeg_baseline_q(\d+)', dir_path.name)
                    if jpeg_match:
                        quality = int(jpeg_match.group(1))
                        key = f"JPEG_Q{quality}"

                        # Check if it has the tiles directory
                        if (dir_path / 'tiles').exists():
                            captures[key] = {
                                'type': 'jpeg_baseline',
                                'encoder': 'libjpeg-turbo',
                                'quality': quality,
                                'q': quality,  # For compatibility
                                'j': quality,  # For compatibility
                                'path': dir_path,
                                'name': dir_path.name,
                                'has_tiles': True
                            }
                else:
                    # Try to match ORIGAMI patterns
                    # Jpegli pattern: jpegli_debug_j50_pac
                    jpegli_pattern = re.compile(r'^jpegli_(?:debug_)?j(\d+)(?:_pac)?$')
                    jpegli_match = jpegli_pattern.search(dir_path.name)

                    # New pattern: debug_j50_pac (no quantization)
                    new_pattern = re.compile(r'^(?:debug_)?j(\d+)(?:_pac)?$')
                    new_match = new_pattern.search(dir_path.name)

                    # Legacy pattern: debug_q32_j50_pac (with quantization)
                    legacy_pattern = re.compile(r'q(\d+)_j(\d+)')
                    legacy_match = legacy_pattern.search(dir_path.name.lower())

                    if jpegli_match:
                        j_val = int(jpegli_match.group(1))
                        q_val = 0
                        key = f"JPEGLI_J{j_val}"
                    elif new_match:
                        j_val = int(new_match.group(1))
                        q_val = 0
                        key = f"J{j_val}"
                    elif legacy_match:
                        q_val = int(legacy_match.group(1))
                        j_val = int(legacy_match.group(2))
                        key = f"Q{q_val}_J{j_val}"
                    else:
                        # Use directory name as-is if no pattern found
                        key = dir_path.name
                        q_val = 0
                        j_val = 0

                    # Check if this capture has the necessary files
                    has_files = (
                        (dir_path / 'images').exists() or
                        (dir_path / 'arrays').exists() or
                        (dir_path / 'compress').exists() or
                        (dir_path / 'decompress').exists()
                    )

                    # Determine encoder from key prefix
                    detected_encoder = 'jpegli' if key.startswith('JPEGLI_') else 'libjpeg-turbo'

                    if has_files:
                        captures[key] = {
                            'type': 'origami',
                            'encoder': detected_encoder,
                            'q': q_val,
                            'j': j_val,
                            'path': dir_path,  # Keep as Path object for internal use
                            'name': dir_path.name,
                            'has_images': (dir_path / 'images').exists(),
                            'has_compress': (dir_path / 'compress').exists(),
                            'has_decompress': (dir_path / 'decompress').exists()
                        }

        print(f"Found {len(captures)} capture(s) in paper/runs/")
        for key in sorted(captures.keys()):
            print(f"  - {key}: {captures[key]['name']}")
        return captures

    def do_GET(self):
        """Handle GET requests."""
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path == '/' or path == '/index.html':
            self.serve_viewer()
        elif path == '/captures.json':
            self.serve_capture_list()
        elif path == '/tile_data.json':
            tile_id = query.get('tile', ['L0_0_0'])[0]
            self.serve_tile_comparison(tile_id)
        elif path.startswith('/image/'):
            capture = query.get('capture', [''])[0]
            self.serve_image(path[7:], capture)
        elif path.startswith('/array/'):
            capture = query.get('capture', [''])[0]
            self.serve_array(path[7:], capture)
        elif path == '/manifest':
            capture = query.get('capture', [''])[0]
            self.serve_manifest(capture)
        else:
            super().do_GET()

    def serve_viewer(self):
        """Serve the comparison viewer HTML."""
        html = """<!DOCTYPE html>
<html>
<head>
    <title>ORIGAMI Comparison Viewer</title>
    <style>
        * {
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 10px;
            background: #f5f5f5;
        }

        .header {
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        h1 {
            margin: 0 0 10px 0;
            color: #333;
            font-size: 24px;
        }

        /* Process Diagram */
        .process-diagram {
            background: white;
            padding: 20px;
            margin-top: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .pipeline {
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 20px 0;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 4px;
            overflow-x: auto;
        }

        .process-title {
            font-weight: bold;
            margin: 15px 0 5px 0;
            color: #495057;
            text-align: center;
            font-size: 14px;
        }

        .stage {
            display: flex;
            flex-direction: column;
            align-items: center;
            min-width: 90px;
            padding: 10px;
            margin: 0 5px;
            background: white;
            border: 2px solid #ddd;
            border-radius: 4px;
            font-size: 12px;
            text-align: center;
        }

        .stage.highlight {
            background: #e7f3ff;
            border-color: #007bff;
        }

        .stage-label {
            font-weight: bold;
            color: #333;
            margin-bottom: 5px;
            font-size: 14px;
        }

        .stage-desc {
            color: #666;
            font-size: 11px;
            line-height: 1.3;
        }

        .arrow {
            color: #007bff;
            font-size: 20px;
            margin: 0 5px;
        }

        .controls {
            display: flex;
            gap: 20px;
            align-items: center;
            flex-wrap: wrap;
        }

        .tile-selector {
            display: flex;
            gap: 10px;
            align-items: center;
        }

        .column-controls {
            display: flex;
            gap: 10px;
            align-items: center;
            margin-left: auto;
        }

        select {
            padding: 6px 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            background: white;
            font-size: 14px;
        }

        button {
            padding: 6px 12px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }

        button:hover {
            background: #0056b3;
        }

        button.remove-btn {
            background: #dc3545;
            padding: 4px 8px;
            font-size: 12px;
        }

        button.remove-btn:hover {
            background: #c82333;
        }

        .comparison-grid {
            background: white;
            padding: 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow: auto;
            position: relative;
            resize: both;
            min-width: 400px;
            min-height: 400px;
        }

        .grid-container {
            display: table;
            border-collapse: collapse;
            width: fit-content;
        }

        .grid-row {
            display: table-row;
        }

        .grid-cell {
            display: table-cell;
            padding: 0;
            text-align: center;
            vertical-align: middle;
            border: 1px solid #e0e0e0;
            background: white;
            position: relative;
        }

        .header-cell {
            background: #f8f9fa;
            font-weight: bold;
            font-size: 13px;
            padding: 8px 5px;
            position: sticky;
            top: 0;
            z-index: 10;
            min-width: var(--cell-size, 150px);
            height: 32px;
        }

        .row-label {
            background: #f8f9fa;
            font-weight: bold;
            font-size: 11px;
            text-align: left;
            padding: 8px 10px;
            position: sticky;
            left: 0;
            z-index: 5;
            min-width: 200px;
            height: var(--cell-size, 150px);
            display: flex;
            flex-direction: column;
            justify-content: center;
        }

        .row-label .stage-name {
            font-weight: bold;
            color: #007bff;
            margin-bottom: 3px;
        }

        .row-label .stage-desc {
            font-weight: normal;
            font-size: 10px;
            color: #6c757d;
            line-height: 1.3;
        }

        .image-cell {
            padding: 0;
            width: var(--cell-size, 150px);
            height: var(--cell-size, 150px);
        }

        .image-viewport {
            width: 100%;
            height: 100%;
            background: #f8f9fa;
            overflow: hidden;
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: grab;
            aspect-ratio: 1 / 1;
        }

        /* Resizable handle */
        .resize-handle {
            position: absolute;
            background: #007bff;
            opacity: 0;
            transition: opacity 0.2s;
            z-index: 20;
        }

        .resize-handle:hover {
            opacity: 0.5;
        }

        .resize-handle.horizontal {
            height: 4px;
            width: 100%;
            cursor: row-resize;
            bottom: -2px;
            left: 0;
        }

        .resize-handle.vertical {
            width: 4px;
            height: 100%;
            cursor: col-resize;
            right: -2px;
            top: 0;
        }

        .image-viewport.dragging {
            cursor: grabbing;
        }

        .image-viewport img {
            max-width: 100%;
            max-height: 100%;
            width: auto;
            height: auto;
            image-rendering: pixelated;
            image-rendering: -moz-crisp-edges;
            image-rendering: crisp-edges;
            display: block;
        }

        .no-image {
            color: #999;
            font-size: 11px;
            text-align: center;
            padding: 10px;
        }

        .metric-cell {
            padding: 10px;
            background: #f8f9fa;
            font-family: monospace;
            font-size: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            height: var(--cell-size, 150px);
        }

        .metric-value {
            font-weight: bold;
            color: #007bff;
            font-size: 16px;
        }

        .metric-unit {
            margin-left: 5px;
            color: #6c757d;
            font-size: 12px;
        }

        .metric-error {
            color: #dc3545;
            font-size: 12px;
        }

        .stage-label {
            font-size: 10px;
            color: #666;
            margin-top: 2px;
        }

        .column-header {
            position: relative;
            padding-right: 20px;
        }

        .remove-column {
            position: absolute;
            right: 5px;
            top: 5px;
            width: 16px;
            height: 16px;
            background: #dc3545;
            color: white;
            border: none;
            border-radius: 3px;
            cursor: pointer;
            font-size: 10px;
            line-height: 1;
            padding: 0;
        }

        .navigation {
            display: flex;
            justify-content: center;
            gap: 10px;
            margin-top: 15px;
        }

        .zoom-controls {
            display: flex;
            gap: 10px;
            align-items: center;
            justify-content: center;
            margin: 10px 0;
        }

        .zoom-btn {
            padding: 5px 10px;
            background: #f8f9fa;
            color: #333;
            border: 1px solid #ddd;
            border-radius: 3px;
            cursor: pointer;
            font-size: 12px;
        }

        .zoom-btn.active {
            background: #007bff;
            color: white;
            border-color: #007bff;
        }

        /* Zoom styles */
        .zoom-fit .image-viewport img {
            max-width: 100%;
            max-height: 100%;
        }

        .zoom-actual .image-viewport {
            width: 256px;
            height: 256px;
        }

        .zoom-actual .image-viewport img {
            width: 256px;
            height: 256px;
            max-width: none;
            max-height: none;
        }

        .zoom-2x .image-viewport {
            width: 200px;
            height: 200px;
        }

        .zoom-2x .image-viewport img {
            width: 512px;
            height: 512px;
            max-width: none;
            max-height: none;
        }

        .loading {
            color: #666;
            text-align: center;
            padding: 20px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>ORIGAMI Multi-Run Comparison Viewer</h1>

        <!-- Process Diagram -->
        <div class="process-diagram">
            <h2 style="text-align: center; color: #333;">ORIGAMI Compression &amp; Decompression Pipeline</h2>

            <div class="process-title">Compression Process</div>
            <div class="pipeline">
                <div class="stage highlight">
                    <div class="stage-label">A</div>
                    <div class="stage-desc">Original RGB Tile</div>
                </div>
                <span class="arrow">&rarr;</span>
                <div class="stage highlight">
                    <div class="stage-label">B</div>
                    <div class="stage-desc">RGB&rarr;YCbCr<br>Conversion</div>
                </div>
                <span class="arrow">&rarr;</span>
                <div class="stage">
                    <div class="stage-label">C</div>
                    <div class="stage-desc">L2 Parent<br>Upsampling</div>
                </div>
                <span class="arrow">&rarr;</span>
                <div class="stage highlight">
                    <div class="stage-label">D</div>
                    <div class="stage-desc">Residual<br>Y_orig - Y_pred</div>
                </div>
                <span class="arrow">&rarr;</span>
                <div class="stage highlight">
                    <div class="stage-label">E</div>
                    <div class="stage-desc">JPEG Encode<br>Quality J</div>
                </div>
                <span class="arrow">&rarr;</span>
                <div class="stage">
                    <div class="stage-label">G</div>
                    <div class="stage-desc">Stored<br>Residual</div>
                </div>
            </div>

            <div class="process-title">Decompression Process</div>
            <div class="pipeline">
                <div class="stage">
                    <div class="stage-label">G</div>
                    <div class="stage-desc">Stored<br>Residual</div>
                </div>
                <span class="arrow">&rarr;</span>
                <div class="stage">
                    <div class="stage-label">H</div>
                    <div class="stage-desc">JPEG Decode</div>
                </div>
                <span class="arrow">&rarr;</span>
                <div class="stage">
                    <div class="stage-label">I</div>
                    <div class="stage-desc">De-center<br>R - 128</div>
                </div>
                <span class="arrow">&rarr;</span>
                <div class="stage highlight">
                    <div class="stage-label">J</div>
                    <div class="stage-desc">Reconstruct<br>Y + R_q</div>
                </div>
                <span class="arrow">&rarr;</span>
                <div class="stage highlight">
                    <div class="stage-label">K</div>
                    <div class="stage-desc">YCbCr&rarr;RGB<br>Conversion</div>
                </div>
            </div>

            <div style="margin-top: 10px; font-size: 11px; color: #6c757d; text-align: center;">
                <strong>Key:</strong> J = JPEG quality &bull; L2 = lowest resolution &bull; L1 = medium &bull; L0 = highest
            </div>
        </div>

        <div class="controls">
            <div class="tile-selector">
                <label for="tileSelect">Tile:</label>
                <select id="tileSelect">
                    <option value="">Loading...</option>
                </select>
                <span id="tileInfo" style="font-size: 14px; color: #666;"></span>
            </div>

            <div class="column-controls">
                <label>Add Column:</label>
                <select id="addColumn">
                    <option value="">Select Q/J...</option>
                </select>
                <button onclick="addSelectedColumn()">Add</button>
                <button onclick="loadAllJpegQualities()" title="Load all JPEG qualities for current quantization">Load All J</button>
                <button onclick="loadAllQuantizations()" title="Load all quantizations for current JPEG quality">Load All Q</button>
            </div>
        </div>

        <div class="zoom-controls">
            <label style="font-size: 14px;">Zoom:</label>
            <button class="zoom-btn active" data-zoom="fit" onclick="setZoom('fit')">Fit</button>
            <button class="zoom-btn" data-zoom="actual" onclick="setZoom('actual')">Actual (1:1)</button>
            <button class="zoom-btn" data-zoom="2x" onclick="setZoom('2x')">2x</button>
        </div>
    </div>

    <div class="comparison-grid" id="comparisonGrid">
        <div class="loading">Loading captures...</div>
    </div>

    <div class="navigation">
        <button onclick="previousTile()">&larr; Previous Tile</button>
        <button onclick="nextTile()">Next Tile &rarr;</button>
    </div>

    <script>
        let captures = {};
        let activeColumns = [];
        let availableTiles = [];
        let currentTileIndex = 0;
        let currentZoom = 'fit';
        let globalScale = 1;
        let globalPanX = 0;
        let globalPanY = 0;
        let isDragging = false;
        let dragStartX = 0;
        let dragStartY = 0;

        // Image stages to display as rows
        const imageStages = [
            {
                id: 'original_rgb',
                label: 'Original RGB',
                stage: 'Stage A',
                desc: 'Input 256×256 RGB tile at original resolution',
                type: 'image',
                file: 'original'
            },
            {
                id: 'l2_parent_original',
                label: 'L2 Parent Tile',
                stage: 'Before C',
                desc: 'Lower resolution parent tile (256×256) covering 4× L1 area',
                type: 'l2_parent',  // Special type to load L2_0_0
                file: 'original'
            },
            {
                id: 'l2_parent_upsampled',
                label: 'Prediction Source',
                stage: 'Stage C',
                desc: 'Upsampled prediction for residual calculation (256×256 extracted from larger mosaic)',
                type: 'image',
                file: 'prediction'  // This is the actual prediction used for residual calculation
            },
            {
                id: 'y_original',
                label: 'Y (Luma) Original',
                stage: 'Stage B',
                desc: 'Luma channel extracted from original RGB after YCbCr conversion',
                type: 'image',
                file: 'luma'
            },
            {
                id: 'cb_chroma',
                label: 'Cb (Blue Chroma) Prediction',
                stage: 'From Stage C',
                desc: 'Blue-difference chroma predicted from L2 parent (reused in reconstruction)',
                type: 'image',
                file: 'chroma_cb'
            },
            {
                id: 'cr_chroma',
                label: 'Cr (Red Chroma) Prediction',
                stage: 'From Stage C',
                desc: 'Red-difference chroma predicted from L2 parent (reused in reconstruction)',
                type: 'image',
                file: 'chroma_cr'
            },
            {
                id: 'residual_raw',
                label: 'Y Residual Raw',
                stage: 'Stage D',
                desc: 'Difference between original and predicted luma: R = Y - Y_pred',
                type: 'image',
                file: 'residual_raw'
            },
            {
                id: 'residual_encoded',
                label: 'Encoded Residual',
                stage: 'Stage F',
                desc: 'Quantized residual centered to [0,255] and JPEG compressed',
                type: 'image',
                file: 'residual_centered'
            },
            {
                id: 'jpeg_size',
                label: 'JPEG Residual Size',
                stage: 'Metric',
                desc: 'Size of compressed luminance residual JPEG file',
                type: 'metric',
                metric: 'jpeg_size'
            },
            {
                id: 'reconstructed_rgb',
                label: 'Reconstructed RGB',
                stage: 'Stage K',
                desc: 'Final reconstructed tile after decompression: Y_recon + predicted chroma → RGB',
                type: 'image',
                file: 'reconstructed'  // Changed from 'prediction' to 'reconstructed'
            },
            {
                id: 'final_size',
                label: 'Final Tile Size',
                stage: 'Metric',
                desc: 'Total storage size for this tile (residual + metadata)',
                type: 'metric',
                metric: 'final_size'
            },
            {
                id: 'compression_ratio',
                label: 'Compression Ratio',
                stage: 'Metric',
                desc: 'Ratio compared to uncompressed RGB (196KB)',
                type: 'metric',
                metric: 'compression_ratio'
            },
            {
                id: 'pack_size',
                label: 'Family Pack Size',
                stage: 'Metric',
                desc: 'LZ4-compressed family pack: L0+L1 tiles bundled with byte offset index',
                type: 'metric',
                metric: 'pack_size'
            },
            {
                id: 'psnr',
                label: 'PSNR',
                stage: 'Metric',
                desc: 'Peak Signal-to-Noise Ratio (dB) compared to original',
                type: 'metric',
                metric: 'psnr'
            },
            {
                id: 'ssim',
                label: 'SSIM',
                stage: 'Metric',
                desc: 'Structural Similarity Index (0-1) compared to original',
                type: 'metric',
                metric: 'ssim'
            },
            {
                id: 'mse',
                label: 'MSE',
                stage: 'Metric',
                desc: 'Mean Squared Error (lower is better)',
                type: 'metric',
                metric: 'mse'
            },
            {
                id: 'vif',
                label: 'VIF',
                stage: 'Metric',
                desc: 'Visual Information Fidelity (0-1, higher is better)',
                type: 'metric',
                metric: 'vif'
            },
            {
                id: 'delta_e',
                label: 'Delta E',
                stage: 'Metric',
                desc: 'CIE Delta E color difference (lower is better)',
                type: 'metric',
                metric: 'delta_e'
            }
        ];

        // Cell size variable for resizing
        let cellSize = 150;

        async function loadCaptures() {
            try {
                const response = await fetch('/captures.json');
                captures = await response.json();

                // Initialize with just the first available column
                const firstCapture = Object.keys(captures).sort()[0];
                if (firstCapture) {
                    activeColumns = [firstCapture];
                }

                // Populate add column dropdown
                updateAddColumnDropdown();

                // Build tile list
                buildTileList();

                // Load first tile
                if (availableTiles.length > 0) {
                    loadTile(0);
                }
            } catch (error) {
                console.error('Failed to load captures:', error);
            }
        }

        function buildTileList() {
            availableTiles = [];

            // Add L2 tile (parent)
            availableTiles.push({id: 'L2_0_0', level: 'L2', x: 0, y: 0});

            // Add L1 tiles
            for (let y = 0; y < 2; y++) {
                for (let x = 0; x < 2; x++) {
                    availableTiles.push({id: `L1_${x}_${y}`, level: 'L1', x, y});
                }
            }

            // Add L0 tiles
            for (let y = 0; y < 4; y++) {
                for (let x = 0; x < 4; x++) {
                    availableTiles.push({id: `L0_${x}_${y}`, level: 'L0', x, y});
                }
            }

            // Update tile selector
            const select = document.getElementById('tileSelect');
            select.innerHTML = '';
            availableTiles.forEach((tile, index) => {
                const option = document.createElement('option');
                option.value = index;
                option.textContent = `${tile.id} (${tile.level} at ${tile.x},${tile.y})`;
                select.appendChild(option);
            });
        }

        function updateAddColumnDropdown() {
            const select = document.getElementById('addColumn');
            select.innerHTML = '<option value="">Select Q/J...</option>';

            Object.keys(captures).forEach(key => {
                if (!activeColumns.includes(key)) {
                    const option = document.createElement('option');
                    option.value = key;
                    option.textContent = key.replace('_', '/');
                    select.appendChild(option);
                }
            });
        }

        async function loadTile(index) {
            if (index < 0 || index >= availableTiles.length) return;

            currentTileIndex = index;
            const tile = availableTiles[index];

            document.getElementById('tileSelect').value = index;
            document.getElementById('tileInfo').textContent =
                `Level: ${tile.level}, Position: (${tile.x}, ${tile.y})`;

            // Build comparison grid
            buildComparisonGrid(tile);
        }

        function buildComparisonGrid(tile) {
            const grid = document.getElementById('comparisonGrid');

            let html = '<div class="grid-container">';

            // Header row
            html += '<div class="grid-row">';
            html += '<div class="grid-cell header-cell">Stage</div>';
            html += '<div class="grid-cell header-cell">Original</div>';

            activeColumns.forEach(col => {
                html += `<div class="grid-cell header-cell column-header">
                    ${col.replace('_', '/')}
                    <button class="remove-column" onclick="removeColumn('${col}')">&times;</button>
                </div>`;
            });
            html += '</div>';

            // Data rows for each image stage
            imageStages.forEach(stage => {
                html += '<div class="grid-row">';
                html += `<div class="grid-cell row-label">
                    <div class="stage-name">${stage.label} (${stage.stage})</div>
                    <div class="stage-desc">${stage.desc}</div>
                </div>`;

                if (stage.type === 'metric') {
                    // Metric row - display text/numbers
                    // Original column for metrics (show baseline sizes)
                    html += '<div class="grid-cell image-cell">';
                    html += `<div class="metric-cell" id="orig_${stage.id}">`;
                    if (stage.metric === 'compression_ratio') {
                        html += '<span class="metric-value">1:1</span>';
                    } else if (stage.metric === 'pack_size') {
                        html += '<span class="metric-value">N/A</span>';
                    } else if (stage.metric === 'jpeg_size' || stage.metric === 'final_size') {
                        // Calculate baseline sizes based on typical JPEG Q95 compression
                        // For 256x256 RGB image at Q95: approximately 35-40KB
                        // Based on actual measurements from ORIGAMI paper
                        const baselinePerTileL0 = 27; // KB for L0 tiles at Q95
                        const baselinePerTileL1 = 27.5; // KB for L1 tiles at Q95
                        const baselinePerTileL2 = 28.5; // KB for L2 tile at Q95

                        let totalKB = 0;
                        if (stage.metric === 'jpeg_size') {
                            // Per-tile baseline size
                            if (tile.level === 'L0') {
                                totalKB = baselinePerTileL0;
                            } else if (tile.level === 'L1') {
                                totalKB = baselinePerTileL1;
                            }
                        } else if (stage.metric === 'final_size') {
                            // Show actual size for the specific tile level we're viewing
                            if (tile.level === 'L0') {
                                totalKB = baselinePerTileL0;
                            } else if (tile.level === 'L1') {
                                totalKB = baselinePerTileL1;
                            } else if (tile.level === 'L2') {
                                totalKB = baselinePerTileL2;
                            }
                        }
                        html += `<span class="metric-value">${totalKB.toFixed(1)}</span><span class="metric-unit">KB</span>`;
                    } else if (stage.metric === 'psnr') {
                        html += '<span class="metric-value">&infin;</span><span class="metric-unit">dB</span>';
                    } else if (stage.metric === 'ssim') {
                        html += '<span class="metric-value">1.0000</span>';
                    } else if (stage.metric === 'mse') {
                        html += '<span class="metric-value">0.0</span>';
                    } else if (stage.metric === 'vif') {
                        html += '<span class="metric-value">1.0000</span>';
                    } else if (stage.metric === 'delta_e') {
                        html += '<span class="metric-value">0.00</span>';
                    } else {
                        html += '<span class="metric-value">Baseline</span>';
                    }
                    html += '</div>';
                    html += '</div>';

                    // Capture columns for metrics
                    activeColumns.forEach(col => {
                        html += '<div class="grid-cell image-cell">';
                        html += `<div class="metric-cell" id="${col}_${stage.id}">`;
                        html += '<span class="metric-value">Loading...</span>';
                        html += '</div>';
                        html += '</div>';
                    });
                } else {
                    // Image row - display images
                    // Original column
                    html += '<div class="grid-cell image-cell">';
                    html += '<div class="image-viewport">';
                    html += `<div id="orig_${stage.id}" class="loading">...</div>`;
                    html += '</div>';
                    html += '</div>';

                    // Capture columns
                    activeColumns.forEach(col => {
                        html += '<div class="grid-cell image-cell">';
                        html += '<div class="image-viewport">';
                        html += `<div id="${col}_${stage.id}" class="loading">...</div>`;
                        html += '</div>';
                        html += '</div>';
                    });
                }

                html += '</div>';
            });

            html += '</div>';
            grid.innerHTML = html;

            // Load images for all columns
            loadTileImages(tile);

            // Setup interactions after a short delay to ensure images are in DOM
            setTimeout(setupImageInteractions, 100);
        }

        async function loadTileImages(tile) {
            // Load original column - always use Q256_J90 or highest quality available as reference
            // Find the highest quality capture to use as reference for original
            let referenceCapture = null;

            // Priority order: Q256_J90, Q32_J50, then highest JPEG quality
            if (captures['Q256_J90']) {
                referenceCapture = 'Q256_J90';
            } else if (captures['Q32_J50']) {
                referenceCapture = 'Q32_J50';
            } else {
                // Find highest quality JPEG baseline
                const jpegQualities = Object.keys(captures)
                    .filter(k => k.startsWith('JPEG_Q'))
                    .map(k => ({ key: k, quality: parseInt(k.replace('JPEG_Q', '')) }))
                    .sort((a, b) => b.quality - a.quality);
                if (jpegQualities.length > 0) {
                    referenceCapture = jpegQualities[0].key;
                }
            }

            if (referenceCapture && captures[referenceCapture]) {
                imageStages.forEach(stage => {
                    if (stage.type !== 'metric') {
                        loadStageImage('orig', stage, tile, referenceCapture, true);
                    }
                });
            }

            // Load each capture column
            activeColumns.forEach(col => {
                if (captures[col]) {
                    imageStages.forEach(stage => {
                        if (stage.type === 'metric') {
                            loadMetric(col, stage, tile, col);
                        } else {
                            loadStageImage(col, stage, tile, col, false);
                        }
                    });
                }
            });
        }

        async function loadMetric(columnId, stage, tile, captureKey) {
            const container = document.getElementById(`${columnId}_${stage.id}`);
            if (!container) return;

            try {
                // Try to load from analysis manifest
                const capture = captures[captureKey];
                const manifestPath = `/manifest?capture=${captureKey}`;
                const response = await fetch(manifestPath);

                if (response.ok) {
                    const manifest = await response.json();
                    let value = null;
                    let html = '';

                    // Check if this is a JPEG baseline capture
                    const isJpegBaseline = capture.type === 'jpeg_baseline' || manifest.type === 'jpeg_baseline';

                    // Helper to read ORIGAMI metrics from decompression_phase
                    function getOrigamiMetric(manifest, tile, metricName) {
                        const level = tile.level;
                        const coords = tile.id.split('_').slice(1).join('_');
                        const tileKey = `tile_${coords}`;
                        if (manifest.decompression_phase && manifest.decompression_phase[level]) {
                            const tileData = manifest.decompression_phase[level][tileKey];
                            if (tileData && tileData[metricName] !== undefined) {
                                return tileData[metricName];
                            }
                        }
                        return null;
                    }

                    // Different metrics come from different parts of the manifest
                    switch(stage.metric) {
                        case 'jpeg_size':
                            if (isJpegBaseline) {
                                // For JPEG baselines, this is the baseline reference size
                                // Show "Baseline" for the baseline column
                                html = `<span class="metric-value">Baseline</span>`;
                            } else {
                                // For ORIGAMI, get residual size from size_comparison section
                                if (manifest.size_comparison) {
                                    if (tile.level === 'L0') {
                                        value = manifest.size_comparison.origami_L0_residuals;
                                    } else if (tile.level === 'L1') {
                                        value = manifest.size_comparison.origami_L1_residuals;
                                    }
                                    if (value) {
                                        // This is total for all tiles at that level, divide by number of tiles
                                        const tilesAtLevel = tile.level === 'L0' ? 16 : 4;
                                        const bytesPerTile = value / tilesAtLevel;
                                        const kb = (bytesPerTile / 1024).toFixed(2);
                                        html = `<span class="metric-value">${kb}</span><span class="metric-unit">KB</span>`;
                                    }
                                }
                            }
                            break;
                        case 'final_size':
                            if (isJpegBaseline) {
                                // For JPEG baselines, get tile size directly
                                if (manifest.tiles && manifest.tiles[tile.id]) {
                                    value = manifest.tiles[tile.id].size_bytes;
                                    const kb = (value / 1024).toFixed(2);
                                    html = `<span class="metric-value">${kb}</span><span class="metric-unit">KB</span>`;
                                }
                            } else {
                                // For ORIGAMI, get total ORIGAMI bytes
                                if (manifest.size_comparison && manifest.size_comparison.origami_total) {
                                    value = manifest.size_comparison.origami_total;
                                    const tilesTotal = 20; // 16 L0 + 4 L1
                                    const bytesPerTile = value / tilesTotal;
                                    const kb = (bytesPerTile / 1024).toFixed(2);
                                    html = `<span class="metric-value">${kb}</span><span class="metric-unit">KB</span>`;
                                }
                            }
                            break;
                        case 'compression_ratio':
                            if (isJpegBaseline) {
                                // For JPEG baselines, get from statistics
                                if (manifest.statistics && manifest.statistics.compression_ratio) {
                                    value = manifest.statistics.compression_ratio;
                                    html = `<span class="metric-value">${value.toFixed(1)}:1</span>`;
                                }
                            } else {
                                // For ORIGAMI, get from size_comparison
                                if (manifest.size_comparison && manifest.size_comparison.overall_compression_ratio) {
                                    value = manifest.size_comparison.overall_compression_ratio;
                                    html = `<span class="metric-value">${value.toFixed(1)}:1</span>`;
                                }
                            }
                            break;
                        case 'pack_size':
                            if (isJpegBaseline) {
                                // JPEG baseline: read from manifest.pack.size
                                if (manifest.pack && manifest.pack.size) {
                                    value = manifest.pack.size;
                                    const kb = (value / 1024).toFixed(1);
                                    html = `<span class="metric-value">${kb}</span><span class="metric-unit">KB</span>`;
                                }
                            } else {
                                // ORIGAMI: read from manifest.pac_file.size
                                if (manifest.pac_file && manifest.pac_file.size) {
                                    value = manifest.pac_file.size;
                                    const kb = (value / 1024).toFixed(1);
                                    html = `<span class="metric-value">${kb}</span><span class="metric-unit">KB</span>`;
                                }
                            }
                            break;
                        case 'psnr':
                            if (isJpegBaseline) {
                                // For JPEG baselines, get from tiles object
                                if (manifest.tiles && manifest.tiles[tile.id]) {
                                    value = manifest.tiles[tile.id].psnr;
                                    if (value) {
                                        html = `<span class="metric-value">${value.toFixed(2)}</span><span class="metric-unit">dB</span>`;
                                    }
                                }
                            } else {
                                value = getOrigamiMetric(manifest, tile, 'final_psnr');
                                if (value !== null) {
                                    html = `<span class="metric-value">${value.toFixed(2)}</span><span class="metric-unit">dB</span>`;
                                }
                            }
                            break;
                        case 'ssim':
                            if (isJpegBaseline) {
                                // For JPEG baselines, get from tiles object
                                if (manifest.tiles && manifest.tiles[tile.id]) {
                                    value = manifest.tiles[tile.id].ssim;
                                    if (value) {
                                        html = `<span class="metric-value">${value.toFixed(4)}</span>`;
                                    }
                                }
                            } else {
                                value = getOrigamiMetric(manifest, tile, 'final_ssim');
                                if (value !== null) {
                                    html = `<span class="metric-value">${value.toFixed(4)}</span>`;
                                }
                            }
                            break;
                        case 'mse':
                            if (isJpegBaseline) {
                                // For JPEG baselines, get from tiles object
                                if (manifest.tiles && manifest.tiles[tile.id]) {
                                    value = manifest.tiles[tile.id].mse;
                                    if (value) {
                                        html = `<span class="metric-value">${value.toFixed(1)}</span>`;
                                    }
                                }
                            } else {
                                value = getOrigamiMetric(manifest, tile, 'final_mse');
                                if (value !== null) {
                                    html = `<span class="metric-value">${value.toFixed(1)}</span>`;
                                }
                            }
                            break;
                        case 'vif':
                            // VIF requires sewar library to be installed
                            if (isJpegBaseline) {
                                // For JPEG baselines, get from tiles object
                                if (manifest.tiles && manifest.tiles[tile.id]) {
                                    value = manifest.tiles[tile.id].vif;
                                    if (value) {
                                        html = `<span class="metric-value">${value.toFixed(4)}</span>`;
                                    }
                                }
                            } else {
                                value = getOrigamiMetric(manifest, tile, 'final_vif');
                                if (value !== null) {
                                    html = `<span class="metric-value">${value.toFixed(4)}</span>`;
                                }
                            }
                            break;
                        case 'delta_e':
                            if (isJpegBaseline) {
                                // For JPEG baselines, get from tiles object
                                if (manifest.tiles && manifest.tiles[tile.id]) {
                                    value = manifest.tiles[tile.id].delta_e;
                                    if (value) {
                                        html = `<span class="metric-value">${value.toFixed(2)}</span>`;
                                    }
                                }
                            } else {
                                value = getOrigamiMetric(manifest, tile, 'final_delta_e');
                                if (value !== null) {
                                    html = `<span class="metric-value">${value.toFixed(2)}</span>`;
                                }
                            }
                            break;
                    }

                    if (html) {
                        container.innerHTML = html;
                    } else {
                        container.innerHTML = '<span class="metric-error">N/A</span>';
                    }
                } else {
                    container.innerHTML = '<span class="metric-error">No data</span>';
                }
            } catch (error) {
                console.error('Error loading metric:', error);
                container.innerHTML = '<span class="metric-error">Error</span>';
            }
        }

        function loadStageImage(columnId, stage, tile, captureKey, isOriginal) {
            const container = document.getElementById(`${columnId}_${stage.id}`);
            if (!container) return;

            const capture = captures[captureKey];
            let imagePath = null;

            // Special case: Show original RGB in the Original column for reconstructed stage
            if (isOriginal && stage.id === 'reconstructed_rgb') {
                // Show the original RGB tile for comparison
                imagePath = `/image/${tile.id}_original.png?capture=${captureKey}`;
            }
            // Special handling for L2 parent types
            else if (stage.type === 'l2_parent') {
                // Show the L2 parent tile (before upsampling)
                const parentX = tile.level === 'L2' ? tile.x : Math.floor(tile.x / (tile.level === 'L0' ? 4 : 2));
                const parentY = tile.level === 'L2' ? tile.y : Math.floor(tile.y / (tile.level === 'L0' ? 4 : 2));
                imagePath = `/image/L2_${parentX}_${parentY}_${stage.file}.png?capture=${captureKey}`;
            } else if (stage.type === 'image') {
                imagePath = `/image/${tile.id}_${stage.file}.png?capture=${captureKey}`;
            } else if (stage.type === 'array') {
                imagePath = `/array/${tile.id}_${stage.file}.npy?capture=${captureKey}`;
            }

            if (imagePath) {
                // For non-original columns showing original data, skip
                if (!isOriginal && (stage.id === 'original_rgb' || stage.id === 'y_original' || stage.id === 'l2_parent_original')) {
                    // Use original from first column
                    return;
                }

                const img = document.createElement('img');
                img.src = imagePath;
                img.onerror = () => {
                    container.innerHTML = '<div class="no-image">N/A</div>';
                };
                container.innerHTML = '';
                container.appendChild(img);
            } else {
                container.innerHTML = '<div class="no-image">N/A</div>';
            }
        }

        function addSelectedColumn() {
            const select = document.getElementById('addColumn');
            const col = select.value;

            if (col && !activeColumns.includes(col)) {
                activeColumns.push(col);
                activeColumns.sort();
                updateAddColumnDropdown();
                loadTile(currentTileIndex);
            }
        }

        function removeColumn(col) {
            const index = activeColumns.indexOf(col);
            if (index > -1 && activeColumns.length > 1) {
                activeColumns.splice(index, 1);
                updateAddColumnDropdown();
                loadTile(currentTileIndex);
            }
        }

        function previousTile() {
            if (currentTileIndex > 0) {
                loadTile(currentTileIndex - 1);
            }
        }

        function nextTile() {
            if (currentTileIndex < availableTiles.length - 1) {
                loadTile(currentTileIndex + 1);
            }
        }

        function setZoom(zoom) {
            currentZoom = zoom;
            document.body.className = `zoom-${zoom}`;

            // Reset transform values when changing zoom mode
            globalScale = 1;
            globalPanX = 0;
            globalPanY = 0;

            // Update all images
            applyTransform();

            // Update buttons
            document.querySelectorAll('.zoom-btn').forEach(btn => {
                btn.classList.toggle('active', btn.dataset.zoom === zoom);
            });
        }

        function applyTransform() {
            document.querySelectorAll('.image-viewport img').forEach(img => {
                if (currentZoom === 'fit') {
                    img.style.transform = '';
                } else {
                    img.style.transform = `translate(${globalPanX}px, ${globalPanY}px) scale(${globalScale})`;
                }
            });
        }

        function setupImageInteractions() {
            // Remove existing viewport listeners by replacing elements
            document.querySelectorAll('.image-viewport').forEach(viewport => {
                const clone = viewport.cloneNode(true);
                viewport.parentNode.replaceChild(clone, viewport);
            });

            // Add fresh listeners to all viewports
            document.querySelectorAll('.image-viewport').forEach(viewport => {
                // Scroll zoom with controlled increments
                viewport.addEventListener('wheel', (e) => {
                    if (currentZoom === 'fit') return;
                    e.preventDefault();
                    e.stopPropagation();

                    // Use smaller increment for smoother zooming
                    const zoomSpeed = 0.05;  // Even smaller for more control
                    const delta = e.deltaY > 0 ? (1 - zoomSpeed) : (1 + zoomSpeed);

                    globalScale *= delta;
                    globalScale = Math.max(0.5, Math.min(globalScale, 5));

                    applyTransform();
                }, { passive: false });

                // Mouse drag start
                viewport.addEventListener('mousedown', (e) => {
                    if (currentZoom === 'fit') return;
                    isDragging = true;
                    dragStartX = e.clientX - globalPanX;
                    dragStartY = e.clientY - globalPanY;
                    viewport.classList.add('dragging');
                    e.preventDefault();
                });
            });

            // Setup resize functionality
            setupResizeHandles();
        }

        function setupResizeHandles() {
            const grid = document.getElementById('comparisonGrid');
            let isResizing = false;
            let startX, startY, startSize;

            // Create a resize handle
            const resizeHandle = document.createElement('div');
            resizeHandle.style.position = 'fixed';
            resizeHandle.style.width = '20px';
            resizeHandle.style.height = '20px';
            resizeHandle.style.background = '#007bff';
            resizeHandle.style.cursor = 'nwse-resize';
            resizeHandle.style.opacity = '0.5';
            resizeHandle.style.display = 'none';
            resizeHandle.style.zIndex = '1000';
            document.body.appendChild(resizeHandle);

            // Position resize handle at bottom-right corner of any cell
            grid.addEventListener('mousemove', (e) => {
                if (isResizing) return;

                const cell = e.target.closest('.image-cell');
                if (cell) {
                    const rect = cell.getBoundingClientRect();
                    const distX = e.clientX - (rect.right - 20);
                    const distY = e.clientY - (rect.bottom - 20);

                    if (distX > 0 && distX < 20 && distY > 0 && distY < 20) {
                        resizeHandle.style.display = 'block';
                        resizeHandle.style.left = (rect.right - 20) + 'px';
                        resizeHandle.style.top = (rect.bottom - 20) + 'px';
                    } else {
                        resizeHandle.style.display = 'none';
                    }
                } else {
                    resizeHandle.style.display = 'none';
                }
            });

            // Start resizing
            resizeHandle.addEventListener('mousedown', (e) => {
                isResizing = true;
                startX = e.clientX;
                startY = e.clientY;
                startSize = cellSize;
                e.preventDefault();
                document.body.style.cursor = 'nwse-resize';
            });

            // Resize
            document.addEventListener('mousemove', (e) => {
                if (!isResizing) return;

                const deltaX = e.clientX - startX;
                const deltaY = e.clientY - startY;
                const delta = Math.max(deltaX, deltaY); // Use max to maintain square aspect

                cellSize = Math.max(100, Math.min(400, startSize + delta));
                document.documentElement.style.setProperty('--cell-size', cellSize + 'px');

                // Reapply transform after resize
                applyTransform();
            });

            // Stop resizing
            document.addEventListener('mouseup', () => {
                if (isResizing) {
                    isResizing = false;
                    document.body.style.cursor = '';
                    resizeHandle.style.display = 'none';
                }
            });
        }

        function loadAllJpegQualities() {
            if (activeColumns.length === 0) return;

            // Get the quantization from the first active column
            const firstCol = activeColumns[0];
            const match = firstCol.match(/Q(\\d+)_J(\\d+)/);
            if (!match) return;

            const targetQ = match[1];

            // Find all columns with same Q value
            const newColumns = Object.keys(captures).filter(key => {
                const m = key.match(/Q(\\d+)_J(\\d+)/);
                return m && m[1] === targetQ;
            }).sort();

            if (newColumns.length > 0) {
                activeColumns = newColumns;
                updateAddColumnDropdown();
                loadTile(currentTileIndex);
            }
        }

        function loadAllQuantizations() {
            if (activeColumns.length === 0) return;

            // Get the JPEG quality from the first active column
            const firstCol = activeColumns[0];
            const match = firstCol.match(/Q(\\d+)_J(\\d+)/);
            if (!match) return;

            const targetJ = match[2];

            // Find all columns with same J value
            const newColumns = Object.keys(captures).filter(key => {
                const m = key.match(/Q(\\d+)_J(\\d+)/);
                return m && m[2] === targetJ;
            }).sort();

            if (newColumns.length > 0) {
                activeColumns = newColumns;
                updateAddColumnDropdown();
                loadTile(currentTileIndex);
            }
        }

        // Setup global event handlers once
        document.addEventListener('mousemove', (e) => {
            if (!isDragging) return;
            e.preventDefault();

            globalPanX = e.clientX - dragStartX;
            globalPanY = e.clientY - dragStartY;

            applyTransform();
        });

        document.addEventListener('mouseup', () => {
            if (isDragging) {
                isDragging = false;
                document.querySelectorAll('.image-viewport.dragging').forEach(vp => {
                    vp.classList.remove('dragging');
                });
            }
        });

        // Tile selector change handler
        document.getElementById('tileSelect').addEventListener('change', function() {
            loadTile(parseInt(this.value));
        });

        // Initialize on load
        loadCaptures();
    </script>
</body>
</html>"""

        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))

    def serve_capture_list(self):
        """Serve JSON list of available captures."""
        # Convert Path objects to strings for JSON serialization
        captures_for_json = {}
        for key, capture in self.captures.items():
            if capture.get('type') == 'jpeg_baseline':
                # For JPEG baselines, use quality as both q and j for compatibility
                captures_for_json[key] = {
                    'q': capture['quality'],
                    'j': capture['quality'],
                    'name': capture['name'],
                    'type': 'jpeg_baseline',
                    'encoder': capture.get('encoder', 'libjpeg-turbo')
                }
            else:
                # For ORIGAMI captures
                captures_for_json[key] = {
                    'q': capture.get('q', 0),
                    'j': capture.get('j', 0),
                    'name': capture['name'],
                    'type': capture.get('type', 'origami'),
                    'encoder': capture.get('encoder', 'libjpeg-turbo')
                }

        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(captures_for_json).encode('utf-8'))

    def serve_image(self, image_path, capture_key):
        """Serve an image file from a specific capture."""
        if capture_key not in self.captures:
            self.send_error(404, f"Capture not found: {capture_key}")
            return

        capture = self.captures[capture_key]

        # Handle JPEG baseline captures
        if capture.get('type') == 'jpeg_baseline':
            # JPEG baselines only have original tiles in tiles/ directory
            if 'original' in image_path or 'reconstructed' in image_path or image_path.endswith('.jpg'):
                # Map to the actual JPEG file
                if image_path.endswith('.jpg'):
                    # Direct JPEG request (e.g., L0_0_0.jpg)
                    jpeg_path = capture['path'] / 'tiles' / image_path
                else:
                    # PNG request for original/reconstructed (e.g., L0_0_0_original.png)
                    tile_id = image_path.replace('_original.png', '').replace('_reconstructed.png', '')
                    jpeg_path = capture['path'] / 'tiles' / f'{tile_id}.jpg'

                if jpeg_path.exists():
                    with open(jpeg_path, 'rb') as f:
                        data = f.read()
                    self.send_response(200)
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
            # JPEG baselines don't have predictions, residuals, etc.
            self.send_error(404, f"JPEG baseline doesn't have: {image_path}")
            return

        # Handle ORIGAMI captures
        elif capture.get('has_images', False):
            # Old format: images/
            full_path = capture['path'] / "images" / image_path
        elif capture.get('has_compress', False) or capture.get('has_decompress', False):
            # New format: compress/ for compression phase, decompress/ for decompression
            # Uses dynamic glob to find numbered files like 017_L1_0_0_residual_centered.png
            compress_dir = capture['path'] / "compress"
            decompress_dir = capture['path'] / "decompress"

            base_name = image_path.replace('.png', '').replace('.jpg', '')

            # L2 files don't have coordinates in the filename (e.g., 001_L2_original.png)
            # but we request them as L2_0_0_original.png, so strip the coordinates
            l2_base_name = None
            if base_name.startswith('L2_'):
                parts = base_name.split('_')
                # L2_0_0_original -> L2_original
                if len(parts) >= 4:
                    l2_base_name = parts[0] + '_' + '_'.join(parts[3:])

            full_path = None

            # Try decompress directory first for reconstructed files
            if 'reconstructed' in image_path and decompress_dir.exists():
                matches = list(decompress_dir.glob(f"*_{base_name}.png"))
                if matches:
                    full_path = matches[0]

            # Try compress directory
            if full_path is None and compress_dir.exists():
                # Try exact name first (with coordinates)
                matches = list(compress_dir.glob(f"*_{base_name}.png"))
                if not matches:
                    matches = list(compress_dir.glob(f"*_{base_name}.jpg"))
                # Try L2 name without coordinates
                if not matches and l2_base_name:
                    matches = list(compress_dir.glob(f"*_{l2_base_name}.png"))
                    if not matches:
                        matches = list(compress_dir.glob(f"*_{l2_base_name}.jpg"))
                if matches:
                    full_path = matches[0]

            # Final fallback
            if full_path is None:
                full_path = compress_dir / image_path
        else:
            # Fallback to images directory
            full_path = capture['path'] / "images" / image_path

        # Check for L2 upsampled images (might need to generate)
        if 'L2' in image_path and 'upsampled' in image_path:
            # Try to find the L2 tile and serve it upsampled
            base_name = image_path.replace('_upsampled.png', '.png')
            if capture.get('has_images', False):
                base_path = capture['path'] / "images" / base_name.replace('L2', 'L1')
            else:
                base_path = capture['path'] / "compress" / base_name.replace('L2', 'L1')
            if base_path.exists():
                full_path = base_path

        if not full_path.exists():
            self.send_error(404, f"Image not found: {image_path}")
            return

        with open(full_path, 'rb') as f:
            data = f.read()

        self.send_response(200)
        self.send_header('Content-Type', 'image/png')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def serve_array(self, array_path, capture_key):
        """Convert numpy array to image and serve it."""
        if capture_key not in self.captures:
            self.send_error(404, f"Capture not found: {capture_key}")
            return

        capture = self.captures[capture_key]

        # Check both possible directory structures
        if capture.get('has_images', False):
            # Old format: arrays/
            full_path = capture['path'] / "arrays" / array_path
        else:
            # New format doesn't have arrays directory
            # Arrays are not available in the new format
            self.send_error(404, f"Arrays not available for this capture")
            return

        if not full_path.exists():
            self.send_error(404, f"Array not found: {array_path}")
            return

        try:
            # Load array
            arr = np.load(full_path)

            # Convert to uint8 for display
            if arr.dtype == np.float32 or arr.dtype == np.float64:
                if 'Cb' in array_path or 'Cr' in array_path:
                    arr = np.clip(arr + 128, 0, 255)
                elif 'residual' in array_path:
                    abs_max = np.abs(arr).max()
                    if abs_max > 0:
                        arr = (arr / abs_max) * 127 + 128
                    else:
                        arr = np.full_like(arr, 128)
                    arr = np.clip(arr, 0, 255)
                else:
                    arr = np.clip(arr, 0, 255)
                arr = arr.astype(np.uint8)

            # Create PIL image
            if len(arr.shape) == 3:
                img = Image.fromarray(arr)
            else:
                img = Image.fromarray(arr, mode='L')

            # Convert to bytes
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            data = buffer.getvalue()

            self.send_response(200)
            self.send_header('Content-Type', 'image/png')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        except Exception as e:
            self.send_error(500, f"Error processing array: {str(e)}")

    def serve_manifest(self, capture_key):
        """Serve the analysis manifest JSON file."""
        if capture_key not in self.captures:
            self.send_error(404, f"Capture not found: {capture_key}")
            return

        capture = self.captures[capture_key]
        manifest_path = capture['path'] / 'analysis_manifest.json'

        if not manifest_path.exists():
            # Try alternative paths
            manifest_path = capture['path'] / 'manifest.json'
            if not manifest_path.exists():
                self.send_error(404, "Manifest not found")
                return

        try:
            with open(manifest_path, 'r') as f:
                data = f.read()

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(data.encode('utf-8'))
        except Exception as e:
            print(f"Error serving manifest: {e}")
            self.send_error(500, "Failed to load manifest")


def run_server(paper_dir, port=8080):
    """Run the comparison viewer server."""
    paper_path = pathlib.Path(paper_dir)

    if not paper_path.exists():
        print(f"Error: Paper directory not found: {paper_dir}")
        return

    # Create handler with paper directory
    def handler_factory(*args, **kwargs):
        return ComparisonViewerHandler(*args, paper_dir=paper_dir, **kwargs)

    server = HTTPServer(('', port), handler_factory)

    print(f"ORIGAMI Comparison Viewer")
    print(f"Scanning directory: {paper_path}")
    print(f"Server running at: http://localhost:{port}")
    print(f"Press Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.shutdown()


def main():
    parser = argparse.ArgumentParser(description="ORIGAMI Comparison Viewer")
    parser.add_argument("--paper-dir", default="paper", help="Paper directory with debug captures")
    parser.add_argument("--port", type=int, default=8099, help="Server port (default: 8099)")

    args = parser.parse_args()

    run_server(args.paper_dir, args.port)


if __name__ == "__main__":
    main()