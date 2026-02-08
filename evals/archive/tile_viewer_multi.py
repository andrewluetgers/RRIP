#!/usr/bin/env python3
"""
Multi-capture tile viewer server for ORIGAMI debug tiles.
Allows selecting different Quantization (Q) and JPEG Quality (J) values.
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


class MultiCaptureHandler(SimpleHTTPRequestHandler):
    """Handler for serving multiple capture directories."""

    def __init__(self, *args, paper_dir=None, **kwargs):
        self.paper_dir = pathlib.Path(paper_dir) if paper_dir else pathlib.Path.cwd()
        self.captures = self.scan_captures()
        super().__init__(*args, **kwargs)

    def scan_captures(self):
        """Scan for all debug capture directories."""
        captures = {}

        # Pattern to match debug directories with timestamp format YYYYMMDD_HHMMSS
        # Handles: debug_L0-1024_q{Q}_j{J}_{YYYYMMDD}_{HHMMSS} format
        pattern = re.compile(r'debug_.*_q(\d+)_j(\d+)_\d+_\d+$')

        for dir_path in self.paper_dir.iterdir():
            if dir_path.is_dir() and dir_path.name.startswith('debug_'):
                match = pattern.match(dir_path.name)
                if match:
                    q_val = int(match.group(1))
                    j_val = int(match.group(2))

                    # Check if this capture has the necessary files
                    if (dir_path / 'images').exists() or (dir_path / 'arrays').exists():
                        if q_val not in captures:
                            captures[q_val] = {}
                        captures[q_val][j_val] = dir_path

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
        elif path == '/tiles.json':
            q = int(query.get('q', [32])[0])
            j = int(query.get('j', [30])[0])
            self.serve_tile_list(q, j)
        elif path.startswith('/image/'):
            q = int(query.get('q', [32])[0])
            j = int(query.get('j', [30])[0])
            self.serve_image(path[7:], q, j)
        elif path.startswith('/array/'):
            q = int(query.get('q', [32])[0])
            j = int(query.get('j', [30])[0])
            self.serve_array(path[7:], q, j)
        else:
            super().do_GET()

    def serve_viewer(self):
        """Serve the main viewer HTML with Q/J selection."""
        html = """<!DOCTYPE html>
<html>
<head>
    <title>ORIGAMI Multi-Capture Tile Viewer</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }

        .header {
            text-align: center;
            margin-bottom: 30px;
        }

        h1 {
            color: #333;
            margin: 0 0 10px 0;
        }

        .metadata {
            color: #666;
            font-size: 14px;
        }

        .controls {
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }

        .capture-selector {
            display: flex;
            align-items: center;
            gap: 20px;
            justify-content: center;
            margin-bottom: 15px;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 4px;
        }

        .tile-selector {
            display: flex;
            align-items: center;
            gap: 20px;
            justify-content: center;
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

        .zoom-btn:hover:not(.active) {
            background: #e9ecef;
            color: #000;
        }

        .zoom-value {
            font-size: 12px;
            color: #666;
            min-width: 60px;
            text-align: center;
        }

        select {
            padding: 8px 12px;
            font-size: 16px;
            border: 1px solid #ddd;
            border-radius: 4px;
            background: white;
        }

        .tile-info {
            color: #666;
            font-size: 14px;
        }

        .viewer {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .image-viewport {
            width: 100%;
            height: 300px;
            border: 1px solid #ddd;
            border-radius: 4px;
            background: #f8f9fa;
            overflow: hidden;
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: grab;
            user-select: none;
        }

        .image-viewport.dragging {
            cursor: grabbing;
        }

        .image-viewport img {
            image-rendering: pixelated;
            image-rendering: -moz-crisp-edges;
            image-rendering: crisp-edges;
            display: block;
            position: relative;
            transform-origin: center center;
            transition: none;
        }

        .image-viewport img.transitioning {
            transition: transform 0.2s ease;
        }

        /* Zoom styles */
        .zoom-fit .image-viewport img {
            position: relative;
            transform: none !important;
            max-width: 100%;
            max-height: 100%;
            width: auto;
            height: auto;
        }

        .zoom-actual .image-viewport img,
        .zoom-2x .image-viewport img,
        .zoom-4x .image-viewport img {
            position: absolute;
            left: 50%;
            top: 50%;
            transform: translate(-50%, -50%);
        }

        .zoom-actual .image-viewport img {
            width: 256px;
            height: 256px;
        }

        .zoom-2x .image-viewport img {
            width: 512px;
            height: 512px;
        }

        .zoom-4x .image-viewport img {
            width: 1024px;
            height: 1024px;
        }

        .channel-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
        }

        .comparison-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin-bottom: 20px;
        }

        .image-container {
            text-align: center;
        }

        .image-container h3 {
            margin: 0 0 5px 0;
            color: #555;
            font-size: 14px;
        }

        .no-image {
            width: 100%;
            max-width: 256px;
            aspect-ratio: 1;
            background: #f0f0f0;
            border: 1px solid #ddd;
            border-radius: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #999;
            font-size: 12px;
            margin: 0 auto;
        }

        .navigation {
            display: flex;
            justify-content: center;
            gap: 10px;
            margin-top: 20px;
        }

        button {
            padding: 8px 16px;
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

        button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }

        .loading {
            color: #666;
            text-align: center;
            padding: 20px;
        }

        .status-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #28a745;
            margin-left: 10px;
        }

        .status-indicator.error {
            background: #dc3545;
        }

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

        .stage {
            display: flex;
            flex-direction: column;
            align-items: center;
            min-width: 120px;
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
        }

        .stage-desc {
            color: #666;
            font-size: 11px;
        }

        .arrow {
            color: #007bff;
            font-size: 20px;
            margin: 0 5px;
        }

        .process-title {
            font-weight: bold;
            color: #333;
            margin-bottom: 10px;
            font-size: 14px;
        }

        .stage-indicator {
            display: inline-block;
            padding: 2px 6px;
            background: #e7f3ff;
            border: 1px solid #007bff;
            border-radius: 3px;
            font-size: 10px;
            color: #007bff;
            margin-left: 5px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>ORIGAMI Multi-Capture Tile Viewer</h1>
        <div class="metadata" id="metadata">
            Loading captures...
        </div>
    </div>

    <div class="controls">
        <div class="capture-selector">
            <label for="qSelect">Quantization (Q):</label>
            <select id="qSelect">
                <option value="">Loading...</option>
            </select>

            <label for="jSelect">JPEG Quality (J):</label>
            <select id="jSelect">
                <option value="">Loading...</option>
            </select>

            <button id="loadBtn" onclick="loadCapture()">Load Capture</button>
            <span class="status-indicator" id="status"></span>
        </div>

        <div class="tile-selector">
            <label for="tileSelect">Select Tile:</label>
            <select id="tileSelect">
                <option value="">No capture loaded</option>
            </select>
            <div class="tile-info" id="tileInfo"></div>
        </div>
    </div>

    <div class="viewer" id="viewer">
        <div class="zoom-controls">
            <label style="font-size: 14px; margin-right: 10px;">Zoom:</label>
            <button class="zoom-btn active" data-zoom="fit">Fit</button>
            <button class="zoom-btn" data-zoom="actual">Actual (1:1)</button>
            <button class="zoom-btn" data-zoom="2x">2x</button>
            <button class="zoom-btn" data-zoom="4x">4x</button>
            <span class="zoom-value" id="zoomValue">Fit</span>
        </div>

        <div id="mainContent" class="zoom-fit">
            <h2>Full Color Tiles</h2>
            <div class="comparison-grid">
                <div class="image-container">
                    <h3>Original RGB <span class="stage-indicator">Stage A</span></h3>
                    <div class="image-viewport">
                        <div id="original"></div>
                    </div>
                </div>
                <div class="image-container">
                    <h3>Reconstructed RGB <span class="stage-indicator">Stage K</span></h3>
                    <div class="image-viewport">
                        <div id="prediction"></div>
                    </div>
                </div>
            </div>

            <h2>YCbCr Channel Decomposition</h2>
            <div class="channel-grid">
                <div class="image-container">
                    <h3>Y (Luma) - Original <span class="stage-indicator">Stage B</span></h3>
                    <div class="image-viewport">
                        <div id="Y_ground_truth"></div>
                    </div>
                </div>
                <div class="image-container">
                    <h3>Y (Luma) - Reconstructed <span class="stage-indicator">Stage J</span></h3>
                    <div class="image-viewport">
                        <div id="Y_prediction"></div>
                    </div>
                </div>
                <div class="image-container">
                    <h3>Cb (Blue Chroma) <span class="stage-indicator">Stage J</span></h3>
                    <div class="image-viewport">
                        <div id="Cb"></div>
                    </div>
                </div>
                <div class="image-container">
                    <h3>Cr (Red Chroma) <span class="stage-indicator">Stage J</span></h3>
                    <div class="image-viewport">
                        <div id="Cr"></div>
                    </div>
                </div>
            </div>

            <h2>Residuals</h2>
            <div class="comparison-grid">
                <div class="image-container">
                    <h3>Y Residual <span class="stage-indicator">Stage D</span></h3>
                    <p style="font-size: 12px; color: #666;">Original - Prediction (Gray=0, Black=-ve, White=+ve)</p>
                    <div class="image-viewport">
                        <div id="residual_raw"></div>
                    </div>
                </div>
                <div class="image-container">
                    <h3>Encoded Residual <span class="stage-indicator">Stage F</span></h3>
                    <p style="font-size: 12px; color: #666;">JPEG compressed residual</p>
                    <div class="image-viewport">
                        <div id="residual_encoded"></div>
                    </div>
                </div>
            </div>
        </div>

        <div class="navigation">
            <button id="prevBtn" onclick="previousTile()">&larr; Previous</button>
            <button id="nextBtn" onclick="nextTile()">Next &rarr;</button>
        </div>
    </div>

    <div class="process-diagram">
        <h2 style="text-align: center; color: #333;">ORIGAMI Compression & Decompression Pipeline</h2>

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
            <div class="stage">
                <div class="stage-label">E</div>
                <div class="stage-desc">Quantization<br>Q-levels</div>
            </div>
            <span class="arrow">&rarr;</span>
            <div class="stage highlight">
                <div class="stage-label">F</div>
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
                <div class="stage-desc">L2 Parent<br>Upsampling</div>
            </div>
            <span class="arrow">&rarr;</span>
            <div class="stage highlight">
                <div class="stage-label">J</div>
                <div class="stage-desc">Add Residual<br>Y_pred + Residual</div>
            </div>
            <span class="arrow">&rarr;</span>
            <div class="stage highlight">
                <div class="stage-label">K</div>
                <div class="stage-desc">YCbCr&rarr;RGB<br>Conversion</div>
            </div>
            <span class="arrow">&rarr;</span>
            <div class="stage highlight">
                <div class="stage-label">K</div>
                <div class="stage-desc">Reconstructed<br>RGB Tile</div>
            </div>
        </div>

        <div style="margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 4px;">
            <p style="margin: 0; color: #666; font-size: 13px;">
                <strong>Key Points:</strong><br>
                &bull; <strong>Stages A &amp; K:</strong> Full RGB images (original vs reconstructed)<br>
                &bull; <strong>Stages B &amp; J:</strong> YCbCr color space (luma and chroma channels)<br>
                &bull; <strong>Stage C &amp; I:</strong> L2 parent tile upsampled 4&times; to create prediction<br>
                &bull; <strong>Stage D:</strong> Residual = Original - Prediction (only for Y/luma channel)<br>
                &bull; <strong>Stage F:</strong> Residual after JPEG compression with quality J<br>
                &bull; <strong>Chroma channels (Cb/Cr):</strong> Reused from prediction, not stored as residuals
            </p>
        </div>
    </div>

    <script>
        let captures = {};
        let tiles = [];
        let currentIndex = 0;
        let currentZoom = 'fit';
        let currentQ = null;
        let currentJ = null;

        async function loadCaptureList() {
            try {
                const response = await fetch('/captures.json');
                captures = await response.json();

                // Populate Q selector
                const qSelect = document.getElementById('qSelect');
                qSelect.innerHTML = '';

                const qValues = Object.keys(captures).map(Number).sort((a, b) => a - b);
                qValues.forEach(q => {
                    const option = document.createElement('option');
                    option.value = q;
                    option.textContent = `Q${q}`;
                    qSelect.appendChild(option);
                });

                if (qValues.length > 0) {
                    qSelect.value = qValues[0];
                    updateJValues();
                }

                document.getElementById('metadata').textContent =
                    `Found ${qValues.length} quantization levels`;

                return true;

            } catch (error) {
                console.error('Failed to load captures:', error);
                document.getElementById('metadata').textContent = 'Error loading captures';
                document.getElementById('status').classList.add('error');
                return false;
            }
        }

        function updateJValues() {
            const q = document.getElementById('qSelect').value;
            if (!q || !captures[q]) return;

            const jSelect = document.getElementById('jSelect');
            jSelect.innerHTML = '';

            const jValues = Object.keys(captures[q]).map(Number).sort((a, b) => a - b);
            jValues.forEach(j => {
                const option = document.createElement('option');
                option.value = j;
                option.textContent = `J${j}`;
                jSelect.appendChild(option);
            });

            if (jValues.length > 0) {
                jSelect.value = jValues[0];
            }
        }

        async function loadCapture() {
            const q = document.getElementById('qSelect').value;
            const j = document.getElementById('jSelect').value;

            if (!q || !j) {
                alert('Please select both Q and J values');
                return;
            }

            currentQ = q;
            currentJ = j;

            const status = document.getElementById('status');
            status.classList.remove('error');

            try {
                const response = await fetch(`/tiles.json?q=${q}&j=${j}`);
                tiles = await response.json();

                // Update metadata
                document.getElementById('metadata').textContent =
                    `Capture: Q${q} J${j} - ${tiles.capture_name || 'Unknown'}`;

                // Populate tile selector
                const select = document.getElementById('tileSelect');
                select.innerHTML = '';

                tiles.tiles.forEach((tile, index) => {
                    const option = document.createElement('option');
                    option.value = index;
                    option.textContent = `${tile.id} (${tile.level} at ${tile.x},${tile.y})`;
                    select.appendChild(option);
                });

                // Show first tile
                if (tiles.tiles.length > 0) {
                    showTile(0);
                }
            } catch (error) {
                console.error('Failed to load tiles:', error);
                document.getElementById('metadata').textContent = `Error loading Q${q} J${j}`;
                status.classList.add('error');
            }
        }

        function showTile(index) {
            if (index < 0 || index >= tiles.tiles.length) return;

            currentIndex = index;
            const tile = tiles.tiles[index];

            // Update selector
            document.getElementById('tileSelect').value = index;

            // Update info
            document.getElementById('tileInfo').textContent =
                `Level: ${tile.level}, Position: (${tile.x}, ${tile.y})`;

            // Update images with Q and J parameters
            const imageTypes = [
                {id: 'original', path: tile.images?.original},
                {id: 'prediction', path: tile.images?.prediction},
                {id: 'Y_ground_truth', path: tile.arrays?.Y_ground_truth},
                {id: 'Y_prediction', path: tile.arrays?.Y_prediction},
                {id: 'Cb', path: tile.arrays?.Cb},
                {id: 'Cr', path: tile.arrays?.Cr},
                {id: 'residual_raw', path: tile.arrays?.residual_raw},
                {id: 'residual_encoded', path: tile.images?.residual_encoded}
            ];

            imageTypes.forEach(({id, path}) => {
                const container = document.getElementById(id);
                if (!container) return;

                if (path) {
                    // Add Q and J parameters to the path
                    const urlPath = `${path}${path.includes('?') ? '&' : '?'}q=${currentQ}&j=${currentJ}`;
                    const img = document.createElement('img');
                    img.src = urlPath;
                    img.alt = id;
                    img.onerror = () => {
                        container.innerHTML = '<div class="no-image">Failed to load</div>';
                    };
                    container.innerHTML = '';
                    container.appendChild(img);
                } else {
                    container.innerHTML = '<div class="no-image">Not available</div>';
                }
            });

            // Update navigation buttons
            document.getElementById('prevBtn').disabled = currentIndex === 0;
            document.getElementById('nextBtn').disabled = currentIndex === tiles.tiles.length - 1;

            // Setup image interaction after images load
            setTimeout(setupImageInteraction, 100);
        }

        function setZoom(zoom) {
            currentZoom = zoom;
            const mainContent = document.getElementById('mainContent');

            // Remove all zoom classes
            mainContent.className = mainContent.className.replace(/zoom-\\w+/g, '');

            // Add new zoom class
            mainContent.classList.add(`zoom-${zoom}`);

            // Reset global transform when changing zoom
            resetGlobalTransform();

            // Update buttons
            document.querySelectorAll('.zoom-btn').forEach(btn => {
                btn.classList.toggle('active', btn.dataset.zoom === zoom);
            });

            // Update zoom value display
            const zoomText = {
                'fit': 'Fit',
                'actual': '100%',
                '2x': '200%',
                '4x': '400%'
            };
            document.getElementById('zoomValue').textContent = zoomText[zoom] || zoom;

            // Re-setup interactions for new zoom level
            setTimeout(setupImageInteraction, 100);
        }

        function previousTile() {
            if (currentIndex > 0) {
                showTile(currentIndex - 1);
            }
        }

        function nextTile() {
            if (currentIndex < tiles.tiles.length - 1) {
                showTile(currentIndex + 1);
            }
        }

        // Initialize
        document.getElementById('qSelect').addEventListener('change', updateJValues);

        document.getElementById('tileSelect').addEventListener('change', function() {
            showTile(parseInt(this.value));
        });

        // Zoom control handlers
        document.querySelectorAll('.zoom-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                setZoom(btn.dataset.zoom);
            });
        });

        // Global transform state for synchronized pan/zoom
        let globalTransform = {
            x: 0,
            y: 0,
            scale: 1
        };

        // Setup drag and zoom for image viewports with synchronization
        function setupImageInteraction() {
            let isDragging = false;
            let startX, startY;
            let draggedViewport = null;

            // Apply transform to all images
            function applyGlobalTransform() {
                document.querySelectorAll('.image-viewport img').forEach(img => {
                    if (currentZoom === 'fit') {
                        img.style.transform = '';
                    } else {
                        const baseTransform = `translate(-50%, -50%)`;
                        const panZoomTransform = `translate(${globalTransform.x}px, ${globalTransform.y}px) scale(${globalTransform.scale})`;
                        img.style.transform = `${baseTransform} ${panZoomTransform}`;
                    }
                });
            }

            document.querySelectorAll('.image-viewport').forEach(viewport => {
                // Mouse wheel zoom (synchronized)
                viewport.addEventListener('wheel', (e) => {
                    if (currentZoom === 'fit') return;
                    e.preventDefault();

                    const delta = e.deltaY > 0 ? 0.9 : 1.1;
                    globalTransform.scale *= delta;
                    globalTransform.scale = Math.max(0.5, Math.min(globalTransform.scale, 5));

                    applyGlobalTransform();
                });

                // Mouse drag start
                viewport.addEventListener('mousedown', (e) => {
                    if (currentZoom === 'fit') return;
                    isDragging = true;
                    draggedViewport = viewport;
                    viewport.classList.add('dragging');
                    startX = e.clientX - globalTransform.x;
                    startY = e.clientY - globalTransform.y;
                    e.preventDefault();
                });
            });

            // Global mouse move (for dragging)
            window.addEventListener('mousemove', (e) => {
                if (!isDragging) return;
                e.preventDefault();
                globalTransform.x = e.clientX - startX;
                globalTransform.y = e.clientY - startY;
                applyGlobalTransform();
            });

            // Global mouse up
            window.addEventListener('mouseup', () => {
                if (draggedViewport) {
                    draggedViewport.classList.remove('dragging');
                    draggedViewport = null;
                }
                isDragging = false;
            });

            // Initial application
            applyGlobalTransform();
        }

        // Reset global transform when changing zoom modes
        function resetGlobalTransform() {
            globalTransform.x = 0;
            globalTransform.y = 0;
            globalTransform.scale = 1;
        }

        // Load captures on startup and auto-load first
        loadCaptureList().then(() => {
            // Auto-load the first capture
            const qSelect = document.getElementById('qSelect');
            const jSelect = document.getElementById('jSelect');
            if (qSelect.value && jSelect.value) {
                loadCapture();
            }
        });
    </script>
</body>
</html>"""

        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))

    def serve_capture_list(self):
        """Serve JSON list of available captures."""
        capture_data = {}

        for q_val, j_dict in self.captures.items():
            capture_data[q_val] = {}
            for j_val, path in j_dict.items():
                # Check if capture has necessary files
                has_data = (path / 'images').exists() or (path / 'arrays').exists()
                capture_data[q_val][j_val] = {
                    'path': str(path),
                    'has_data': has_data
                }

        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(capture_data).encode('utf-8'))

    def serve_tile_list(self, q, j):
        """Serve JSON list of tiles for a specific Q/J capture."""
        if q not in self.captures or j not in self.captures[q]:
            self.send_error(404, f"Capture not found: Q{q} J{j}")
            return

        capture_dir = self.captures[q][j]
        tiles_data = {
            "capture_name": capture_dir.name,
            "q": q,
            "j": j,
            "tiles": []
        }

        # Find all L1 and L0 tiles
        for level in ['L1', 'L0']:
            max_coord = 2 if level == 'L1' else 4

            for y in range(max_coord):
                for x in range(max_coord):
                    tile_id = f"{level}_{x}_{y}"
                    tile_info = {
                        "id": tile_id,
                        "level": level,
                        "x": x,
                        "y": y,
                        "images": {},
                        "arrays": {}
                    }

                    # Check for images
                    img_dir = capture_dir / "images"
                    if img_dir.exists():
                        # Original
                        if (img_dir / f"{tile_id}_original.png").exists():
                            tile_info["images"]["original"] = f"/image/{tile_id}_original.png"

                        # Prediction
                        if (img_dir / f"{tile_id}_prediction.png").exists():
                            tile_info["images"]["prediction"] = f"/image/{tile_id}_prediction.png"

                        # Encoded residual
                        if (img_dir / f"{tile_id}_residual_encoded.png").exists():
                            tile_info["images"]["residual_encoded"] = f"/image/{tile_id}_residual_encoded.png"

                    # Check for arrays
                    arr_dir = capture_dir / "arrays"
                    if arr_dir.exists():
                        # Y channels
                        if (arr_dir / f"{tile_id}_Y_gt.npy").exists():
                            tile_info["arrays"]["Y_ground_truth"] = f"/array/{tile_id}_Y_gt.npy"

                        if (arr_dir / f"{tile_id}_Y_pred.npy").exists():
                            tile_info["arrays"]["Y_prediction"] = f"/array/{tile_id}_Y_pred.npy"

                        # Chroma channels - check both predicted and ground truth
                        for chroma in ['Cb', 'Cr']:
                            # Prediction chroma
                            if (arr_dir / f"{tile_id}_{chroma}_pred.npy").exists():
                                tile_info["arrays"][chroma] = f"/array/{tile_id}_{chroma}_pred.npy"
                            # Ground truth chroma (fallback)
                            elif (arr_dir / f"{tile_id}_{chroma}_gt.npy").exists():
                                tile_info["arrays"][chroma] = f"/array/{tile_id}_{chroma}_gt.npy"

                        # Residual
                        if (arr_dir / f"{tile_id}_residual_raw.npy").exists():
                            tile_info["arrays"]["residual_raw"] = f"/array/{tile_id}_residual_raw.npy"

                    if tile_info["images"] or tile_info["arrays"]:
                        tiles_data["tiles"].append(tile_info)

        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(tiles_data).encode('utf-8'))

    def serve_image(self, image_path, q, j):
        """Serve an image file from a specific Q/J capture."""
        if q not in self.captures or j not in self.captures[q]:
            self.send_error(404, f"Capture not found: Q{q} J{j}")
            return

        capture_dir = self.captures[q][j]
        full_path = capture_dir / "images" / image_path

        if not full_path.exists():
            self.send_error(404, f"Image not found: {image_path}")
            return

        # Determine content type
        ext = full_path.suffix.lower()
        content_type = 'image/png' if ext == '.png' else 'image/jpeg'

        with open(full_path, 'rb') as f:
            data = f.read()

        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def serve_array(self, array_path, q, j):
        """Convert numpy array to image and serve it from a specific Q/J capture."""
        if q not in self.captures or j not in self.captures[q]:
            self.send_error(404, f"Capture not found: Q{q} J{j}")
            return

        capture_dir = self.captures[q][j]
        full_path = capture_dir / "arrays" / array_path

        if not full_path.exists():
            self.send_error(404, f"Array not found: {array_path}")
            return

        try:
            # Load array
            arr = np.load(full_path)

            # Convert to uint8 for display
            if arr.dtype == np.float32 or arr.dtype == np.float64:
                # Handle different array types
                if 'Cb' in array_path or 'Cr' in array_path:
                    # Chroma channels - centered at 128
                    arr = np.clip(arr + 128, 0, 255)
                elif 'residual' in array_path:
                    # Residuals - visualize as centered difference
                    abs_max = np.abs(arr).max()
                    if abs_max > 0:
                        # Scale to use full range, centered at 128
                        arr = (arr / abs_max) * 127 + 128
                    else:
                        arr = np.full_like(arr, 128)
                    arr = np.clip(arr, 0, 255)
                else:
                    # Luma channel
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


def run_server(paper_dir, port=8080):
    """Run the multi-capture tile viewer server."""
    paper_path = pathlib.Path(paper_dir)

    if not paper_path.exists():
        print(f"Error: Paper directory not found: {paper_dir}")
        return

    # Create handler with paper directory
    def handler_factory(*args, **kwargs):
        return MultiCaptureHandler(*args, paper_dir=paper_dir, **kwargs)

    server = HTTPServer(('', port), handler_factory)

    print(f"ORIGAMI Multi-Capture Tile Viewer")
    print(f"Scanning directory: {paper_path}")
    print(f"Server running at: http://localhost:{port}")
    print(f"Press Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.shutdown()


def main():
    parser = argparse.ArgumentParser(description="ORIGAMI Multi-Capture Tile Viewer")
    parser.add_argument("--paper-dir", default="paper", help="Paper directory with debug captures (default: paper)")
    parser.add_argument("--port", type=int, default=8080, help="Server port (default: 8080)")

    args = parser.parse_args()

    run_server(args.paper_dir, args.port)


if __name__ == "__main__":
    main()