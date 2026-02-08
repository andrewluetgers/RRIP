#!/usr/bin/env python3
"""
Simple HTTP server for viewing ORIGAMI debug tiles.
Serves tile images and provides an interactive viewer interface.
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


class TileViewerHandler(SimpleHTTPRequestHandler):
    """Custom handler for serving tile images and viewer interface."""

    def __init__(self, *args, capture_dir=None, **kwargs):
        self.capture_dir = pathlib.Path(capture_dir) if capture_dir else pathlib.Path.cwd()
        super().__init__(*args, **kwargs)

    def do_GET(self):
        """Handle GET requests."""
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == '/' or path == '/index.html':
            self.serve_viewer()
        elif path == '/tiles.json':
            self.serve_tile_list()
        elif path.startswith('/image/'):
            self.serve_image(path[7:])  # Remove /image/ prefix
        elif path.startswith('/array/'):
            self.serve_array(path[7:])  # Remove /array/ prefix
        else:
            super().do_GET()

    def serve_viewer(self):
        """Serve the main viewer HTML."""
        html = """<!DOCTYPE html>
<html>
<head>
    <title>ORIGAMI Tile Viewer</title>
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
            background: white;
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
            background: #f0f0f0;
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

        .image-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
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

        .image-viewport {
            width: 100%;
            height: 300px;
            border: 1px solid #ddd;
            border-radius: 4px;
            background: #f8f9fa;
            overflow: auto;
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .image-viewport img {
            image-rendering: pixelated;
            image-rendering: -moz-crisp-edges;
            image-rendering: crisp-edges;
            display: block;
        }

        /* Zoom styles */
        .zoom-fit .image-viewport {
            overflow: hidden;
        }

        .zoom-fit .image-viewport img {
            max-width: 100%;
            max-height: 100%;
            width: auto;
            height: auto;
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
    </style>
</head>
<body>
    <div class="header">
        <h1>ORIGAMI Tile Viewer</h1>
        <div class="metadata" id="metadata">
            Loading capture information...
        </div>
    </div>

    <div class="controls">
        <div class="tile-selector">
            <label for="tileSelect">Select Tile:</label>
            <select id="tileSelect">
                <option value="">Loading...</option>
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
                    <h3>Original</h3>
                    <div class="image-viewport">
                        <div id="original"></div>
                    </div>
                </div>
                <div class="image-container">
                    <h3>Prediction</h3>
                    <div class="image-viewport">
                        <div id="prediction"></div>
                    </div>
                </div>
            </div>

            <h2>YCbCr Channel Decomposition</h2>
            <div class="channel-grid">
                <div class="image-container">
                    <h3>Y (Luminance) - Original</h3>
                    <div class="image-viewport">
                        <div id="Y_ground_truth"></div>
                    </div>
                </div>
                <div class="image-container">
                    <h3>Y (Luminance) - Prediction</h3>
                    <div class="image-viewport">
                        <div id="Y_prediction"></div>
                    </div>
                </div>
                <div class="image-container">
                    <h3>Cb (Blue Chroma)</h3>
                    <div class="image-viewport">
                        <div id="Cb"></div>
                    </div>
                </div>
                <div class="image-container">
                    <h3>Cr (Red Chroma)</h3>
                    <div class="image-viewport">
                        <div id="Cr"></div>
                    </div>
                </div>
            </div>

            <h2>Residuals</h2>
            <div class="comparison-grid">
                <div class="image-container">
                    <h3>Y Residual (Original - Prediction)</h3>
                    <p style="font-size: 12px; color: #666;">Gray=0, Black=negative, White=positive</p>
                    <div class="image-viewport">
                        <div id="residual_raw"></div>
                    </div>
                </div>
                <div class="image-container">
                    <h3>Encoded Residual</h3>
                    <p style="font-size: 12px; color: #666;">As stored in JPEG</p>
                    <div class="image-viewport">
                        <div id="residual_encoded"></div>
                    </div>
                </div>
            </div>
        </div>

        <div class="navigation">
            <button id="prevBtn" onclick="previousTile()">← Previous</button>
            <button id="nextBtn" onclick="nextTile()">Next →</button>
        </div>
    </div>

    <script>
        let tiles = [];
        let currentIndex = 0;
        let currentZoom = 'fit';

        async function loadTiles() {
            try {
                const response = await fetch('/tiles.json');
                tiles = await response.json();

                // Update metadata
                document.getElementById('metadata').textContent =
                    `Capture: ${tiles.capture_name || 'Unknown'}`;

                // Populate selector
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
                document.getElementById('metadata').textContent = 'Error loading tiles';
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

            // Update images
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
                    container.innerHTML = `<img src="${path}" alt="${id}" onerror="this.parentElement.parentElement.innerHTML='<div class=\\"no-image\\">Failed to load</div>'">`;
                } else {
                    container.innerHTML = '<div class="no-image">Not available</div>';
                }
            });

            // Update navigation buttons
            document.getElementById('prevBtn').disabled = currentIndex === 0;
            document.getElementById('nextBtn').disabled = currentIndex === tiles.tiles.length - 1;
        }

        function setZoom(zoom) {
            currentZoom = zoom;
            const mainContent = document.getElementById('mainContent');

            // Remove all zoom classes
            mainContent.className = mainContent.className.replace(/zoom-\w+/g, '');

            // Add new zoom class
            mainContent.classList.add(`zoom-${zoom}`);

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
        document.getElementById('tileSelect').addEventListener('change', function() {
            showTile(parseInt(this.value));
        });

        // Zoom control handlers
        document.querySelectorAll('.zoom-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                setZoom(btn.dataset.zoom);
            });
        });

        // Load tiles on startup
        loadTiles();
    </script>
</body>
</html>"""

        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))

    def serve_tile_list(self):
        """Serve JSON list of available tiles."""
        tiles_data = {
            "capture_name": self.capture_dir.name,
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
                    img_dir = self.capture_dir / "images"
                    if img_dir.exists():
                        # Original
                        if (img_dir / f"{tile_id}_original.png").exists():
                            tile_info["images"]["original"] = f"/image/{tile_id}_original.png"

                        # Prediction
                        if (img_dir / f"{tile_id}_prediction.png").exists():
                            tile_info["images"]["prediction"] = f"/image/{tile_id}_prediction.png"

                        # Encoded residual (only for L0 tiles)
                        if level == 'L0' and (img_dir / f"{tile_id}_residual_encoded.png").exists():
                            tile_info["images"]["residual_encoded"] = f"/image/{tile_id}_residual_encoded.png"

                    # Check for arrays
                    arr_dir = self.capture_dir / "arrays"
                    if arr_dir.exists():
                        # Y channels
                        if (arr_dir / f"{tile_id}_Y_gt.npy").exists():
                            tile_info["arrays"]["Y_ground_truth"] = f"/array/{tile_id}_Y_gt.npy"

                        if (arr_dir / f"{tile_id}_Y_pred.npy").exists():
                            tile_info["arrays"]["Y_prediction"] = f"/array/{tile_id}_Y_pred.npy"

                        # Chroma
                        if (arr_dir / f"{tile_id}_Cb_pred.npy").exists():
                            tile_info["arrays"]["Cb"] = f"/array/{tile_id}_Cb_pred.npy"

                        if (arr_dir / f"{tile_id}_Cr_pred.npy").exists():
                            tile_info["arrays"]["Cr"] = f"/array/{tile_id}_Cr_pred.npy"

                        # Residual
                        if (arr_dir / f"{tile_id}_residual_raw.npy").exists():
                            tile_info["arrays"]["residual_raw"] = f"/array/{tile_id}_residual_raw.npy"

                    if tile_info["images"] or tile_info["arrays"]:
                        tiles_data["tiles"].append(tile_info)

        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(tiles_data).encode('utf-8'))

    def serve_image(self, image_path):
        """Serve an image file."""
        full_path = self.capture_dir / "images" / image_path

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

    def serve_array(self, array_path):
        """Convert numpy array to image and serve it."""
        full_path = self.capture_dir / "arrays" / array_path

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
                    # Residuals - visualize as absolute difference
                    # Scale to make differences more visible
                    # Negative values (darker in prediction) will be darker
                    # Positive values (brighter in prediction) will be brighter
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


def run_server(capture_dir, port=8080):
    """Run the tile viewer server."""
    capture_path = pathlib.Path(capture_dir)

    if not capture_path.exists():
        print(f"Error: Capture directory not found: {capture_dir}")
        return

    # Create handler with capture directory
    def handler_factory(*args, **kwargs):
        return TileViewerHandler(*args, capture_dir=capture_dir, **kwargs)

    server = HTTPServer(('', port), handler_factory)

    print(f"ORIGAMI Tile Viewer Server")
    print(f"Serving capture: {capture_path.name}")
    print(f"Server running at: http://localhost:{port}")
    print(f"Press Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.shutdown()


def main():
    parser = argparse.ArgumentParser(description="ORIGAMI Tile Viewer Server")
    parser.add_argument("--capture-dir", required=True, help="Debug capture directory")
    parser.add_argument("--port", type=int, default=8080, help="Server port (default: 8080)")

    args = parser.parse_args()

    run_server(args.capture_dir, args.port)


if __name__ == "__main__":
    main()