#!/usr/bin/env python3
"""
Generate an interactive HTML viewer for exploring ORIGAMI debug tile images.
Shows L0 tiles (20 total: 16 L0 + 4 L1) with YCbCr channel decomposition.
"""

import argparse
import pathlib
import json
import base64
import numpy as np
from PIL import Image
from io import BytesIO


def image_to_base64(image_path):
    """Convert image file to base64 data URI."""
    if not pathlib.Path(image_path).exists():
        return None

    with open(image_path, 'rb') as f:
        img_data = f.read()

    # Detect image type from extension
    ext = pathlib.Path(image_path).suffix.lower()
    mime_type = 'image/png' if ext == '.png' else 'image/jpeg'

    base64_str = base64.b64encode(img_data).decode('utf-8')
    return f"data:{mime_type};base64,{base64_str}"


def array_to_base64(array_path):
    """Convert numpy array to grayscale image and encode as base64."""
    if not pathlib.Path(array_path).exists():
        return None

    arr = np.load(array_path)

    # Convert to uint8 for display
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        # Assume YCbCr channels - handle appropriately
        if 'chroma' in str(array_path) or 'Cb' in str(array_path) or 'Cr' in str(array_path):
            # Chroma channels are centered at 128
            arr = np.clip(arr + 128, 0, 255)
        else:
            # Luma channel
            arr = np.clip(arr, 0, 255)
        arr = arr.astype(np.uint8)

    # Create PIL image
    if len(arr.shape) == 3:
        # RGB image
        img = Image.fromarray(arr)
    else:
        # Grayscale
        img = Image.fromarray(arr, mode='L')

    # Convert to base64
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{base64_str}"


def generate_viewer_html(capture_dir):
    """Generate HTML viewer for a debug capture."""
    capture_path = pathlib.Path(capture_dir)

    # Load metadata
    metadata_path = capture_path / "capture_metadata.json"
    if not metadata_path.exists():
        metadata_path = capture_path / "analysis_manifest.json"

    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
    else:
        metadata = {}

    # Find all L0 and L1 tiles
    tiles = []

    # Check for L1 tiles (4 tiles: 0_0, 0_1, 1_0, 1_1)
    for y in range(2):
        for x in range(2):
            tile_id = f"L1_{x}_{y}"
            tile_info = {
                "id": tile_id,
                "level": "L1",
                "x": x,
                "y": y,
                "images": {}
            }

            # Check for images
            img_dir = capture_path / "images"
            if img_dir.exists():
                # Original RGB
                orig_path = img_dir / f"{tile_id}_original.png"
                if orig_path.exists():
                    tile_info["images"]["original"] = image_to_base64(orig_path)

                # Prediction
                pred_path = img_dir / f"{tile_id}_prediction.png"
                if pred_path.exists():
                    tile_info["images"]["prediction"] = image_to_base64(pred_path)

            # Check for arrays (YCbCr channels)
            arr_dir = capture_path / "arrays"
            if arr_dir.exists():
                # Luma channels
                y_gt_path = arr_dir / f"{tile_id}_Y_gt.npy"
                if y_gt_path.exists():
                    tile_info["images"]["Y_ground_truth"] = array_to_base64(y_gt_path)

                y_pred_path = arr_dir / f"{tile_id}_Y_pred.npy"
                if y_pred_path.exists():
                    tile_info["images"]["Y_prediction"] = array_to_base64(y_pred_path)

                # Chroma channels
                cb_path = arr_dir / f"{tile_id}_Cb_pred.npy"
                if cb_path.exists():
                    tile_info["images"]["Cb"] = array_to_base64(cb_path)

                cr_path = arr_dir / f"{tile_id}_Cr_pred.npy"
                if cr_path.exists():
                    tile_info["images"]["Cr"] = array_to_base64(cr_path)

                # Residuals
                res_raw_path = arr_dir / f"{tile_id}_residual_raw.npy"
                if res_raw_path.exists():
                    tile_info["images"]["residual_raw"] = array_to_base64(res_raw_path)

            if tile_info["images"]:
                tiles.append(tile_info)

    # Check for L0 tiles (16 tiles: 0_0 through 3_3)
    for y in range(4):
        for x in range(4):
            tile_id = f"L0_{x}_{y}"
            tile_info = {
                "id": tile_id,
                "level": "L0",
                "x": x,
                "y": y,
                "images": {}
            }

            # Check for images
            img_dir = capture_path / "images"
            if img_dir.exists():
                # Original RGB
                orig_path = img_dir / f"{tile_id}_original.png"
                if orig_path.exists():
                    tile_info["images"]["original"] = image_to_base64(orig_path)

                # Prediction
                pred_path = img_dir / f"{tile_id}_prediction.png"
                if pred_path.exists():
                    tile_info["images"]["prediction"] = image_to_base64(pred_path)

                # Residual images
                res_enc_path = img_dir / f"{tile_id}_residual_encoded.png"
                if res_enc_path.exists():
                    tile_info["images"]["residual_encoded"] = image_to_base64(res_enc_path)

            # Check for arrays (YCbCr channels)
            arr_dir = capture_path / "arrays"
            if arr_dir.exists():
                # Luma channels
                y_gt_path = arr_dir / f"{tile_id}_Y_gt.npy"
                if y_gt_path.exists():
                    tile_info["images"]["Y_ground_truth"] = array_to_base64(y_gt_path)

                y_pred_path = arr_dir / f"{tile_id}_Y_pred.npy"
                if y_pred_path.exists():
                    tile_info["images"]["Y_prediction"] = array_to_base64(y_pred_path)

                # Chroma channels
                cb_path = arr_dir / f"{tile_id}_Cb_pred.npy"
                if cb_path.exists():
                    tile_info["images"]["Cb"] = array_to_base64(cb_path)

                cr_path = arr_dir / f"{tile_id}_Cr_pred.npy"
                if cr_path.exists():
                    tile_info["images"]["Cr"] = array_to_base64(cr_path)

                # Residuals
                res_raw_path = arr_dir / f"{tile_id}_residual_raw.npy"
                if res_raw_path.exists():
                    tile_info["images"]["residual_raw"] = array_to_base64(res_raw_path)

            if tile_info["images"]:
                tiles.append(tile_info)

    # Generate HTML
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
            margin: 0 0 10px 0;
            color: #555;
            font-size: 14px;
        }

        .image-container img {
            width: 100%;
            border: 1px solid #ddd;
            border-radius: 4px;
            background: white;
            image-rendering: pixelated;
            image-rendering: -moz-crisp-edges;
            image-rendering: crisp-edges;
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
            aspect-ratio: 1;
            background: #f0f0f0;
            border: 1px solid #ddd;
            border-radius: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #999;
            font-size: 12px;
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
    </style>
</head>
<body>
    <div class="header">
        <h1>ORIGAMI Tile Viewer</h1>
        <div class="metadata">""" + f"""
            Capture: {capture_path.name}
        </div>
    </div>

    <div class="controls">
        <div class="tile-selector">
            <label for="tileSelect">Select Tile:</label>
            <select id="tileSelect">"""

    # Add tile options
    for tile in tiles:
        html += f"""
                <option value="{tile['id']}">{tile['id']} ({tile['level']} at {tile['x']},{tile['y']})</option>"""

    html += """
            </select>
            <div class="tile-info" id="tileInfo"></div>
        </div>
    </div>

    <div class="viewer" id="viewer">
        <h2>Full Color Tiles</h2>
        <div class="comparison-grid">
            <div class="image-container">
                <h3>Original</h3>
                <div id="original"></div>
            </div>
            <div class="image-container">
                <h3>Prediction</h3>
                <div id="prediction"></div>
            </div>
        </div>

        <h2>YCbCr Channel Decomposition</h2>
        <div class="channel-grid">
            <div class="image-container">
                <h3>Y (Luminance) - Original</h3>
                <div id="Y_ground_truth"></div>
            </div>
            <div class="image-container">
                <h3>Y (Luminance) - Prediction</h3>
                <div id="Y_prediction"></div>
            </div>
            <div class="image-container">
                <h3>Cb (Blue Chroma)</h3>
                <div id="Cb"></div>
            </div>
            <div class="image-container">
                <h3>Cr (Red Chroma)</h3>
                <div id="Cr"></div>
            </div>
        </div>

        <h2>Residuals</h2>
        <div class="image-grid">
            <div class="image-container">
                <h3>Raw Residual</h3>
                <div id="residual_raw"></div>
            </div>
            <div class="image-container">
                <h3>Encoded Residual</h3>
                <div id="residual_encoded"></div>
            </div>
        </div>

        <div class="navigation">
            <button id="prevBtn" onclick="previousTile()">← Previous</button>
            <button id="nextBtn" onclick="nextTile()">Next →</button>
        </div>
    </div>

    <script>
        // Tile data
        const tiles = """ + json.dumps(tiles, indent=8) + """;

        let currentIndex = 0;

        function showTile(index) {
            if (index < 0 || index >= tiles.length) return;

            currentIndex = index;
            const tile = tiles[index];

            // Update selector
            document.getElementById('tileSelect').value = tile.id;

            // Update info
            document.getElementById('tileInfo').textContent =
                `Level: ${tile.level}, Position: (${tile.x}, ${tile.y})`;

            // Update images
            const imageTypes = [
                'original', 'prediction',
                'Y_ground_truth', 'Y_prediction', 'Cb', 'Cr',
                'residual_raw', 'residual_encoded'
            ];

            imageTypes.forEach(type => {
                const container = document.getElementById(type);
                if (!container) return;

                if (tile.images[type]) {
                    container.innerHTML = `<img src="${tile.images[type]}" alt="${type}">`;
                } else {
                    container.innerHTML = '<div class="no-image">Not available</div>';
                }
            });

            // Update navigation buttons
            document.getElementById('prevBtn').disabled = currentIndex === 0;
            document.getElementById('nextBtn').disabled = currentIndex === tiles.length - 1;
        }

        function previousTile() {
            if (currentIndex > 0) {
                showTile(currentIndex - 1);
            }
        }

        function nextTile() {
            if (currentIndex < tiles.length - 1) {
                showTile(currentIndex + 1);
            }
        }

        // Initialize
        document.getElementById('tileSelect').addEventListener('change', function() {
            const index = tiles.findIndex(t => t.id === this.value);
            if (index >= 0) {
                showTile(index);
            }
        });

        // Show first tile
        if (tiles.length > 0) {
            showTile(0);
        }
    </script>
</body>
</html>"""

    return html


def main():
    parser = argparse.ArgumentParser(description="Generate tile viewer HTML")
    parser.add_argument("--capture-dir", required=True, help="Debug capture directory")
    parser.add_argument("--output", default="tile_viewer.html", help="Output HTML file")

    args = parser.parse_args()

    capture_dir = pathlib.Path(args.capture_dir)
    if not capture_dir.exists():
        print(f"Error: Capture directory not found: {capture_dir}")
        return

    print(f"Generating viewer for: {capture_dir}")

    html = generate_viewer_html(capture_dir)

    output_path = pathlib.Path(args.output)
    output_path.write_text(html)

    print(f"Viewer saved to: {output_path}")
    print(f"Open in browser: file://{output_path.absolute()}")


if __name__ == "__main__":
    main()