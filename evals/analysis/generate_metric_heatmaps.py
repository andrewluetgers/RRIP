#!/usr/bin/env python3
"""
Generate HTML heat map tables for compression metrics visualization.
Creates an interactive HTML page with heat maps for each metric.
"""

import argparse
import pathlib
import json
import numpy as np
from datetime import datetime
import glob


def load_analysis_data(capture_dirs):
    """Load analysis data from multiple captures."""
    data = {}

    for capture_dir in capture_dirs:
        capture_path = pathlib.Path(capture_dir)

        # Extract JPEG quality and quantization from folder name
        folder_name = capture_path.name
        jpeg_q = None
        quant = None

        # Extract quantization
        if "_q" in folder_name:
            parts = folder_name.split("_q")
            if len(parts) > 1:
                quant_str = parts[1].split("_")[0]
                try:
                    quant = int(quant_str)
                except ValueError:
                    pass

        # Extract JPEG quality
        if "_j" in folder_name:
            parts = folder_name.split("_j")
            if len(parts) > 1:
                jpeg_str = parts[1].split("_")[0]
                try:
                    jpeg_q = int(jpeg_str)
                except ValueError:
                    pass

        if jpeg_q is None or quant is None:
            continue

        # Try to load analysis manifest first
        manifest_path = capture_path / "analysis_manifest.json"

        # If no analysis manifest, try old manifest.json
        if not manifest_path.exists():
            old_manifest_path = capture_path / "manifest.json"
            if old_manifest_path.exists():
                # For old captures, we need to calculate metrics from raw data
                print(f"Processing old capture: {capture_dir}")
                # Try to run analysis
                import subprocess
                result = subprocess.run(["python", "cli/wsi_residual_analyze.py",
                              "--capture-dir", str(capture_dir)],
                              capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"  Warning: Analysis failed for {capture_dir}")

        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)

            # Extract metrics
            metrics = extract_metrics(manifest)
            metrics["jpeg_quality"] = jpeg_q
            metrics["quantization"] = quant
            metrics["capture_dir"] = str(capture_dir)

            # Store by Q and J
            if quant not in data:
                data[quant] = {}
            data[quant][jpeg_q] = metrics

    return data


def extract_metrics(manifest):
    """Extract average metrics from analysis manifest."""
    metrics = {}

    # Get compression ratio from statistics
    if "statistics" in manifest:
        stats = manifest["statistics"]
        metrics["compression_ratio"] = stats.get("compression_ratio", 0)
        metrics["savings_pct"] = stats.get("savings_pct", 0)

    # Get quality metrics
    if "compression_phase" in manifest:
        l0_psnrs = []
        l0_ssims = []
        l0_vifs = []
        l0_delta_es = []

        for key, info in manifest["compression_phase"].items():
            if key.startswith("L0_") and "prediction_metrics" in info:
                m = info["prediction_metrics"]
                if "psnr" in m:
                    l0_psnrs.append(m["psnr"])
                if "ssim" in m:
                    l0_ssims.append(m["ssim"])
                if "vif" in m:
                    l0_vifs.append(m["vif"])
                if "delta_e" in m and isinstance(m["delta_e"], dict):
                    l0_delta_es.append(m["delta_e"]["mean"])

        if l0_psnrs:
            metrics["psnr"] = np.mean(l0_psnrs)
        if l0_ssims:
            metrics["ssim"] = np.mean(l0_ssims)
        if l0_vifs:
            metrics["vif"] = np.mean(l0_vifs)
        if l0_delta_es:
            metrics["delta_e"] = np.mean(l0_delta_es)

    return metrics


def generate_html(data):
    """Generate HTML with heat map tables for each metric."""

    # Get all quantizations and JPEG qualities
    quants = sorted(data.keys())
    jpeg_qualities = set()
    for q_data in data.values():
        jpeg_qualities.update(q_data.keys())
    jpeg_qualities = sorted(jpeg_qualities)

    # Define metrics to visualize with Wikipedia links
    metrics_config = [
        {
            "key": "compression_ratio",
            "title": "Compression Ratio",
            "format": ".2f",
            "unit": ":1",
            "higher_better": True,
            "color": "blue",
            "wiki_link": "https://en.wikipedia.org/wiki/Data_compression_ratio",
            "description": "Ratio of uncompressed size to compressed size"
        },
        {
            "key": "psnr",
            "title": "PSNR (Peak Signal-to-Noise Ratio)",
            "format": ".2f",
            "unit": " dB",
            "higher_better": True,
            "color": "green",
            "wiki_link": "https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio",
            "description": "Ratio between maximum possible signal power and corrupting noise power"
        },
        {
            "key": "ssim",
            "title": "SSIM (Structural Similarity Index)",
            "format": ".4f",
            "unit": "",
            "higher_better": True,
            "color": "purple",
            "wiki_link": "https://en.wikipedia.org/wiki/Structural_similarity",
            "description": "Perceptual metric that quantifies image quality degradation"
        },
        {
            "key": "vif",
            "title": "VIF (Visual Information Fidelity)",
            "format": ".3f",
            "unit": "",
            "higher_better": True,
            "color": "teal",
            "wiki_link": "https://en.wikipedia.org/wiki/Video_quality",
            "description": "Information theoretic metric measuring visual information shared between images"
        },
        {
            "key": "delta_e",
            "title": "ΔE (Delta E - CIE 2000)",
            "format": ".2f",
            "unit": "",
            "higher_better": False,  # Lower is better
            "color": "orange",
            "wiki_link": "https://en.wikipedia.org/wiki/Color_difference#CIEDE2000",
            "description": "Perceptual color difference metric (ΔE < 2.3 = barely perceptible)"
        }
    ]

    html = """
<!DOCTYPE html>
<html>
<head>
    <title>ORIGAMI Compression Metrics Heat Maps</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }
        h1 {
            text-align: center;
            color: #333;
        }
        .metric-section {
            background: white;
            margin: 30px auto;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            max-width: 800px;
        }
        h2 {
            color: #555;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 10px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        th, td {
            padding: 15px;
            text-align: center;
            border: 1px solid #ddd;
            position: relative;
        }
        th {
            background: #f8f8f8;
            font-weight: 600;
        }
        td.metric-cell {
            font-size: 14px;
            font-weight: 500;
        }
        .value {
            position: relative;
            z-index: 2;
        }
        .metric-info {
            color: #666;
            font-size: 12px;
            margin-top: 10px;
        }
        .legend {
            display: flex;
            justify-content: center;
            margin-top: 15px;
            gap: 20px;
            font-size: 12px;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 5px;
        }
        .legend-box {
            width: 20px;
            height: 10px;
            border: 1px solid #ddd;
        }
        .timestamp {
            text-align: center;
            color: #999;
            font-size: 12px;
            margin-top: 30px;
        }
        .navigation {
            position: sticky;
            top: 0;
            background: white;
            padding: 15px;
            border-bottom: 2px solid #e0e0e0;
            margin: -20px -20px 20px -20px;
            z-index: 100;
        }
        .nav-buttons {
            display: flex;
            justify-content: center;
            gap: 10px;
        }
        .nav-btn {
            padding: 8px 16px;
            background: #007bff;
            color: white;
            text-decoration: none;
            border-radius: 4px;
            font-size: 14px;
        }
        .nav-btn:hover {
            background: #0056b3;
        }
    </style>
</head>
<body>
    <h1>ORIGAMI Compression Metrics Analysis</h1>

    <div class="navigation">
        <div class="nav-buttons">
"""

    # Add navigation buttons
    for config in metrics_config:
        html += f'            <a href="#{config["key"]}" class="nav-btn">{config["title"]}</a>\n'

    html += """        </div>
    </div>
"""

    # Generate table for each metric
    for config in metrics_config:
        metric_key = config["key"]

        # Get min/max for normalization across all data
        all_values = []
        for q_data in data.values():
            for j_data in q_data.values():
                if metric_key in j_data:
                    all_values.append(j_data[metric_key])

        if all_values:
            min_val = min(all_values)
            max_val = max(all_values)
        else:
            min_val = 0
            max_val = 1

        html += f"""
    <div class="metric-section" id="{metric_key}">
        <h2>{config["title"]} <a href="{config["wiki_link"]}" target="_blank" style="font-size: 14px; text-decoration: none;">&#9432;</a></h2>
        <p style="color: #666; font-size: 14px; margin-top: -10px;">{config["description"]}</p>
        <table>
            <tr>
                <th style="border-right: 2px solid #333;">JPEG &rarr;<br>Quantization &darr;</th>
"""

        # Add column headers for JPEG qualities
        for jq in jpeg_qualities:
            html += f'                <th>J{jq}</th>\n'
        html += '            </tr>\n'

        # Add rows for each quantization level
        for q in quants:
            html += f'            <tr>\n'
            html += f'                <th style="border-right: 2px solid #333;">Q{q}</th>\n'

            for jq in jpeg_qualities:
                if q in data and jq in data[q]:
                    value = data[q][jq].get(metric_key, None)

                    if value is not None:
                        # Calculate color intensity (0-1)
                        if max_val > min_val:
                            if config["higher_better"]:
                                intensity = (value - min_val) / (max_val - min_val)
                            else:
                                intensity = 1 - (value - min_val) / (max_val - min_val)
                        else:
                            intensity = 0.5

                        # Generate color based on metric type
                        if config["color"] == "blue":
                            color = f"rgba(33, 150, 243, {intensity * 0.4 + 0.1})"
                        elif config["color"] == "green":
                            color = f"rgba(76, 175, 80, {intensity * 0.4 + 0.1})"
                        elif config["color"] == "purple":
                            color = f"rgba(156, 39, 176, {intensity * 0.4 + 0.1})"
                        elif config["color"] == "teal":
                            color = f"rgba(0, 150, 136, {intensity * 0.4 + 0.1})"
                        elif config["color"] == "orange":
                            color = f"rgba(255, 152, 0, {(1-intensity) * 0.4 + 0.1})"  # Invert for lower-is-better
                        else:
                            color = f"rgba(100, 100, 100, {intensity * 0.4 + 0.1})"

                        format_str = config["format"]
                        formatted_value = f"{value:{format_str}}{config['unit']}"

                        html += f'                <td class="metric-cell" style="background-color: {color}">'
                        html += f'<span class="value">{formatted_value}</span></td>\n'
                    else:
                        html += '                <td class="metric-cell" style="background: #f0f0f0">-</td>\n'
                else:
                    html += '                <td class="metric-cell" style="background: #f0f0f0">-</td>\n'

            html += '            </tr>\n'

        html += '        </table>\n'

        # Add metric info
        better = "Higher" if config["higher_better"] else "Lower"
        html += f'        <div class="metric-info">{better} values are better. '
        html += f'<a href="{config["wiki_link"]}" target="_blank">Learn more on Wikipedia →</a></div>\n'

        # Add legend
        html += '        <div class="legend">\n'
        if config["higher_better"]:
            html += f'            <div class="legend-item"><div class="legend-box" style="background: {config["color"]}; opacity: 0.1"></div> Low (worse)</div>\n'
            html += f'            <div class="legend-item"><div class="legend-box" style="background: {config["color"]}; opacity: 0.5"></div> High (better)</div>\n'
        else:
            html += f'            <div class="legend-item"><div class="legend-box" style="background: {config["color"]}; opacity: 0.5"></div> Low (better)</div>\n'
            html += f'            <div class="legend-item"><div class="legend-box" style="background: {config["color"]}; opacity: 0.1"></div> High (worse)</div>\n'
        html += '        </div>\n'
        html += '    </div>\n'

    # Add timestamp
    html += f"""
    <div class="timestamp">
        Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    </div>
</body>
</html>
"""

    return html


def main():
    parser = argparse.ArgumentParser(description="Generate metric heat map visualization")
    parser.add_argument("--capture-pattern", default="paper/debug_L0-1024_q32_j*",
                       help="Pattern to match capture directories")
    parser.add_argument("--output", default="paper/metric_heatmaps.html",
                       help="Output HTML file")

    args = parser.parse_args()

    # Find all matching capture directories
    capture_dirs = glob.glob(args.capture_pattern)
    if not capture_dirs:
        print(f"No capture directories found matching: {args.capture_pattern}")
        return

    print(f"Found {len(capture_dirs)} capture directories")

    # Load data from captures
    data = load_analysis_data(capture_dirs)

    if not data:
        print("No valid data found in captures")
        return

    print(f"Loaded data for JPEG qualities: {sorted(data.keys())}")

    # Generate HTML
    html = generate_html(data)

    # Save HTML
    output_path = pathlib.Path(args.output)
    output_path.write_text(html)

    print(f"Heat map visualization saved to: {output_path}")
    print(f"Open in browser: file://{output_path.absolute()}")


if __name__ == "__main__":
    main()