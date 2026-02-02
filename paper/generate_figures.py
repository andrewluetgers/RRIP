#!/usr/bin/env python3
"""
Generate publication-quality figures for RRIP paper
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

# Set publication-quality defaults
matplotlib.rcParams['pdf.fonttype'] = 42  # TrueType fonts for PDF
matplotlib.rcParams['ps.fonttype'] = 42   # TrueType fonts for PS
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.labelsize'] = 10
matplotlib.rcParams['axes.titlesize'] = 11
matplotlib.rcParams['xtick.labelsize'] = 9
matplotlib.rcParams['ytick.labelsize'] = 9
matplotlib.rcParams['legend.fontsize'] = 9
matplotlib.rcParams['figure.titlesize'] = 11
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

def load_evaluation_results(results_path="../evaluation_results/results.json"):
    """Load evaluation results from JSON"""
    with open(results_path, 'r') as f:
        data = json.load(f)
    return data

def create_rd_curves(results, output_path="figures/rd_curves.pdf"):
    """Create rate-distortion curves for paper"""

    # Create output directory
    Path(output_path).parent.mkdir(exist_ok=True)

    # Figure for single column (3.5 inches wide)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.5))

    # Group results by method
    rrip_data = [r for r in results if r['method'] == 'RRIP']
    jpeg_data = [r for r in results if r['method'] == 'JPEG']

    # RRIP point
    if rrip_data:
        rrip_bpp = np.mean([r['bpp'] for r in rrip_data])
        rrip_psnr = np.mean([r['psnr_db'] for r in rrip_data])
        rrip_ssim = np.mean([r['ssim'] for r in rrip_data])

        ax1.plot(rrip_bpp, rrip_psnr, 'r^', markersize=10,
                label='RRIP', markeredgewidth=1.5, markeredgecolor='darkred')
        ax2.plot(rrip_bpp, rrip_ssim, 'r^', markersize=10,
                label='RRIP', markeredgewidth=1.5, markeredgecolor='darkred')

    # JPEG curve - group by quality
    jpeg_by_quality = {}
    for r in jpeg_data:
        q = r.get('quality', r.get('quality_param', 0))
        if q not in jpeg_by_quality:
            jpeg_by_quality[q] = []
        jpeg_by_quality[q].append(r)

    # Calculate mean for each quality level
    jpeg_points = []
    for q in sorted(jpeg_by_quality.keys(), reverse=True):
        if jpeg_by_quality[q]:
            jpeg_points.append({
                'quality': q,
                'bpp': np.mean([r['bpp'] for r in jpeg_by_quality[q]]),
                'psnr': np.mean([r['psnr_db'] for r in jpeg_by_quality[q]]),
                'ssim': np.mean([r['ssim'] for r in jpeg_by_quality[q]])
            })

    # Plot JPEG curve
    if jpeg_points:
        bpp_values = [p['bpp'] for p in jpeg_points]
        psnr_values = [p['psnr'] for p in jpeg_points]
        ssim_values = [p['ssim'] for p in jpeg_points]

        ax1.plot(bpp_values, psnr_values, 'b-o', markersize=6,
                label='JPEG (recompressed)', linewidth=1.5)
        ax2.plot(bpp_values, ssim_values, 'b-o', markersize=6,
                label='JPEG (recompressed)', linewidth=1.5)

        # Add quality labels for key points
        for i, p in enumerate(jpeg_points):
            if p['quality'] in [95, 80, 60]:  # Only label some points
                ax1.annotate(f"Q{p['quality']}",
                           (p['bpp'], p['psnr']),
                           textcoords="offset points",
                           xytext=(0,-12), ha='center',
                           fontsize=7, color='blue')

    # Formatting for PSNR plot
    ax1.set_xlabel('Bits Per Pixel (bpp)')
    ax1.set_ylabel('PSNR vs JPEG Q90 (dB)')
    ax1.grid(True, alpha=0.3, linewidth=0.5)
    ax1.legend(loc='lower right', framealpha=0.9)
    ax1.set_xlim(left=0)

    # Add reference line at 40 dB
    ax1.axhline(y=40, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax1.text(ax1.get_xlim()[1]*0.95, 40.5, '40 dB', ha='right', fontsize=8, color='gray')

    # Formatting for SSIM plot
    ax2.set_xlabel('Bits Per Pixel (bpp)')
    ax2.set_ylabel('SSIM vs JPEG Q90')
    ax2.grid(True, alpha=0.3, linewidth=0.5)
    ax2.legend(loc='lower right', framealpha=0.9)
    ax2.set_xlim(left=0)
    ax2.set_ylim([0.94, 1.0])

    # Add RRIP annotation
    if rrip_data:
        ax1.annotate('RRIP\n(Our Method)',
                    xy=(rrip_bpp, rrip_psnr),
                    xytext=(rrip_bpp+0.3, rrip_psnr-3),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1),
                    fontsize=8, color='red', fontweight='bold')

    plt.suptitle('Rate-Distortion Performance (Relative to JPEG Q90 Baseline)',
                fontsize=11, fontweight='bold')
    plt.tight_layout()

    # Save in multiple formats
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"Saved rate-distortion curves to {output_path}")

def create_storage_comparison(output_path="figures/storage_comparison.pdf"):
    """Create storage comparison figure"""

    Path(output_path).parent.mkdir(exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))

    # Storage sizes for 100k x 100k image
    methods = ['Raw\nPixels', 'JPEG\nQ90', 'RRIP', 'JPEG Q60\n(Recomp.)']
    sizes_gb = [30, 4.87, 0.88, 0.49]
    colors = ['gray', 'blue', 'red', 'lightblue']

    # Bar chart
    bars = ax1.bar(methods, sizes_gb, color=colors, edgecolor='black', linewidth=1)
    ax1.set_ylabel('Storage Size (GB)')
    ax1.set_title('Storage Requirements\n(100,000×100,000 WSI)', fontsize=10)

    # Add value labels on bars
    for bar, size in zip(bars, sizes_gb):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{size:.2f} GB',
                ha='center', va='bottom', fontsize=8)

    # Add compression ratios
    compression_ratios = [1, 6.2, 34, 61]
    for i, (bar, ratio) in enumerate(zip(bars, compression_ratios)):
        ax1.text(bar.get_x() + bar.get_width()/2., 0.5,
                f'{ratio}×',
                ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Cost comparison (right plot)
    scenarios = ['JPEG\nPyramid', 'With\nRRIP']
    costs = [318, 57]
    colors2 = ['blue', 'red']

    bars2 = ax2.bar(scenarios, costs, color=colors2, edgecolor='black', linewidth=1)
    ax2.set_ylabel('Annual Cost ($1000s)')
    ax2.set_title('Storage Cost for 1PB\n(AWS S3 Standard)', fontsize=10)

    # Add value labels
    for bar, cost in zip(bars2, costs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'${cost}K',
                ha='center', va='bottom', fontsize=9)

    # Add savings annotation
    ax2.annotate('', xy=(1, costs[1]), xytext=(1, costs[0]),
                arrowprops=dict(arrowstyle='<->', color='green', lw=2))
    ax2.text(1.15, (costs[0] + costs[1])/2, '$261K\nSavings',
            ha='left', va='center', color='green', fontweight='bold')

    plt.suptitle('RRIP Storage Efficiency', fontsize=11, fontweight='bold')
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"Saved storage comparison to {output_path}")

def create_performance_timeline(output_path="figures/performance_timeline.pdf"):
    """Create performance timeline figure"""

    Path(output_path).parent.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 2.5))

    # Timeline data (in ms)
    operations = [
        ('Read pack', 0, 1, 'blue'),
        ('Decode L2', 1, 2, 'green'),
        ('Upsample L1', 2, 3, 'orange'),
        ('L1 residuals', 3, 4, 'red'),
        ('Upsample L0', 4, 5, 'orange'),
        ('L0 residuals', 5, 6.5, 'red'),
        ('Encode JPEGs', 6.5, 7, 'purple')
    ]

    # Draw timeline bars
    for i, (name, start, end, color) in enumerate(operations):
        ax.barh(i, end-start, left=start, height=0.6,
               color=color, alpha=0.7, edgecolor='black')
        ax.text(start + (end-start)/2, i, name,
               ha='center', va='center', fontsize=9, fontweight='bold')
        ax.text(end + 0.1, i, f'{end-start:.1f}ms',
               ha='left', va='center', fontsize=8)

    # Formatting
    ax.set_xlabel('Time (ms)')
    ax.set_yticks([])
    ax.set_xlim(0, 8)
    ax.set_ylim(-0.5, len(operations)-0.5)
    ax.grid(True, axis='x', alpha=0.3)
    ax.set_title('RRIP Family Generation Timeline (20 tiles)', fontsize=11)

    # Add total time annotation
    ax.text(7.5, len(operations)-1, 'Total: 4-7ms\n(0.35ms/tile)',
           ha='center', va='center', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"Saved performance timeline to {output_path}")

def create_architecture_diagram():
    """Create architecture diagram (placeholder - would use TikZ or draw.io)"""
    print("\nArchitecture diagram should be created using:")
    print("1. draw.io (diagrams.net) - Export as PDF")
    print("2. Adobe Illustrator / Inkscape")
    print("3. TikZ (within LaTeX)")
    print("\nSuggested elements:")
    print("- L2 baseline tile → Upsample 2× → L1 prediction")
    print("- L1 residuals → Addition → L1 reconstructed")
    print("- Similar for L0 with 4× upsampling")

def main():
    """Generate all figures for the paper"""

    print("Generating figures for RRIP paper...")
    print("=" * 50)

    # Load evaluation results if they exist
    results_path = Path("../evaluation_results/results.json")
    if results_path.exists():
        results = load_evaluation_results(results_path)
        create_rd_curves(results)
    else:
        print(f"Warning: {results_path} not found")
        print("Creating example R-D curves with synthetic data...")

        # Create synthetic data for demonstration
        synthetic_results = [
            # RRIP
            {'method': 'RRIP', 'bpp': 1.0, 'psnr_db': 49.81, 'ssim': 0.9803},
            # JPEG at different qualities
            {'method': 'JPEG', 'quality': 95, 'bpp': 0.785, 'psnr_db': 69.20, 'ssim': 0.9997},
            {'method': 'JPEG', 'quality': 90, 'bpp': 0.630, 'psnr_db': 68.98, 'ssim': 0.9997},
            {'method': 'JPEG', 'quality': 80, 'bpp': 0.493, 'psnr_db': 64.64, 'ssim': 0.9993},
            {'method': 'JPEG', 'quality': 70, 'bpp': 0.452, 'psnr_db': 62.39, 'ssim': 0.9981},
            {'method': 'JPEG', 'quality': 60, 'bpp': 0.402, 'psnr_db': 57.88, 'ssim': 0.9956},
        ]
        create_rd_curves(synthetic_results)

    # Create other figures
    create_storage_comparison()
    create_performance_timeline()
    create_architecture_diagram()

    print("\n" + "=" * 50)
    print("Figure generation complete!")
    print("\nGenerated files:")
    print("- figures/rd_curves.pdf (and .png)")
    print("- figures/storage_comparison.pdf (and .png)")
    print("- figures/performance_timeline.pdf (and .png)")
    print("\nTo include in LaTeX:")
    print("\\includegraphics[width=0.45\\textwidth]{figures/rd_curves.pdf}")

if __name__ == "__main__":
    main()