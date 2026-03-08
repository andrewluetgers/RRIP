#!/usr/bin/env python3
"""
Generate training progress plots from training_log.jsonl.

Produces a multi-panel figure showing how all key metrics evolve during training,
with baseline references (bilinear/bicubic) as horizontal dashed lines.

Can be run while training is in progress — reads whatever is logged so far.

Usage:
  python plot_training.py --log checkpoints/training_log.jsonl
  python plot_training.py --log checkpoints/training_log.jsonl --output training_plots.png
  python plot_training.py --log checkpoints/training_log.jsonl --live  # re-plot every 30s
"""

import argparse
import json
import os
import sys
import time

import numpy as np


def load_log(log_path: str):
    """Load training log, return header and entries."""
    header = None
    entries = []

    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("type") == "header":
                header = entry
            else:
                entries.append(entry)

    return header, entries


def plot_training(log_path: str, output_path: str, show: bool = False):
    """Generate training plots from log file."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    header, entries = load_log(log_path)
    if not entries:
        print("No training entries found in log.")
        return

    # Separate train-only and val entries
    all_epochs = [e["epoch"] for e in entries]
    all_loss = [e["loss"] for e in entries]
    all_train_psnr = [e["train_psnr"] for e in entries]

    val_entries = [e for e in entries if e.get("type") == "val"]
    val_epochs = [e["epoch"] for e in val_entries]
    val_psnr = [e["val_psnr"] for e in val_entries]

    # Residual stats from val entries
    has_residual = val_entries and "residual" in val_entries[0]

    # Visual metrics from val entries
    val_ssim = [e["val_ssim"] for e in val_entries if e.get("val_ssim") is not None]
    val_delta_e = [e["val_delta_e"] for e in val_entries if e.get("val_delta_e") is not None]
    val_mse = [e["val_mse"] for e in val_entries if e.get("val_mse") is not None]
    val_ssimulacra2 = [e["val_ssimulacra2"] for e in val_entries if e.get("val_ssimulacra2") is not None]
    has_visual_metrics = bool(val_ssim or val_delta_e or val_mse or val_ssimulacra2)
    # Epochs for visual metrics (may be subset if metrics were added mid-training)
    vm_epochs = [e["epoch"] for e in val_entries if e.get("val_ssim") is not None or e.get("val_mse") is not None]
    vm_epochs_s2 = [e["epoch"] for e in val_entries if e.get("val_ssimulacra2") is not None]

    # Baselines from header
    baselines = header.get("baselines", {}) if header else {}
    config = header.get("config", {}) if header else {}

    # Figure layout: +1 row for SSIM/Delta E, +1 row if SSIMULACRA2 present
    n_extra = 0
    if has_visual_metrics:
        n_extra += 1
    if val_ssimulacra2:
        n_extra += 1
    n_rows = (4 if has_residual else 2) + n_extra
    fig = plt.figure(figsize=(14, 4 * n_rows))
    gs = GridSpec(n_rows, 2, figure=fig, hspace=0.35, wspace=0.3)

    fig.suptitle(
        f"WSI SR Training — {config.get('mode', '?')} mode, "
        f"{config.get('channels', '?')}ch × {config.get('blocks', '?')}blk, "
        f"batch={config.get('batch', '?')}, crop={config.get('crop', '?')}, "
        f"lr={config.get('lr', '?')}",
        fontsize=13, y=0.995,
    )

    # --- Panel 1: Loss ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(all_epochs, all_loss, "b-", alpha=0.7, linewidth=0.8, label="MAE loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # --- Panel 2: PSNR ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(all_epochs, all_train_psnr, "b-", alpha=0.5, linewidth=0.8, label="Train PSNR")
    if val_epochs:
        ax2.plot(val_epochs, val_psnr, "r-o", markersize=3, linewidth=1.2, label="Val PSNR")
        # Mark best
        best_idx = np.argmax(val_psnr)
        ax2.axhline(y=val_psnr[best_idx], color="r", linestyle=":", alpha=0.4)
        ax2.annotate(f"Best: {val_psnr[best_idx]:.2f} dB (ep {val_epochs[best_idx]})",
                     xy=(val_epochs[best_idx], val_psnr[best_idx]),
                     fontsize=8, color="red")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("PSNR (dB)")
    ax2.set_title("PSNR (higher = better)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # --- Visual Metrics Row (SSIM, Delta E, MSE) ---
    row_offset = 1  # next available row after Loss/PSNR
    if has_visual_metrics:
        # --- Panel: SSIM ---
        ax_ssim = fig.add_subplot(gs[row_offset, 0])
        if val_ssim:
            ax_ssim.plot(vm_epochs[:len(val_ssim)], val_ssim, "g-o", markersize=3, linewidth=1.2, label="Val SSIM")
            best_ssim_idx = np.argmax(val_ssim)
            ax_ssim.axhline(y=val_ssim[best_ssim_idx], color="g", linestyle=":", alpha=0.4)
            ax_ssim.annotate(f"Best: {val_ssim[best_ssim_idx]:.4f} (ep {vm_epochs[best_ssim_idx]})",
                             xy=(vm_epochs[best_ssim_idx], val_ssim[best_ssim_idx]),
                             fontsize=8, color="green")
        for method, color, ls in [("bilinear", "orange", "--"), ("bicubic", "purple", "--")]:
            if method in baselines and baselines[method].get("ssim") is not None:
                ax_ssim.axhline(y=baselines[method]["ssim"], color=color, linestyle=ls,
                               alpha=0.6, label=f"{method}: {baselines[method]['ssim']:.4f}")
        ax_ssim.set_xlabel("Epoch")
        ax_ssim.set_ylabel("SSIM")
        ax_ssim.set_title("SSIM (higher = better)")
        ax_ssim.grid(True, alpha=0.3)
        ax_ssim.legend(fontsize=8)

        # --- Panel: Delta E + MSE ---
        ax_de = fig.add_subplot(gs[row_offset, 1])
        if val_delta_e:
            ax_de.plot(vm_epochs[:len(val_delta_e)], val_delta_e, "m-o", markersize=3, linewidth=1.2, label="Val Delta E")
            best_de_idx = np.argmin(val_delta_e)
            ax_de.axhline(y=val_delta_e[best_de_idx], color="m", linestyle=":", alpha=0.4)
            ax_de.annotate(f"Best: {val_delta_e[best_de_idx]:.2f} (ep {vm_epochs[best_de_idx]})",
                           xy=(vm_epochs[best_de_idx], val_delta_e[best_de_idx]),
                           fontsize=8, color="purple")
        for method, color, ls in [("bilinear", "orange", "--"), ("bicubic", "purple", "--")]:
            if method in baselines and baselines[method].get("delta_e") is not None:
                ax_de.axhline(y=baselines[method]["delta_e"], color=color, linestyle=ls,
                             alpha=0.6, label=f"{method}: {baselines[method]['delta_e']:.2f}")
        ax_de.set_xlabel("Epoch")
        ax_de.set_ylabel("Delta E (CIE2000)")
        ax_de.set_title("Delta E CIE2000 (lower = better)")
        ax_de.grid(True, alpha=0.3)
        ax_de.legend(fontsize=8)

        row_offset += 1

    # --- SSIMULACRA2 Row ---
    if val_ssimulacra2:
        # --- Panel: SSIMULACRA2 ---
        ax_s2 = fig.add_subplot(gs[row_offset, 0])
        ax_s2.plot(vm_epochs_s2[:len(val_ssimulacra2)], val_ssimulacra2, "c-o",
                   markersize=3, linewidth=1.2, label="Val SSIMULACRA2")
        best_s2_idx = np.argmax(val_ssimulacra2)
        ax_s2.axhline(y=val_ssimulacra2[best_s2_idx], color="c", linestyle=":", alpha=0.4)
        ax_s2.annotate(f"Best: {val_ssimulacra2[best_s2_idx]:.1f} (ep {vm_epochs_s2[best_s2_idx]})",
                       xy=(vm_epochs_s2[best_s2_idx], val_ssimulacra2[best_s2_idx]),
                       fontsize=8, color="teal")
        for method, color, ls in [("bilinear", "orange", "--"), ("bicubic", "purple", "--")]:
            if method in baselines and baselines[method].get("ssimulacra2") is not None:
                ax_s2.axhline(y=baselines[method]["ssimulacra2"], color=color, linestyle=ls,
                             alpha=0.6, label=f"{method}: {baselines[method]['ssimulacra2']:.1f}")
        ax_s2.set_xlabel("Epoch")
        ax_s2.set_ylabel("SSIMULACRA2")
        ax_s2.set_title("SSIMULACRA2 (higher = better, >90 excellent)")
        ax_s2.grid(True, alpha=0.3)
        ax_s2.legend(fontsize=8)

        # --- Panel: MSE ---
        ax_mse = fig.add_subplot(gs[row_offset, 1])
        if val_mse:
            ax_mse.plot(vm_epochs[:len(val_mse)], val_mse, "r-o", markersize=3, linewidth=1.2, label="Val MSE")
            best_mse_idx = np.argmin(val_mse)
            ax_mse.annotate(f"Best: {val_mse[best_mse_idx]:.1f} (ep {vm_epochs[best_mse_idx]})",
                            xy=(vm_epochs[best_mse_idx], val_mse[best_mse_idx]),
                            fontsize=8, color="red")
        for method, color, ls in [("bilinear", "orange", "--"), ("bicubic", "purple", "--")]:
            if method in baselines and baselines[method].get("mse") is not None:
                ax_mse.axhline(y=baselines[method]["mse"], color=color, linestyle=ls,
                              alpha=0.6, label=f"{method}: {baselines[method]['mse']:.1f}")
        ax_mse.set_xlabel("Epoch")
        ax_mse.set_ylabel("MSE")
        ax_mse.set_title("Mean Squared Error (lower = better)")
        ax_mse.grid(True, alpha=0.3)
        ax_mse.legend(fontsize=8)

        row_offset += 1

    if has_residual:
        res_sizes = [e["residual"]["size_kb"] for e in val_entries]
        res_max = [e["residual"]["max_dev"] for e in val_entries]
        res_p99 = [e["residual"]["p99_dev"] for e in val_entries]
        res_p95 = [e["residual"]["p95_dev"] for e in val_entries]
        res_mean = [e["residual"]["mean_dev"] for e in val_entries]
        res_pct10 = [e["residual"]["pct_over_10"] for e in val_entries]
        res_pct20 = [e["residual"]["pct_over_20"] for e in val_entries]
        res_pct30 = [e["residual"]["pct_over_30"] for e in val_entries]

        # --- Panel: Residual Size ---
        ax3 = fig.add_subplot(gs[row_offset, 0])
        ax3.plot(val_epochs, res_sizes, "g-o", markersize=3, linewidth=1.2, label="SR model")
        # Baseline references
        for method, color, ls in [("bilinear", "orange", "--"), ("bicubic", "purple", "--")]:
            if method in baselines:
                bl_size = baselines[method]["size_kb"]
                ax3.axhline(y=bl_size, color=color, linestyle=ls, alpha=0.6,
                           label=f"{method}: {bl_size:.1f} KB")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Size (KB)")
        ax3.set_title("Residual Size @ q80 (lower = better prediction)")
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=8)

        # --- Panel: Deviation percentiles ---
        ax4 = fig.add_subplot(gs[row_offset, 1])
        ax4.plot(val_epochs, res_max, "r-o", markersize=3, linewidth=1.2, label="max", alpha=0.7)
        ax4.plot(val_epochs, res_p99, "orange", marker="s", markersize=2, linewidth=1, label="p99")
        ax4.plot(val_epochs, res_p95, "g-^", markersize=2, linewidth=1, label="p95")
        ax4.plot(val_epochs, res_mean, "b-", linewidth=1, label="mean")
        # Baseline references
        for method, ls in [("bilinear", ":"), ("bicubic", "--")]:
            if method in baselines:
                ax4.axhline(y=baselines[method]["p99_dev"], color="gray", linestyle=ls,
                           alpha=0.4, label=f"{method} p99: {baselines[method]['p99_dev']:.1f}")
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Pixel deviation (0-255)")
        ax4.set_title("Luma Deviation Distribution (lower = better)")
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=7, ncol=2)

        # --- Panel: Percentage of outlier pixels ---
        ax5 = fig.add_subplot(gs[row_offset + 1, 0])
        ax5.plot(val_epochs, res_pct10, "y-o", markersize=3, linewidth=1, label=">10 px")
        ax5.plot(val_epochs, res_pct20, "orange", marker="s", markersize=3, linewidth=1.2, label=">20 px")
        ax5.plot(val_epochs, res_pct30, "r-^", markersize=3, linewidth=1.2, label=">30 px")
        # Baseline references
        for method, ls in [("bilinear", ":"), ("bicubic", "--")]:
            if method in baselines:
                ax5.axhline(y=baselines[method]["pct_over_20"], color="gray", linestyle=ls,
                           alpha=0.4, label=f"{method} >20: {baselines[method]['pct_over_20']:.2f}%")
        ax5.set_xlabel("Epoch")
        ax5.set_ylabel("% of pixels")
        ax5.set_title("Outlier Pixel Percentage (lower = fewer artifacts)")
        ax5.grid(True, alpha=0.3)
        ax5.legend(fontsize=7, ncol=2)

        # --- Panel: Learning rate ---
        ax6 = fig.add_subplot(gs[row_offset + 1, 1])
        all_lr = [e.get("lr", 0) for e in entries]
        ax6.plot(all_epochs, all_lr, "k-", linewidth=1)
        ax6.set_xlabel("Epoch")
        ax6.set_ylabel("Learning Rate")
        ax6.set_title("Learning Rate Schedule")
        ax6.set_yscale("log")
        ax6.grid(True, alpha=0.3)

        # --- Panel: Combined quality view (PSNR vs residual size) ---
        ax7 = fig.add_subplot(gs[row_offset + 2, 0])
        sc = ax7.scatter(res_sizes, val_psnr, c=val_epochs, cmap="viridis",
                        s=30, edgecolors="k", linewidth=0.5)
        plt.colorbar(sc, ax=ax7, label="Epoch")
        # Baseline points
        for method, marker, color in [("bilinear", "x", "orange"), ("bicubic", "+", "purple")]:
            if method in baselines and baselines[method].get("psnr") is not None:
                ax7.scatter([baselines[method]["size_kb"]], [baselines[method]["psnr"]],
                           marker=marker, color=color, s=100, zorder=5,
                           label=f"{method}: {baselines[method]['psnr']:.1f}dB")
        ax7.set_xlabel("Residual Size (KB)")
        ax7.set_ylabel("Val PSNR (dB)")
        ax7.set_title("Quality vs Compression Efficiency (up-left = best)")
        ax7.grid(True, alpha=0.3)
        ax7.legend(fontsize=8)

        # --- Panel: Improvement over baselines ---
        ax8 = fig.add_subplot(gs[row_offset + 2, 1])
        if "bicubic" in baselines:
            bl_size = baselines["bicubic"]["size_kb"]
            improvement = [(bl_size - s) / bl_size * 100 for s in res_sizes]
            ax8.plot(val_epochs, improvement, "g-o", markersize=3, linewidth=1.2,
                    label="Residual size reduction vs bicubic")
            ax8.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            ax8.fill_between(val_epochs, 0, improvement, alpha=0.1, color="green")
        ax8.set_xlabel("Epoch")
        ax8.set_ylabel("Improvement (%)")
        ax8.set_title("Storage Savings vs Bicubic Baseline")
        ax8.grid(True, alpha=0.3)
        ax8.legend(fontsize=8)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plots to: {output_path}")

    if show:
        plt.show()
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Plot training progress")
    ap.add_argument("--log", required=True, help="Path to training_log.jsonl")
    ap.add_argument("--output", "-o", default=None,
                    help="Output image path (default: <log_dir>/training_plots.png)")
    ap.add_argument("--show", action="store_true", help="Show interactive plot")
    ap.add_argument("--live", action="store_true",
                    help="Re-plot every 30 seconds (for monitoring during training)")
    ap.add_argument("--interval", type=int, default=30,
                    help="Live update interval in seconds (default: 30)")
    args = ap.parse_args()

    if args.output is None:
        log_dir = os.path.dirname(os.path.abspath(args.log))
        args.output = os.path.join(log_dir, "training_plots.png")

    if args.live:
        print(f"Live mode: updating {args.output} every {args.interval}s")
        print("Press Ctrl+C to stop.")
        while True:
            try:
                plot_training(args.log, args.output, show=False)
                print(f"  Updated at {time.strftime('%H:%M:%S')}")
                time.sleep(args.interval)
            except KeyboardInterrupt:
                print("\nStopped.")
                break
            except Exception as e:
                print(f"  Error: {e}")
                time.sleep(args.interval)
    else:
        plot_training(args.log, args.output, show=args.show)


if __name__ == "__main__":
    main()
