#!/usr/bin/env python3
"""Analyze V4 split-seed sweep results.

Reads manifest.json from all runs, computes rate-quality metrics,
identifies Pareto-optimal configs, and compares against JPEG/JXL baselines.

Usage:
    python evals/scripts/analyze_v4_sweep.py [--runs-dir evals/runs]
"""
import json
import os
import sys
from pathlib import Path

RUNS_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("evals/runs")


def load_run(run_dir: Path) -> dict | None:
    manifest = run_dir / "manifest.json"
    if not manifest.exists():
        return None
    try:
        data = json.loads(manifest.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    name = run_dir.name
    result = {"name": name, "dir": str(run_dir)}

    # JPEG/JXL baselines
    if data.get("type") == "jpeg_baseline":
        cfg = data.get("configuration", {})
        tiles = data.get("tiles", {})
        l0_tiles = {k: v for k, v in tiles.items() if k.startswith("L0_")}
        if not l0_tiles:
            return None
        total_bytes = sum(t["size_bytes"] for t in l0_tiles.values())
        avg_psnr = sum(t["psnr"] for t in l0_tiles.values()) / len(l0_tiles)
        result.update({
            "type": "baseline",
            "encoder": cfg.get("encoder", "libjpeg-turbo"),
            "quality": cfg.get("jpeg_quality"),
            "total_bytes": total_bytes,
            "avg_l0_psnr": avg_psnr,
            "num_tiles": len(l0_tiles),
        })
        return result

    # ORIGAMI runs (v2/v4)
    tiles = data.get("tiles", [])
    if isinstance(tiles, dict):
        # Dict-format tiles (baseline-like manifest without "type" field)
        l0_tiles_d = {k: v for k, v in tiles.items() if k.startswith("L0_")}
        if not l0_tiles_d:
            return None
        total_bytes = sum(t["size_bytes"] for t in l0_tiles_d.values())
        avg_psnr = sum(t["psnr"] for t in l0_tiles_d.values()) / len(l0_tiles_d)
        result.update({
            "type": "baseline",
            "encoder": data.get("configuration", {}).get("encoder", "unknown"),
            "quality": data.get("configuration", {}).get("jpeg_quality"),
            "total_bytes": total_bytes,
            "avg_l0_psnr": avg_psnr,
            "num_tiles": len(l0_tiles_d),
        })
        return result
    l0_tiles = [t for t in tiles if t.get("level") == "L0"]
    if not l0_tiles:
        return None

    avg_psnr = sum(t["y_psnr_db"] for t in l0_tiles) / len(l0_tiles)
    total_bytes = (data.get("l2_bytes", 0) or 0) + (data.get("fused_l0_bytes", 0) or 0)

    result.update({
        "type": "origami",
        "pipeline_version": data.get("pipeline_version", 2),
        "encoder": data.get("encoder", "?"),
        "total_bytes": total_bytes,
        "avg_l0_psnr": avg_psnr,
        "l2_bytes": data.get("l2_bytes", 0),
        "fused_l0_bytes": data.get("fused_l0_bytes", 0),
        "l0q": data.get("l0q"),
        "num_tiles": len(l0_tiles),
    })

    pv = data.get("pipeline_version", 2)
    if pv == 4:
        result.update({
            "seed_luma_size": data.get("seed_luma_size"),
            "seed_luma_q": data.get("seed_luma_q"),
            "seed_chroma_size": data.get("seed_chroma_size"),
            "seed_chroma_q": data.get("seed_chroma_q"),
            "seed_luma_bytes": data.get("seed_luma_bytes", 0),
            "seed_chroma_bytes": data.get("seed_chroma_bytes", 0),
        })
    else:
        result.update({
            "baseq": data.get("baseq"),
            "seed_size": f"{data.get('l2_w', '?')}x{data.get('l2_h', '?')}",
        })

    return result


def find_pareto(points: list[dict], x_key="total_bytes", y_key="avg_l0_psnr") -> list[dict]:
    """Find Pareto front: minimize bytes, maximize PSNR."""
    sorted_pts = sorted(points, key=lambda p: p[x_key])
    front = []
    best_psnr = -float("inf")
    for p in sorted_pts:
        if p[y_key] > best_psnr:
            front.append(p)
            best_psnr = p[y_key]
    return front


def main():
    all_runs = []
    for d in sorted(RUNS_DIR.iterdir()):
        if not d.is_dir():
            continue
        r = load_run(d)
        if r:
            all_runs.append(r)

    # Categorize
    jpeg_baselines = [r for r in all_runs if r["type"] == "baseline" and r["encoder"] == "libjpeg-turbo"]
    jxl_baselines = [r for r in all_runs if r["type"] == "baseline" and r["encoder"] == "jpegxl"]
    v2_runs = [r for r in all_runs if r["type"] == "origami" and r["pipeline_version"] != 4]
    v4_runs = [r for r in all_runs if r["type"] == "origami" and r["pipeline_version"] == 4]

    print(f"Loaded {len(all_runs)} runs: {len(jpeg_baselines)} JPEG, {len(jxl_baselines)} JXL, {len(v2_runs)} v2, {len(v4_runs)} v4")
    print()

    # === Rate-Quality table ===
    def bytes_to_kb(b):
        return b / 1024

    def print_table(title, runs, extra_cols=None):
        if not runs:
            return
        print(f"=== {title} ({len(runs)} runs) ===")
        sorted_runs = sorted(runs, key=lambda r: r["total_bytes"])
        for r in sorted_runs:
            kb = bytes_to_kb(r["total_bytes"])
            psnr = r["avg_l0_psnr"]
            bpp = r["total_bytes"] * 8 / (1024 * 1024)  # bits per pixel for 1024x1024
            extra = ""
            if extra_cols:
                extra = "  " + "  ".join(str(r.get(c, "")) for c in extra_cols)
            print(f"  {r['name']:60s}  {kb:7.1f} KB  {psnr:5.2f} dB  {bpp:.3f} bpp{extra}")
        print()

    print_table("JPEG Baselines", jpeg_baselines)
    print_table("JPEG-XL Baselines", jxl_baselines)
    print_table("V2 ORIGAMI (single seed)", v2_runs, ["baseq", "l0q"])
    print_table("V4 ORIGAMI (split seed) — Top 30 by PSNR/byte",
                sorted(v4_runs, key=lambda r: r["avg_l0_psnr"] / (r["total_bytes"] / 1024), reverse=True)[:30],
                ["seed_luma_size", "seed_luma_q", "seed_chroma_size", "seed_chroma_q", "l0q"])

    # === Pareto fronts ===
    print("=== Pareto Fronts (minimize bytes, maximize PSNR) ===")
    print()

    all_for_pareto = jpeg_baselines + jxl_baselines + v2_runs + v4_runs
    pareto = find_pareto(all_for_pareto)
    print(f"Global Pareto front ({len(pareto)} points):")
    for p in pareto:
        kb = bytes_to_kb(p["total_bytes"])
        psnr = p["avg_l0_psnr"]
        print(f"  {p['name']:60s}  {kb:7.1f} KB  {psnr:5.2f} dB")
    print()

    v4_pareto = find_pareto(v4_runs)
    print(f"V4-only Pareto front ({len(v4_pareto)} points):")
    for p in v4_pareto:
        kb = bytes_to_kb(p["total_bytes"])
        psnr = p["avg_l0_psnr"]
        luma = f"SL={p.get('seed_luma_q')}@{p.get('seed_luma_size')}"
        chroma = f"SC={p.get('seed_chroma_q')}@{p.get('seed_chroma_size')}"
        res_kb = bytes_to_kb(p.get("fused_l0_bytes", 0))
        seed_kb = bytes_to_kb(p.get("l2_bytes", 0))
        print(f"  {p['name']:60s}  {kb:7.1f} KB  {psnr:5.2f} dB  ({luma} {chroma} L0Q={p.get('l0q')}  seed={seed_kb:.1f}KB res={res_kb:.1f}KB)")
    print()

    # === Comparison: V4 vs JXL at matched byte budgets ===
    if jxl_baselines and v4_runs:
        print("=== V4 vs JPEG-XL at matched byte budgets ===")
        print()
        jxl_sorted = sorted(jxl_baselines, key=lambda r: r["total_bytes"])
        for jxl in jxl_sorted:
            jxl_kb = bytes_to_kb(jxl["total_bytes"])
            jxl_psnr = jxl["avg_l0_psnr"]
            # Find closest v4 run by total bytes
            closest = min(v4_runs, key=lambda r: abs(r["total_bytes"] - jxl["total_bytes"]))
            c_kb = bytes_to_kb(closest["total_bytes"])
            c_psnr = closest["avg_l0_psnr"]
            delta = c_psnr - jxl_psnr
            sign = "+" if delta >= 0 else ""
            print(f"  JXL Q{jxl['quality']:2d}: {jxl_kb:6.1f} KB {jxl_psnr:5.2f} dB  |  Closest V4: {c_kb:6.1f} KB {c_psnr:5.2f} dB  ({sign}{delta:.2f} dB)  [{closest['name']}]")

            # Find best v4 within ±10% of JXL byte budget
            budget_lo = jxl["total_bytes"] * 0.9
            budget_hi = jxl["total_bytes"] * 1.1
            in_budget = [r for r in v4_runs if budget_lo <= r["total_bytes"] <= budget_hi]
            if in_budget:
                best = max(in_budget, key=lambda r: r["avg_l0_psnr"])
                b_kb = bytes_to_kb(best["total_bytes"])
                b_psnr = best["avg_l0_psnr"]
                b_delta = b_psnr - jxl_psnr
                b_sign = "+" if b_delta >= 0 else ""
                print(f"    Best V4 ±10%: {b_kb:6.1f} KB {b_psnr:5.2f} dB  ({b_sign}{b_delta:.2f} dB)  [{best['name']}]")
            else:
                print(f"    No V4 run within ±10% of {jxl_kb:.1f} KB")
        print()

    # === Comparison: V4 vs JXL at matched PSNR ===
    if jxl_baselines and v4_runs:
        print("=== V4 vs JPEG-XL at matched quality (PSNR) ===")
        print()
        jxl_sorted = sorted(jxl_baselines, key=lambda r: r["avg_l0_psnr"])
        for jxl in jxl_sorted:
            jxl_kb = bytes_to_kb(jxl["total_bytes"])
            jxl_psnr = jxl["avg_l0_psnr"]
            # Find v4 runs within ±0.5 dB
            similar_q = [r for r in v4_runs if abs(r["avg_l0_psnr"] - jxl_psnr) < 0.5]
            if similar_q:
                smallest = min(similar_q, key=lambda r: r["total_bytes"])
                s_kb = bytes_to_kb(smallest["total_bytes"])
                savings = (1 - smallest["total_bytes"] / jxl["total_bytes"]) * 100
                sign = "+" if savings < 0 else ""
                print(f"  JXL Q{jxl['quality']:2d}: {jxl_psnr:5.2f} dB @ {jxl_kb:6.1f} KB  |  V4: {smallest['avg_l0_psnr']:5.2f} dB @ {s_kb:6.1f} KB  ({sign}{savings:.1f}% size)  [{smallest['name']}]")
            else:
                print(f"  JXL Q{jxl['quality']:2d}: {jxl_psnr:5.2f} dB @ {jxl_kb:6.1f} KB  |  No V4 within ±0.5 dB")
        print()

    # === Key findings ===
    if v4_runs:
        print("=== Key Findings ===")
        print()

        # Best PSNR/byte ratio
        best_eff = max(v4_runs, key=lambda r: r["avg_l0_psnr"] / (r["total_bytes"] / 1024))
        print(f"Best PSNR/KB: {best_eff['name']}")
        print(f"  {bytes_to_kb(best_eff['total_bytes']):.1f} KB, {best_eff['avg_l0_psnr']:.2f} dB")
        print()

        # Best PSNR overall
        best_q = max(v4_runs, key=lambda r: r["avg_l0_psnr"])
        print(f"Best PSNR: {best_q['name']}")
        print(f"  {bytes_to_kb(best_q['total_bytes']):.1f} KB, {best_q['avg_l0_psnr']:.2f} dB")
        print()

        # Smallest size above 35 dB
        above_35 = [r for r in v4_runs if r["avg_l0_psnr"] >= 35.0]
        if above_35:
            smallest_35 = min(above_35, key=lambda r: r["total_bytes"])
            print(f"Smallest V4 above 35 dB: {smallest_35['name']}")
            print(f"  {bytes_to_kb(smallest_35['total_bytes']):.1f} KB, {smallest_35['avg_l0_psnr']:.2f} dB")
            print()

        # Effect of luma size (aggregated)
        print("Effect of luma seed size (avg across all other params):")
        for ls in [128, 256, 384, 512]:
            subset = [r for r in v4_runs if r.get("seed_luma_size") == ls]
            if subset:
                avg_p = sum(r["avg_l0_psnr"] for r in subset) / len(subset)
                avg_kb = sum(bytes_to_kb(r["total_bytes"]) for r in subset) / len(subset)
                avg_seed = sum(bytes_to_kb(r.get("l2_bytes", 0)) for r in subset) / len(subset)
                avg_res = sum(bytes_to_kb(r.get("fused_l0_bytes", 0)) for r in subset) / len(subset)
                print(f"  LS={ls:3d}: {avg_p:.2f} dB  {avg_kb:.1f} KB total  (seed={avg_seed:.1f}KB + res={avg_res:.1f}KB)  n={len(subset)}")
        print()

        print("Effect of chroma seed size (avg across all other params):")
        for cs in sorted(set(r.get("seed_chroma_size") for r in v4_runs if r.get("seed_chroma_size"))):
            subset = [r for r in v4_runs if r.get("seed_chroma_size") == cs]
            if subset:
                avg_p = sum(r["avg_l0_psnr"] for r in subset) / len(subset)
                avg_kb = sum(bytes_to_kb(r["total_bytes"]) for r in subset) / len(subset)
                avg_chroma_kb = sum(bytes_to_kb(r.get("seed_chroma_bytes", 0)) for r in subset) / len(subset)
                print(f"  CS={cs:3d}: {avg_p:.2f} dB  {avg_kb:.1f} KB total  (chroma seed={avg_chroma_kb:.1f}KB)  n={len(subset)}")
        print()

        print("Effect of chroma seed quality (avg across all other params):")
        for cq in sorted(set(r.get("seed_chroma_q") for r in v4_runs if r.get("seed_chroma_q"))):
            subset = [r for r in v4_runs if r.get("seed_chroma_q") == cq]
            if subset:
                avg_p = sum(r["avg_l0_psnr"] for r in subset) / len(subset)
                avg_kb = sum(bytes_to_kb(r["total_bytes"]) for r in subset) / len(subset)
                avg_chroma_kb = sum(bytes_to_kb(r.get("seed_chroma_bytes", 0)) for r in subset) / len(subset)
                print(f"  CQ={cq:2d}: {avg_p:.2f} dB  {avg_kb:.1f} KB total  (chroma seed={avg_chroma_kb:.1f}KB)  n={len(subset)}")
        print()

        print("Effect of luma seed quality (avg across all other params):")
        for lq in sorted(set(r.get("seed_luma_q") for r in v4_runs if r.get("seed_luma_q"))):
            subset = [r for r in v4_runs if r.get("seed_luma_q") == lq]
            if subset:
                avg_p = sum(r["avg_l0_psnr"] for r in subset) / len(subset)
                avg_kb = sum(bytes_to_kb(r["total_bytes"]) for r in subset) / len(subset)
                avg_luma_kb = sum(bytes_to_kb(r.get("seed_luma_bytes", 0)) for r in subset) / len(subset)
                print(f"  LQ={lq:2d}: {avg_p:.2f} dB  {avg_kb:.1f} KB total  (luma seed={avg_luma_kb:.1f}KB)  n={len(subset)}")
        print()

        print("Effect of L0 residual quality (avg across all other params):")
        for l0q in sorted(set(r.get("l0q") for r in v4_runs if r.get("l0q"))):
            subset = [r for r in v4_runs if r.get("l0q") == l0q]
            if subset:
                avg_p = sum(r["avg_l0_psnr"] for r in subset) / len(subset)
                avg_kb = sum(bytes_to_kb(r["total_bytes"]) for r in subset) / len(subset)
                avg_res = sum(bytes_to_kb(r.get("fused_l0_bytes", 0)) for r in subset) / len(subset)
                print(f"  L0Q={l0q:2d}: {avg_p:.2f} dB  {avg_kb:.1f} KB total  (res={avg_res:.1f}KB)  n={len(subset)}")
        print()

        # Chroma size × quality interaction table
        print("Chroma size × quality interaction (avg PSNR / avg chroma KB):")
        cs_vals = sorted(set(r.get("seed_chroma_size") for r in v4_runs if r.get("seed_chroma_size")))
        cq_vals = sorted(set(r.get("seed_chroma_q") for r in v4_runs if r.get("seed_chroma_q")))
        header = "          " + "".join(f"  CQ={cq:2d}        " for cq in cq_vals)
        print(header)
        for cs in cs_vals:
            row = f"  CS={cs:3d}:  "
            for cq in cq_vals:
                subset = [r for r in v4_runs if r.get("seed_chroma_size") == cs and r.get("seed_chroma_q") == cq]
                if subset:
                    avg_p = sum(r["avg_l0_psnr"] for r in subset) / len(subset)
                    avg_ckb = sum(bytes_to_kb(r.get("seed_chroma_bytes", 0)) for r in subset) / len(subset)
                    row += f"{avg_p:.2f}dB/{avg_ckb:.1f}KB  "
                else:
                    row += "  ---          "
            print(row)
        print()

        # Luma size × quality interaction table
        print("Luma size × quality interaction (avg PSNR / avg luma KB):")
        ls_vals = sorted(set(r.get("seed_luma_size") for r in v4_runs if r.get("seed_luma_size")))
        lq_vals = sorted(set(r.get("seed_luma_q") for r in v4_runs if r.get("seed_luma_q")))
        header = "          " + "".join(f"  LQ={lq:2d}         " for lq in lq_vals)
        print(header)
        for ls in ls_vals:
            row = f"  LS={ls:3d}:  "
            for lq in lq_vals:
                subset = [r for r in v4_runs if r.get("seed_luma_size") == ls and r.get("seed_luma_q") == lq]
                if subset:
                    avg_p = sum(r["avg_l0_psnr"] for r in subset) / len(subset)
                    avg_lkb = sum(bytes_to_kb(r.get("seed_luma_bytes", 0)) for r in subset) / len(subset)
                    row += f"{avg_p:.2f}dB/{avg_lkb:.1f}KB  "
                else:
                    row += "  ---           "
            print(row)
        print()


if __name__ == "__main__":
    main()
