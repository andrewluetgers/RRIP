#!/usr/bin/env python3
"""
Generate a markdown report from vegeta benchmark results.

Reads vegeta .bin files from a results directory and produces a formatted
markdown report with latency tables and cache stress analysis.

Usage:
    uv run python evals/scripts/bench_report.py [results_dir]

    # Or auto-detect latest:
    uv run python evals/scripts/bench_report.py
"""

import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def parse_vegeta_report(bin_path: str) -> dict:
    """Run vegeta report on a .bin file and parse the output."""
    try:
        txt = subprocess.check_output(
            ["vegeta", "report", bin_path], text=True, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {}

    result = {}

    # Parse latencies
    m = re.search(
        r'Latencies\s+\[.*?\]\s+([\d.]+[µm]?s),\s*([\d.]+[µm]?s),\s*'
        r'([\d.]+[µm]?s),\s*([\d.]+[µm]?s),\s*([\d.]+[µm]?s),\s*'
        r'([\d.]+[µm]?s),\s*([\d.]+[µm]?s)', txt)
    if m:
        result['min'] = m.group(1)
        result['mean'] = m.group(2)
        result['p50'] = m.group(3)
        result['p90'] = m.group(4)
        result['p95'] = m.group(5)
        result['p99'] = m.group(6)
        result['max'] = m.group(7)

    # Parse requests
    m = re.search(r'Requests\s+\[total.*?\]\s+(\d+)', txt)
    if m:
        result['total'] = int(m.group(1))

    # Parse throughput
    m = re.search(r'throughput\]\s+[\d.]+,\s*[\d.]+,\s*([\d.]+)', txt)
    if m:
        result['throughput'] = float(m.group(1))

    # Parse success ratio
    m = re.search(r'Success\s+\[ratio\]\s+([\d.]+)%', txt)
    if m:
        result['success_pct'] = float(m.group(1))

    # Parse status codes
    codes = {}
    for code, count in re.findall(r'(\d+):(\d+)', txt):
        codes[int(code)] = int(count)
    result['codes'] = codes
    result['ok'] = codes.get(200, 0)
    result['not_found'] = codes.get(404, 0)
    result['errors'] = codes.get(500, 0)

    # Get histogram
    try:
        hist = subprocess.check_output(
            ["vegeta", "report", "-type=hist",
             "-buckets=[0,1ms,2ms,5ms,10ms,20ms,50ms,100ms,500ms]",
             bin_path], text=True, stderr=subprocess.DEVNULL)
        result['histogram'] = hist.strip()
    except Exception:
        pass

    return result


def latency_to_ms(s: str) -> str:
    """Convert vegeta latency string to clean ms representation."""
    if 'µs' in s:
        val = float(s.replace('µs', ''))
        return f'{val/1000:.1f}ms'
    elif 'ms' in s:
        return s
    elif 's' in s:
        val = float(s.replace('s', ''))
        return f'{val*1000:.0f}ms'
    return s


def generate_report(results_dir: str) -> str:
    """Generate markdown report from a results directory."""
    lines = []
    lines.append("# ORIGAMI Tile Server Benchmark Report\n")
    lines.append(f"Results: `{results_dir}`\n")

    # Find all .bin files
    bins = sorted(glob.glob(os.path.join(results_dir, "*.bin")))
    if not bins:
        lines.append("No vegeta .bin files found.\n")
        return "\n".join(lines)

    # Categorize files
    rate_tests = {}  # variant -> rate -> result
    stress_tests = {}  # variant -> hit_pct -> result

    # Try to determine variant from log files
    for log in glob.glob(os.path.join(results_dir, "*.log")):
        variant = Path(log).stem  # e.g., 'jxl80'

    for bin_path in bins:
        name = Path(bin_path).stem

        # Rate test: result_r50, result_r100, etc.
        m = re.match(r'result_r(\d+)', name)
        if m:
            rate = int(m.group(1))
            result = parse_vegeta_report(bin_path)
            if result:
                rate_tests.setdefault('combined', {})[rate] = result
            continue

        # Stress test: stress_0hit, stress_25hit, etc.
        m = re.match(r'stress_(\d+)hit', name)
        if m:
            hit_pct = int(m.group(1))
            result = parse_vegeta_report(bin_path)
            if result:
                stress_tests.setdefault('combined', {})[hit_pct] = result
            continue

    # Rate sweep table
    if rate_tests:
        lines.append("## Rate Sweep\n")
        lines.append("| Rate | Total | 200s | 404s | 500s | P50 | P95 | P99 | Max | Throughput |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")

        for variant, rates in sorted(rate_tests.items()):
            for rate in sorted(rates):
                r = rates[rate]
                lines.append(
                    f"| {rate}/s | {r.get('total', '?')} | {r.get('ok', 0)} | "
                    f"{r.get('not_found', 0)} | {r.get('errors', 0)} | "
                    f"{latency_to_ms(r.get('p50', '?'))} | "
                    f"{latency_to_ms(r.get('p95', '?'))} | "
                    f"{latency_to_ms(r.get('p99', '?'))} | "
                    f"{latency_to_ms(r.get('max', '?'))} | "
                    f"{r.get('throughput', '?'):.0f}/s |")
        lines.append("")

    # Cache stress table
    if stress_tests:
        lines.append("## Cache Stress (200 req/s, L1/L2 cold misses)\n")
        lines.append("| Hit Rate | Total | 200s | 404s | 500s | P50 | P95 | P99 | Max |")
        lines.append("|---|---|---|---|---|---|---|---|---|")

        for variant, hits in sorted(stress_tests.items()):
            for hit_pct in sorted(hits):
                r = hits[hit_pct]
                lines.append(
                    f"| {hit_pct}% | {r.get('total', '?')} | {r.get('ok', 0)} | "
                    f"{r.get('not_found', 0)} | {r.get('errors', 0)} | "
                    f"{latency_to_ms(r.get('p50', '?'))} | "
                    f"{latency_to_ms(r.get('p95', '?'))} | "
                    f"{latency_to_ms(r.get('p99', '?'))} | "
                    f"{latency_to_ms(r.get('max', '?'))} |")
        lines.append("")

    # Histograms
    if stress_tests:
        lines.append("## Latency Distributions\n")
        for variant, hits in sorted(stress_tests.items()):
            for hit_pct in sorted(hits):
                r = hits[hit_pct]
                if 'histogram' in r:
                    lines.append(f"### {hit_pct}% Cache Hit Rate\n")
                    lines.append("```")
                    lines.append(r['histogram'])
                    lines.append("```\n")

    return "\n".join(lines)


def main():
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        # Find latest bench results
        candidates = sorted(glob.glob("evals/runs/bench_vegeta_*") +
                          glob.glob("evals/runs/bench_suite_*"),
                          reverse=True)
        if not candidates:
            print("No benchmark results found")
            sys.exit(1)
        results_dir = candidates[0]

    print(f"Generating report from: {results_dir}")
    report = generate_report(results_dir)

    # Write report
    report_path = os.path.join(results_dir, "REPORT.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report written: {report_path}")

    # Also print to stdout
    print("\n" + report)


if __name__ == "__main__":
    main()
