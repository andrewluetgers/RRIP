#!/usr/bin/env python3
"""
Query the GDC (Genomic Data Commons) API for all TCGA whole-slide image metadata.

Fetches metadata for Diagnostic Slide and Tissue Slide files from TCGA projects,
saves full results as JSON, and prints summary statistics.

No files are downloaded — metadata only.

Usage:
    python tcga_metadata.py [--output tcga_slides_metadata.json]
"""

import argparse
import json
import os
import statistics
import sys
from collections import Counter
from datetime import datetime

import requests

GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
PAGE_SIZE = 1000  # GDC allows up to 10000, but 1000 is safer/faster


def build_filters():
    """Build GDC API filter to get all TCGA slide image files.

    Uses data_type="Slide Image" to match all slides. The experimental_strategy
    field distinguishes "Diagnostic Slide" vs "Tissue Slide".
    """
    return {
        "op": "and",
        "content": [
            {
                "op": "=",
                "content": {
                    "field": "cases.project.program.name",
                    "value": "TCGA",
                },
            },
            {
                "op": "=",
                "content": {
                    "field": "data_type",
                    "value": "Slide Image",
                },
            },
        ],
    }


# Fields to retrieve for each file
FIELDS = [
    "file_id",
    "file_name",
    "file_size",
    "data_type",
    "data_format",
    "data_category",
    "experimental_strategy",
    "access",
    "state",
    "cases.project.project_id",
    "cases.project.name",
    "cases.project.primary_site",
    "cases.project.program.name",
    "cases.samples.sample_type",
    "cases.samples.tissue_type",
    "cases.samples.sample_id",
    "cases.samples.portions.slides.section_location",
    "cases.samples.portions.slides.slide_id",
    "cases.case_id",
    "cases.submitter_id",
]


def query_gdc_slides():
    """Query GDC API for all TCGA slide files with pagination."""
    filters = build_filters()
    all_hits = []
    offset = 0
    total = None

    print("Querying GDC API for TCGA slide image metadata...")
    print(f"Endpoint: {GDC_FILES_ENDPOINT}")
    print(f"Page size: {PAGE_SIZE}")
    print()

    while True:
        params = {
            "filters": json.dumps(filters),
            "fields": ",".join(FIELDS),
            "size": PAGE_SIZE,
            "from": offset,
            "format": "JSON",
        }

        response = requests.get(GDC_FILES_ENDPOINT, params=params, timeout=120)
        response.raise_for_status()
        data = response.json()

        pagination = data["data"]["pagination"]
        hits = data["data"]["hits"]

        if total is None:
            total = pagination["total"]
            print(f"Total slides found: {total}")

        all_hits.extend(hits)
        fetched = len(all_hits)
        print(f"  Fetched {fetched}/{total} records (page {pagination['page']})")

        if fetched >= total:
            break

        offset += PAGE_SIZE

    print(f"\nDone. Retrieved {len(all_hits)} file records.")
    return all_hits


def extract_field(hit, path, default=None):
    """Extract a potentially nested field, handling cases/samples arrays."""
    parts = path.split(".")
    current = hit
    for part in parts:
        if current is None:
            return default
        if isinstance(current, list):
            # Collect from all items in the list
            results = []
            for item in current:
                val = extract_field(item, ".".join([part] + parts[parts.index(part) + 1 :]), default)
                if isinstance(val, list):
                    results.extend(val)
                elif val is not None and val != default:
                    results.append(val)
            return results if results else default
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return default
    return current


def flatten_to_list(val):
    """Ensure a value is a flat list."""
    if val is None:
        return []
    if isinstance(val, list):
        flat = []
        for item in val:
            if isinstance(item, list):
                flat.extend(item)
            else:
                flat.append(item)
        return flat
    return [val]


def print_summary(hits):
    """Print summary statistics from the retrieved metadata."""
    print("\n" + "=" * 70)
    print("TCGA SLIDE IMAGE METADATA SUMMARY")
    print("=" * 70)

    # --- Total count ---
    print(f"\nTotal slide files: {len(hits)}")

    # --- By data_type ---
    data_types = Counter(h.get("data_type", "unknown") for h in hits)
    print("\n--- Slides by data_type ---")
    for dt, count in data_types.most_common():
        print(f"  {dt}: {count}")

    # --- By experimental_strategy ---
    strategies = Counter(h.get("experimental_strategy", "unknown") for h in hits)
    print("\n--- Slides by experimental_strategy ---")
    for s, count in strategies.most_common():
        print(f"  {s}: {count}")

    # --- By data_format ---
    formats = Counter(h.get("data_format", "unknown") for h in hits)
    print("\n--- Slides by data_format ---")
    for f, count in formats.most_common():
        print(f"  {f}: {count}")

    # --- By access type ---
    access_types = Counter(h.get("access", "unknown") for h in hits)
    print("\n--- Slides by access type ---")
    for a, count in access_types.most_common():
        print(f"  {a}: {count}")

    # --- By project_id (cancer type) ---
    project_counter = Counter()
    for h in hits:
        cases = h.get("cases", [])
        if cases:
            for case in cases:
                proj = case.get("project", {})
                pid = proj.get("project_id", "unknown")
                project_counter[pid] += 1
        else:
            project_counter["unknown"] += 1

    print(f"\n--- Slides per cancer type (project_id) --- [{len(project_counter)} projects]")
    for pid, count in project_counter.most_common():
        print(f"  {pid}: {count}")

    # --- By primary_site ---
    site_counter = Counter()
    for h in hits:
        cases = h.get("cases", [])
        if cases:
            for case in cases:
                proj = case.get("project", {})
                site = proj.get("primary_site", "unknown")
                site_counter[site] += 1
        else:
            site_counter["unknown"] += 1

    print(f"\n--- Slides per primary_site (organ/tissue) --- [{len(site_counter)} sites]")
    for site, count in site_counter.most_common():
        print(f"  {site}: {count}")

    # --- By sample_type ---
    sample_type_counter = Counter()
    for h in hits:
        cases = h.get("cases", [])
        if cases:
            for case in cases:
                samples = case.get("samples", [])
                if samples:
                    for sample in samples:
                        st = sample.get("sample_type", "unknown")
                        sample_type_counter[st] += 1
                else:
                    sample_type_counter["unknown"] += 1
        else:
            sample_type_counter["unknown"] += 1

    print(f"\n--- Slides per sample_type --- [{len(sample_type_counter)} types]")
    for st, count in sample_type_counter.most_common():
        print(f"  {st}: {count}")

    # --- By tissue_type ---
    tissue_type_counter = Counter()
    for h in hits:
        cases = h.get("cases", [])
        if cases:
            for case in cases:
                samples = case.get("samples", [])
                if samples:
                    for sample in samples:
                        tt = sample.get("tissue_type", "unknown")
                        if tt is None:
                            tt = "N/A"
                        tissue_type_counter[tt] += 1
                else:
                    tissue_type_counter["unknown"] += 1
        else:
            tissue_type_counter["unknown"] += 1

    print(f"\n--- Slides per tissue_type --- [{len(tissue_type_counter)} types]")
    for tt, count in tissue_type_counter.most_common():
        print(f"  {tt}: {count}")

    # --- File size distribution ---
    sizes = [h.get("file_size", 0) for h in hits if h.get("file_size")]
    if sizes:
        total_bytes = sum(sizes)
        print("\n--- File size distribution ---")
        print(f"  Count:  {len(sizes)}")
        print(f"  Min:    {min(sizes) / 1e6:.1f} MB")
        print(f"  Median: {statistics.median(sizes) / 1e6:.1f} MB")
        print(f"  Mean:   {statistics.mean(sizes) / 1e6:.1f} MB")
        print(f"  Max:    {max(sizes) / 1e6:.1f} MB ({max(sizes) / 1e9:.2f} GB)")
        print(f"  Total:  {total_bytes / 1e12:.2f} TB ({total_bytes / 1e9:.1f} GB)")

    # --- Stain info ---
    print("\n--- Stain / section_location info ---")
    section_locations = Counter()
    slide_count = 0
    for h in hits:
        cases = h.get("cases", [])
        for case in cases:
            samples = case.get("samples", [])
            for sample in samples:
                portions = sample.get("portions", [])
                for portion in portions:
                    slides = portion.get("slides", [])
                    for slide in slides:
                        slide_count += 1
                        loc = slide.get("section_location", "N/A")
                        section_locations[loc] += 1

    if section_locations:
        print(f"  Slides with section_location data: {slide_count}")
        for loc, count in section_locations.most_common():
            print(f"    {loc}: {count}")
    else:
        print("  No section_location data found in slide metadata.")

    print("\n  Note: TCGA slides are predominantly H&E stained.")
    print("  The GDC API does not expose a dedicated 'stain_type' field.")
    print("  IHC slides are generally NOT part of TCGA.")

    # --- File name patterns (check for stain hints) ---
    extensions = Counter()
    for h in hits:
        fname = h.get("file_name", "")
        ext = os.path.splitext(fname)[1].lower()
        extensions[ext] += 1

    print(f"\n--- File extensions ---")
    for ext, count in extensions.most_common():
        print(f"  {ext}: {count}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Query GDC API for TCGA slide image metadata")
    parser.add_argument(
        "--output",
        "-o",
        default="tcga_slides_metadata.json",
        help="Output JSON file path (default: tcga_slides_metadata.json)",
    )
    args = parser.parse_args()

    # Resolve output path relative to script directory if not absolute
    if not os.path.isabs(args.output):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, args.output)
    else:
        output_path = args.output

    hits = query_gdc_slides()

    # Save full results
    output_data = {
        "query_date": datetime.utcnow().isoformat() + "Z",
        "total_files": len(hits),
        "api_endpoint": GDC_FILES_ENDPOINT,
        "fields_requested": FIELDS,
        "files": hits,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved full metadata to: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1e6:.1f} MB")

    # Print summary
    print_summary(hits)


if __name__ == "__main__":
    main()
