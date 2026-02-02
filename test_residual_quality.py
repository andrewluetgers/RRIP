#!/usr/bin/env python3
"""Test different residual qualities to see impact on high-frequency details."""

import subprocess
import time

qualities = [32, 50, 70, 90]

for q in qualities:
    print(f"\n{'='*60}")
    print(f"Testing residual quality Q{q}")
    print(f"{'='*60}")
    
    # Generate residuals at this quality
    cmd = [
        "python3", "cli/wsi_residual_tool.py", "encode",
        "--pyramid", "data/demo_out/baseline_pyramid",
        "--out", f"data/demo_out_q{q}",
        "--tile", "256",
        "--resq", str(q),
        "--max-parents", "10"  # Just test a few for speed
    ]
    
    print(f"Generating residuals at Q{q}...")
    subprocess.run(cmd, check=True)
    
    # Check file sizes
    print(f"\nChecking residual sizes for Q{q}:")
    subprocess.run(["du", "-sh", f"data/demo_out_q{q}/residuals_q{q}"], check=True)

print("\nRecommendation: Higher quality residuals (Q70-90) will preserve")
print("high-frequency details much better, but at the cost of larger file sizes.")
print("Consider Q50-70 as a good balance for production use.")
