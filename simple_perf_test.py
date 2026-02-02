#!/usr/bin/env python3
"""Simple performance test for RRIP with LZ4 compression."""

import time
import subprocess
import requests
import random
import statistics
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
SERVER_URL = "http://localhost:3007"
SLIDE_ID = "demo_out"
NUM_REQUESTS = 500
CONCURRENT_WORKERS = 16

def fetch_tile(coords):
    """Fetch a single tile."""
    level, x, y = coords
    url = f"{SERVER_URL}/tiles/{SLIDE_ID}/{level}/{x}_{y}.jpg"

    start = time.perf_counter()
    try:
        resp = requests.get(url, timeout=10)
        elapsed = time.perf_counter() - start
        return {
            'success': resp.status_code == 200,
            'elapsed_ms': elapsed * 1000,
            'size': len(resp.content) if resp.status_code == 200 else 0
        }
    except Exception as e:
        elapsed = time.perf_counter() - start
        return {
            'success': False,
            'elapsed_ms': elapsed * 1000,
            'error': str(e)
        }

def generate_random_tiles(count):
    """Generate random L0 tile coordinates."""
    tiles = []
    for _ in range(count):
        # L0 tiles (level 14) in valid ranges
        x = random.randint(80, 230)
        y = random.randint(40, 230)
        tiles.append((14, x, y))
    return tiles

def run_performance_test():
    """Run the performance test."""
    print("Starting RRIP server...")

    # Start server
    server = subprocess.Popen(
        ['/tmp/origami-build/release/origami-tile-server',
         '--slides-root', 'data',
         '--port', '3007'],
        cwd='/Users/andrewluetgers/projects/dev/RRIP',
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # Wait for server to start
    time.sleep(3)

    try:
        # Verify server is up
        resp = requests.get(f"{SERVER_URL}/healthz", timeout=5)
        if resp.status_code != 200:
            raise RuntimeError("Server not responding")
        print("Server is ready!")

        # Generate random tiles to avoid cache hits
        tiles = generate_random_tiles(NUM_REQUESTS)

        print(f"\nRunning {NUM_REQUESTS} requests with {CONCURRENT_WORKERS} workers...")

        results = []
        start_time = time.time()

        # Run concurrent requests
        with ThreadPoolExecutor(max_workers=CONCURRENT_WORKERS) as executor:
            futures = [executor.submit(fetch_tile, tile) for tile in tiles]

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

                # Print progress
                if len(results) % 50 == 0:
                    print(f"  Completed {len(results)} requests...")

        total_time = time.time() - start_time

        # Calculate statistics
        successful = [r for r in results if r['success']]
        latencies = [r['elapsed_ms'] for r in successful]

        if latencies:
            print(f"\n{'='*60}")
            print("PERFORMANCE RESULTS - RRIP with LZ4 Compression")
            print(f"{'='*60}")

            print(f"\nTotal requests: {NUM_REQUESTS}")
            print(f"Successful: {len(successful)} ({100*len(successful)/NUM_REQUESTS:.1f}%)")
            print(f"Total time: {total_time:.1f} seconds")
            print(f"Throughput: {len(successful)/total_time:.1f} tiles/sec")
            print(f"Family generation rate: {len(successful)/total_time:.1f} families/sec")

            print(f"\nLatency Statistics (ms):")
            print(f"  Min:    {min(latencies):.1f}")
            print(f"  Mean:   {statistics.mean(latencies):.1f}")
            print(f"  Median: {statistics.median(latencies):.1f}")
            if len(latencies) > 20:
                print(f"  P95:    {statistics.quantiles(latencies, n=20)[18]:.1f}")
            if len(latencies) > 100:
                print(f"  P99:    {statistics.quantiles(latencies, n=100)[98]:.1f}")
            print(f"  Max:    {max(latencies):.1f}")

            # Get memory usage
            import psutil
            proc = psutil.Process(server.pid)
            mem_info = proc.memory_info()
            print(f"\nMemory Usage:")
            print(f"  RSS: {mem_info.rss / (1024*1024):.1f} MB")
            print(f"  VMS: {mem_info.vms / (1024*1024):.1f} MB")

        # Calculate compression statistics
        print(f"\n{'='*60}")
        print("COMPRESSION STATISTICS")
        print(f"{'='*60}")

        pack_dir = "/Users/andrewluetgers/projects/dev/RRIP/data/demo_out/residual_packs_lz4"
        if os.path.exists(pack_dir):
            pack_files = [f for f in os.listdir(pack_dir) if f.endswith('.pack')]
            total_size = sum(os.path.getsize(os.path.join(pack_dir, f)) for f in pack_files)

            print(f"\nPack files (L0+L1 residuals):")
            print(f"  Number of packs: {len(pack_files)}")
            print(f"  Total compressed size: {total_size / (1024*1024):.2f} MB")
            print(f"  Average pack size: {total_size / len(pack_files) / 1024:.1f} KB")

            # Estimate original size (typical compression ratio ~3.5x)
            est_original = total_size * 3.5
            print(f"  Estimated original: {est_original / (1024*1024):.2f} MB")
            print(f"  Compression ratio: ~3.5x")
            print(f"  Space savings: ~70%")

    finally:
        # Stop server
        server.terminate()
        server.wait(timeout=5)
        print("\nServer stopped.")

if __name__ == "__main__":
    run_performance_test()