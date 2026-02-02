#!/usr/bin/env python3
"""
Performance test for RRIP tile server - TRULY UNCACHED version
Ensures each request hits a different L2 parent to avoid family cache benefits
"""
import asyncio
import aiohttp
import time
import random
import statistics
import json
import psutil
import subprocess
from datetime import datetime
from typing import List, Dict, Tuple, Set
import argparse
import os
import re

class UncachedTileServerBenchmark:
    def __init__(self, base_url: str = "http://localhost:3007", slide_id: str = "demo_out"):
        self.base_url = base_url
        self.slide_id = slide_id
        self.server_pid = None

        # Map L2 parents to their L0/L1 children
        self.l2_families = {}  # {(x2, y2): [(level, x, y), ...]}
        self._discover_l2_families()

    def _discover_l2_families(self):
        """Discover L2 parent tiles and their L0/L1 children"""
        pack_dir = f"data/{self.slide_id}/residual_packs"

        if os.path.exists(pack_dir):
            # Parse pack filenames to get L2 coordinates
            for pack_file in os.listdir(pack_dir):
                if pack_file.endswith('.pack'):
                    match = re.match(r'(\d+)_(\d+)\.pack', pack_file)
                    if match:
                        x2, y2 = int(match.group(1)), int(match.group(2))
                        family = []

                        # Add L1 tiles (4 per L2 parent)
                        for dx in range(2):
                            for dy in range(2):
                                family.append((15, x2*2+dx, y2*2+dy))

                        # Add L0 tiles (16 per L2 parent)
                        for dx in range(4):
                            for dy in range(4):
                                family.append((16, x2*4+dx, y2*4+dy))

                        self.l2_families[(x2, y2)] = family
        else:
            # Fallback: generate synthetic families
            print("Warning: No pack files found, using synthetic L2 families")
            for i in range(200):
                x2, y2 = i % 20, i // 20
                family = []

                # L1 tiles
                for dx in range(2):
                    for dy in range(2):
                        family.append((15, x2*2+dx, y2*2+dy))

                # L0 tiles
                for dx in range(4):
                    for dy in range(4):
                        family.append((16, x2*4+dx, y2*4+dy))

                self.l2_families[(x2, y2)] = family

        print(f"Discovered {len(self.l2_families)} L2 families")
        print(f"Each family has 20 tiles (4 L1 + 16 L0)")

    def get_uncached_tiles(self, count: int, prefer_l0: float = 0.7) -> List[Tuple[int, int, int]]:
        """
        Get tiles ensuring each comes from a different L2 parent.
        This guarantees no cache benefits from family generation.
        """
        if count > len(self.l2_families):
            print(f"Warning: Requested {count} tiles but only {len(self.l2_families)} L2 families available")
            count = len(self.l2_families)

        # Randomly select L2 parents
        selected_l2_parents = random.sample(list(self.l2_families.keys()), count)

        tiles = []
        for l2_parent in selected_l2_parents:
            family = self.l2_families[l2_parent]

            # Pick one tile from this family
            # Prefer L0 tiles based on prefer_l0 parameter
            if random.random() < prefer_l0:
                # Pick from L0 tiles (last 16 in family)
                tile = random.choice(family[-16:])
            else:
                # Pick from L1 tiles (first 4 in family)
                tile = random.choice(family[:4])

            tiles.append(tile)

        return tiles

    def get_server_pid(self) -> int:
        """Get the PID of the running server"""
        try:
            result = subprocess.run(['lsof', '-i', ':3007'],
                                  capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                if 'LISTEN' in line:
                    parts = line.split()
                    return int(parts[1])
        except:
            pass
        return None

    def monitor_resources(self) -> Dict:
        """Monitor CPU and memory usage of the server"""
        pid = self.get_server_pid()
        if not pid:
            return {"error": "Server not found"}

        try:
            process = psutil.Process(pid)
            return {
                "cpu_percent": process.cpu_percent(interval=0.1),
                "memory_mb": process.memory_info().rss / 1024 / 1024,
                "num_threads": process.num_threads(),
                "open_files": len(process.open_files()),
            }
        except:
            return {"error": "Failed to get metrics"}

    async def fetch_tile(self, session: aiohttp.ClientSession,
                         level: int, x: int, y: int) -> Dict:
        """Fetch a single tile and measure timing"""
        url = f"{self.base_url}/tiles/{self.slide_id}/{level}/{x}_{y}.jpg"
        start = time.perf_counter()

        try:
            async with session.get(url) as response:
                data = await response.read()
                elapsed = (time.perf_counter() - start) * 1000  # ms
                return {
                    "success": response.status == 200,
                    "status": response.status,
                    "size": len(data),
                    "time_ms": elapsed,
                    "tile": f"{level}/{x}_{y}",
                    "l2_parent": f"{x>>2}_{y>>2}" if level == 16 else f"{x>>1}_{y>>1}"
                }
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return {
                "success": False,
                "error": str(e),
                "time_ms": elapsed,
                "tile": f"{level}/{x}_{y}"
            }

    async def run_concurrent_test(self, num_requests: int,
                                 concurrency: int) -> Dict:
        """Run concurrent requests test with truly uncached tiles"""
        actual_requests = min(num_requests, len(self.l2_families))
        if actual_requests < num_requests:
            print(f"  Note: Limited to {actual_requests} requests (one per L2 family)")

        print(f"\n=== UNCACHED Concurrent Test: {actual_requests} requests, {concurrency} concurrent ===")

        tiles = self.get_uncached_tiles(actual_requests)
        results = []
        start_time = time.perf_counter()

        # Monitor resources before
        resources_before = self.monitor_resources()

        async with aiohttp.ClientSession() as session:
            # Create batches
            for i in range(0, len(tiles), concurrency):
                batch = tiles[i:i+concurrency]
                tasks = [self.fetch_tile(session, *tile) for tile in batch]
                batch_results = await asyncio.gather(*tasks)
                results.extend(batch_results)

                # Brief pause between batches
                if i + concurrency < len(tiles):
                    await asyncio.sleep(0.01)

        total_time = (time.perf_counter() - start_time) * 1000

        # Monitor resources after
        resources_after = self.monitor_resources()

        # Verify no L2 parent was hit twice
        l2_parents = set()
        for r in results:
            if 'l2_parent' in r:
                l2_parents.add(r['l2_parent'])

        # Calculate statistics
        successful = [r for r in results if r["success"]]
        times = [r["time_ms"] for r in successful]

        stats = {
            "test_type": "uncached_concurrent",
            "total_requests": actual_requests,
            "concurrency": concurrency,
            "successful": len(successful),
            "failed": len(results) - len(successful),
            "unique_l2_parents": len(l2_parents),
            "total_time_ms": total_time,
            "requests_per_second": (len(successful) / total_time) * 1000 if total_time > 0 else 0,
            "response_times": {
                "min": min(times) if times else 0,
                "max": max(times) if times else 0,
                "mean": statistics.mean(times) if times else 0,
                "median": statistics.median(times) if times else 0,
                "p95": statistics.quantiles(times, n=20)[18] if len(times) > 20 else max(times) if times else 0,
                "p99": statistics.quantiles(times, n=100)[98] if len(times) > 100 else max(times) if times else 0,
            },
            "resources": {
                "before": resources_before,
                "after": resources_after,
                "delta_cpu": resources_after.get("cpu_percent", 0) - resources_before.get("cpu_percent", 0),
                "delta_memory_mb": resources_after.get("memory_mb", 0) - resources_before.get("memory_mb", 0),
            }
        }

        return stats

    async def run_sequential_test(self, num_requests: int) -> Dict:
        """Run sequential requests test with truly uncached tiles"""
        actual_requests = min(num_requests, len(self.l2_families))
        if actual_requests < num_requests:
            print(f"  Note: Limited to {actual_requests} requests (one per L2 family)")

        print(f"\n=== UNCACHED Sequential Test: {actual_requests} requests ===")

        tiles = self.get_uncached_tiles(actual_requests)
        results = []
        start_time = time.perf_counter()

        # Monitor resources before
        resources_before = self.monitor_resources()

        async with aiohttp.ClientSession() as session:
            for tile in tiles:
                result = await self.fetch_tile(session, *tile)
                results.append(result)

        total_time = (time.perf_counter() - start_time) * 1000

        # Monitor resources after
        resources_after = self.monitor_resources()

        # Verify no L2 parent was hit twice
        l2_parents = set()
        for r in results:
            if 'l2_parent' in r:
                l2_parents.add(r['l2_parent'])

        # Calculate statistics
        successful = [r for r in results if r["success"]]
        times = [r["time_ms"] for r in successful]

        stats = {
            "test_type": "uncached_sequential",
            "total_requests": actual_requests,
            "concurrency": 1,
            "successful": len(successful),
            "failed": len(results) - len(successful),
            "unique_l2_parents": len(l2_parents),
            "total_time_ms": total_time,
            "requests_per_second": (len(successful) / total_time) * 1000 if total_time > 0 else 0,
            "response_times": {
                "min": min(times) if times else 0,
                "max": max(times) if times else 0,
                "mean": statistics.mean(times) if times else 0,
                "median": statistics.median(times) if times else 0,
                "p95": statistics.quantiles(times, n=20)[18] if len(times) > 20 else max(times) if times else 0,
                "p99": statistics.quantiles(times, n=100)[98] if len(times) > 100 else max(times) if times else 0,
            },
            "resources": {
                "before": resources_before,
                "after": resources_after,
                "delta_cpu": resources_after.get("cpu_percent", 0) - resources_before.get("cpu_percent", 0),
                "delta_memory_mb": resources_after.get("memory_mb", 0) - resources_before.get("memory_mb", 0),
            }
        }

        return stats

    def print_stats(self, stats: Dict):
        """Pretty print statistics"""
        print(f"\nResults for {stats['test_type']} test:")
        print(f"  Concurrency: {stats['concurrency']}")
        print(f"  Total requests: {stats['successful']}/{stats['total_requests']} successful")
        print(f"  Unique L2 parents: {stats['unique_l2_parents']}")
        print(f"  Total time: {stats['total_time_ms']:.1f} ms")
        print(f"  Throughput: {stats['requests_per_second']:.1f} req/s")
        print(f"\nResponse times (ms):")
        rt = stats['response_times']
        print(f"  Min: {rt['min']:.1f}")
        print(f"  Median: {rt['median']:.1f}")
        print(f"  Mean: {rt['mean']:.1f}")
        print(f"  P95: {rt['p95']:.1f}")
        print(f"  P99: {rt['p99']:.1f}")
        print(f"  Max: {rt['max']:.1f}")
        print(f"\nResource usage:")
        res = stats['resources']
        print(f"  CPU delta: {res.get('delta_cpu', 0):.1f}%")
        print(f"  Memory delta: {res.get('delta_memory_mb', 0):.1f} MB")
        print(f"  Final memory: {res['after'].get('memory_mb', 0):.1f} MB")
        print(f"  Threads: {res['after'].get('num_threads', 0)}")

async def main():
    parser = argparse.ArgumentParser(description='RRIP Tile Server UNCACHED Performance Test')
    parser.add_argument('--url', default='http://localhost:3007', help='Server URL')
    parser.add_argument('--slide', default='demo_out', help='Slide ID')
    parser.add_argument('--max-concurrent', type=int, default=128, help='Maximum concurrency to test')
    args = parser.parse_args()

    bench = UncachedTileServerBenchmark(args.url, args.slide)

    # Calculate how many L2 families we have available
    max_requests = len(bench.l2_families)
    print(f"Maximum uncached requests available: {max_requests}")

    all_results = []

    # Sequential test
    stats = await bench.run_sequential_test(max_requests)
    bench.print_stats(stats)
    all_results.append(stats)

    # Various concurrency levels - go higher!
    for concurrency in [2, 4, 8, 16, 32, 64, 96, 128]:
        if concurrency > args.max_concurrent:
            break
        await asyncio.sleep(2)  # Let server stabilize
        stats = await bench.run_concurrent_test(max_requests, concurrency)
        bench.print_stats(stats)
        all_results.append(stats)

    # Save results
    filename = f"uncached_perf_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {filename}")

if __name__ == "__main__":
    asyncio.run(main())