#!/usr/bin/env python3
"""
Performance test for RRIP tile server
Tests with random uncached requests at various concurrency levels
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
from typing import List, Dict, Tuple
import argparse

class TileServerBenchmark:
    def __init__(self, base_url: str = "http://localhost:3007", slide_id: str = "demo_out"):
        self.base_url = base_url
        self.slide_id = slide_id
        self.server_pid = None

        # Get available tile coordinates from residual packs
        self.l0_tiles = []
        self.l1_tiles = []
        self._discover_available_tiles()

    def _discover_available_tiles(self):
        """Discover available L0/L1 tiles from pack files"""
        import os
        import re

        pack_dir = f"data/{self.slide_id}/residual_packs"
        if not os.path.exists(pack_dir):
            # Fallback to generating random coordinates
            print("Warning: No pack files found, using random coordinates")
            for i in range(200):
                x2, y2 = i % 20, i // 20
                # L1 tiles (4 per L2 parent)
                for dx in range(2):
                    for dy in range(2):
                        self.l1_tiles.append((15, x2*2+dx, y2*2+dy))
                # L0 tiles (16 per L2 parent)
                for dx in range(4):
                    for dy in range(4):
                        self.l0_tiles.append((16, x2*4+dx, y2*4+dy))
            return

        # Parse pack filenames to get L2 coordinates
        for pack_file in os.listdir(pack_dir):
            if pack_file.endswith('.pack'):
                match = re.match(r'(\d+)_(\d+)\.pack', pack_file)
                if match:
                    x2, y2 = int(match.group(1)), int(match.group(2))
                    # Add L1 tiles
                    for dx in range(2):
                        for dy in range(2):
                            self.l1_tiles.append((15, x2*2+dx, y2*2+dy))
                    # Add L0 tiles
                    for dx in range(4):
                        for dy in range(4):
                            self.l0_tiles.append((16, x2*4+dx, y2*4+dy))

        print(f"Discovered {len(self.l1_tiles)} L1 tiles and {len(self.l0_tiles)} L0 tiles")

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

    def get_random_tiles(self, count: int, prefer_l0: float = 0.7) -> List[Tuple[int, int, int]]:
        """Get random tile coordinates, preferring L0 tiles"""
        tiles = []
        for _ in range(count):
            if random.random() < prefer_l0 and self.l0_tiles:
                tiles.append(random.choice(self.l0_tiles))
            elif self.l1_tiles:
                tiles.append(random.choice(self.l1_tiles))
        return tiles

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
                    "tile": f"{level}/{x}_{y}"
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
        """Run concurrent requests test"""
        print(f"\n=== Concurrent Test: {num_requests} requests, {concurrency} concurrent ===")

        tiles = self.get_random_tiles(num_requests)
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

                # Brief pause between batches to avoid overwhelming
                if i + concurrency < len(tiles):
                    await asyncio.sleep(0.01)

        total_time = (time.perf_counter() - start_time) * 1000

        # Monitor resources after
        resources_after = self.monitor_resources()

        # Calculate statistics
        successful = [r for r in results if r["success"]]
        times = [r["time_ms"] for r in successful]

        stats = {
            "test_type": "concurrent",
            "total_requests": num_requests,
            "concurrency": concurrency,
            "successful": len(successful),
            "failed": len(results) - len(successful),
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
        """Run sequential requests test"""
        print(f"\n=== Sequential Test: {num_requests} requests ===")

        tiles = self.get_random_tiles(num_requests)
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

        # Calculate statistics
        successful = [r for r in results if r["success"]]
        times = [r["time_ms"] for r in successful]

        stats = {
            "test_type": "sequential",
            "total_requests": num_requests,
            "concurrency": 1,
            "successful": len(successful),
            "failed": len(results) - len(successful),
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
    parser = argparse.ArgumentParser(description='RRIP Tile Server Performance Test')
    parser.add_argument('--url', default='http://localhost:3007', help='Server URL')
    parser.add_argument('--slide', default='demo_out', help='Slide ID')
    parser.add_argument('--requests', type=int, default=100, help='Number of requests per test')
    args = parser.parse_args()

    bench = TileServerBenchmark(args.url, args.slide)

    # Warm up
    print("Warming up server...")
    await bench.run_sequential_test(10)

    all_results = []

    # Sequential test
    stats = await bench.run_sequential_test(args.requests)
    bench.print_stats(stats)
    all_results.append(stats)

    # Various concurrency levels
    for concurrency in [2, 4, 8, 16, 32]:
        await asyncio.sleep(2)  # Let server stabilize
        stats = await bench.run_concurrent_test(args.requests, concurrency)
        bench.print_stats(stats)
        all_results.append(stats)

    # Save results
    filename = f"perf_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {filename}")

if __name__ == "__main__":
    asyncio.run(main())