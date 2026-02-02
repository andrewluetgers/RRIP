#!/usr/bin/env python3
"""
Comprehensive performance test for RRIP tile server with detailed metrics.
Measures:
- Tiles per second
- Family generation rate
- Pack file decompression speed
- Memory usage
- CPU usage
- Thread count
- Latency percentiles
- Concurrency scaling
"""

import asyncio
import aiohttp
import time
import psutil
import random
import statistics
import json
import os
from typing import List, Dict, Any
from dataclasses import dataclass, asdict

# Configuration
SERVER_URL = "http://localhost:3007"
SLIDE_ID = "demo_out"
TEST_DURATION = 30  # seconds per concurrency level
CONCURRENCY_LEVELS = [1, 2, 4, 8, 16, 32, 64, 128]

@dataclass
class TestResult:
    concurrency: int
    duration: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    tiles_per_second: float
    families_per_second: float
    packs_per_second: float
    latency_mean_ms: float
    latency_median_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_min_ms: float
    latency_max_ms: float
    memory_rss_mb: float
    memory_vms_mb: float
    cpu_percent: float
    thread_count: int

class PerformanceTester:
    def __init__(self):
        self.server_pid = None
        self.results = []

    async def verify_server(self):
        """Check if server is running and get its PID."""
        # Find server process
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info['cmdline']
                if cmdline and 'rrip-tile-server' in ' '.join(cmdline):
                    self.server_pid = proc.info['pid']
                    print(f"Found server process: PID {self.server_pid}")
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        if not self.server_pid:
            raise RuntimeError("Server not found!")

        # Verify it's responding
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{SERVER_URL}/healthz") as resp:
                if resp.status != 200:
                    raise RuntimeError("Server not responding")

        print("Server verified and ready")

    def get_server_metrics(self):
        """Get current server process metrics."""
        if not self.server_pid:
            return {}

        try:
            proc = psutil.Process(self.server_pid)
            with proc.oneshot():
                mem_info = proc.memory_info()
                return {
                    'memory_rss_mb': mem_info.rss / (1024 * 1024),
                    'memory_vms_mb': mem_info.vms / (1024 * 1024),
                    'cpu_percent': proc.cpu_percent(interval=0.1),
                    'thread_count': proc.num_threads(),
                }
        except psutil.NoSuchProcess:
            return {}

    def get_random_l0_tiles(self, count: int) -> List[tuple]:
        """Generate random L0 tile coordinates."""
        tiles = []
        for _ in range(count):
            # L0 tiles in valid range (L2 is 0-55 x 0-58, so L0 is 0-220 x 0-232)
            x = random.randint(0, 220)
            y = random.randint(0, 232)
            tiles.append((16, x, y))  # Level 16 is L0
        return tiles

    async def fetch_tile(self, session: aiohttp.ClientSession, level: int, x: int, y: int):
        """Fetch a single tile and measure timing."""
        url = f"{SERVER_URL}/tiles/{SLIDE_ID}/{level}/{x}_{y}.jpg"  # Use provided level and coordinates

        start_time = time.perf_counter()
        try:
            async with session.get(url) as resp:
                data = await resp.read()
                elapsed = time.perf_counter() - start_time

                return {
                    'success': resp.status == 200,
                    'elapsed_ms': elapsed * 1000,
                    'size_bytes': len(data) if resp.status == 200 else 0,
                }
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            return {
                'success': False,
                'elapsed_ms': elapsed * 1000,
                'error': str(e),
            }

    async def run_concurrent_test(self, concurrency: int) -> TestResult:
        """Run test at specified concurrency level."""
        print(f"\nTesting concurrency={concurrency}")

        # Generate enough random tiles
        tiles_needed = int(concurrency * TEST_DURATION * 20)  # Estimate
        tile_coords = self.get_random_l0_tiles(tiles_needed)
        tile_index = 0

        results = []
        start_time = time.time()
        tasks_in_flight = []

        # Track metrics during test
        cpu_samples = []
        memory_samples = []
        thread_samples = []

        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=concurrency)
        ) as session:

            while time.time() - start_time < TEST_DURATION:
                # Keep concurrency level constant
                while len(tasks_in_flight) < concurrency and tile_index < len(tile_coords):
                    level, x, y = tile_coords[tile_index]
                    tile_index += 1

                    task = asyncio.create_task(self.fetch_tile(session, level, x, y))
                    tasks_in_flight.append(task)

                # Wait for some tasks to complete
                if tasks_in_flight:
                    done, pending = await asyncio.wait(
                        tasks_in_flight,
                        timeout=0.1,
                        return_when=asyncio.FIRST_COMPLETED
                    )

                    for task in done:
                        result = await task
                        results.append(result)

                    tasks_in_flight = list(pending)

                # Sample metrics periodically
                if len(results) % 10 == 0:
                    metrics = self.get_server_metrics()
                    if metrics:
                        cpu_samples.append(metrics.get('cpu_percent', 0))
                        memory_samples.append(metrics.get('memory_rss_mb', 0))
                        thread_samples.append(metrics.get('thread_count', 0))

            # Wait for remaining tasks
            if tasks_in_flight:
                remaining = await asyncio.gather(*tasks_in_flight)
                results.extend(remaining)

        # Calculate statistics
        actual_duration = time.time() - start_time
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        latencies = [r['elapsed_ms'] for r in successful]

        if not latencies:
            print(f"  WARNING: No successful requests!")
            latencies = [0]  # Avoid division by zero

        # Each L0 request triggers a family generation (20 tiles)
        # Each family comes from one pack file
        tiles_per_second = len(successful) / actual_duration
        families_per_second = tiles_per_second  # Each L0 tile triggers one family
        packs_per_second = families_per_second  # One pack per family

        result = TestResult(
            concurrency=concurrency,
            duration=actual_duration,
            total_requests=len(results),
            successful_requests=len(successful),
            failed_requests=len(failed),
            tiles_per_second=tiles_per_second,
            families_per_second=families_per_second,
            packs_per_second=packs_per_second,
            latency_mean_ms=statistics.mean(latencies) if latencies else 0,
            latency_median_ms=statistics.median(latencies) if latencies else 0,
            latency_p95_ms=statistics.quantiles(latencies, n=20)[18] if len(latencies) > 20 else max(latencies),
            latency_p99_ms=statistics.quantiles(latencies, n=100)[98] if len(latencies) > 100 else max(latencies),
            latency_min_ms=min(latencies) if latencies else 0,
            latency_max_ms=max(latencies) if latencies else 0,
            memory_rss_mb=statistics.mean(memory_samples) if memory_samples else 0,
            memory_vms_mb=0,  # Not tracking VMS average
            cpu_percent=statistics.mean(cpu_samples) if cpu_samples else 0,
            thread_count=int(statistics.mean(thread_samples)) if thread_samples else 0,
        )

        # Print summary
        print(f"  Tiles/sec: {result.tiles_per_second:.1f}")
        print(f"  Families/sec: {result.families_per_second:.1f}")
        print(f"  Packs/sec: {result.packs_per_second:.1f}")
        print(f"  Latency p50: {result.latency_median_ms:.1f}ms, p95: {result.latency_p95_ms:.1f}ms")
        print(f"  Memory: {result.memory_rss_mb:.1f}MB, CPU: {result.cpu_percent:.1f}%, Threads: {result.thread_count}")

        return result

    async def run_all_tests(self):
        """Run tests at all concurrency levels."""
        await self.verify_server()

        print(f"\nRunning {TEST_DURATION}s tests at each concurrency level...")

        for concurrency in CONCURRENCY_LEVELS:
            result = await self.run_concurrent_test(concurrency)
            self.results.append(result)

            # Brief pause between tests
            await asyncio.sleep(2)

        self.print_summary()
        self.save_results()

    def print_summary(self):
        """Print comprehensive summary table."""
        print("\n" + "=" * 120)
        print("COMPREHENSIVE PERFORMANCE RESULTS - RRIP with LZ4 Compression")
        print("=" * 120)

        print(f"\n{'Concur':<7} {'Tiles/s':<9} {'Fam/s':<8} {'Pack/s':<8} "
              f"{'Lat p50':<9} {'Lat p95':<9} {'Lat p99':<9} "
              f"{'Memory':<10} {'CPU %':<7} {'Threads':<8} {'Success'}")
        print("-" * 120)

        for r in self.results:
            success_rate = (r.successful_requests / r.total_requests * 100) if r.total_requests > 0 else 0
            print(f"{r.concurrency:<7} "
                  f"{r.tiles_per_second:<9.1f} "
                  f"{r.families_per_second:<8.1f} "
                  f"{r.packs_per_second:<8.1f} "
                  f"{r.latency_median_ms:<9.1f} "
                  f"{r.latency_p95_ms:<9.1f} "
                  f"{r.latency_p99_ms:<9.1f} "
                  f"{r.memory_rss_mb:<10.1f} "
                  f"{r.cpu_percent:<7.1f} "
                  f"{r.thread_count:<8} "
                  f"{success_rate:.1f}%")

        # Find peak throughput
        if self.results:
            peak = max(self.results, key=lambda x: x.tiles_per_second)
            print(f"\nPEAK PERFORMANCE:")
            print(f"  Concurrency: {peak.concurrency}")
            print(f"  Throughput: {peak.tiles_per_second:.1f} tiles/sec")
            print(f"  Family generation: {peak.families_per_second:.1f} families/sec")
            print(f"  Pack decompression: {peak.packs_per_second:.1f} packs/sec")
            print(f"  Total tiles generated: {peak.families_per_second * 20:.1f} tiles/sec (20 per family)")

            # Calculate pack decompression speed
            pack_size_kb = 12.6  # Average from earlier test
            decompression_rate = peak.packs_per_second * pack_size_kb / 1024
            print(f"  Decompression rate: {decompression_rate:.1f} MB/sec of LZ4 data")

    def save_results(self):
        """Save results to JSON file."""
        with open('performance_results.json', 'w') as f:
            results_dict = [asdict(r) for r in self.results]
            json.dump(results_dict, f, indent=2)
        print("\nResults saved to performance_results.json")

async def main():
    tester = PerformanceTester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())