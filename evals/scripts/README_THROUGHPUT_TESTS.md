# WSI Throughput Optimization Tests

## Overview

These scripts test different concurrency approaches to maximize WSI encoding throughput on the B200 GPU.

## Test Scripts

### Individual Tests

1. **`run_5_concurrent_wsi.sh`** - Sequential (concurrency=1)
   - One slide at a time on GPU
   - CPU pyramids run in background thread
   - Expected: ~85s for 5 slides (if overlap works)

2. **`run_5_slides_2concurrent.sh`** - Concurrent pairs (concurrency=2)
   - Max 2 slides at once (based on ~50% GPU usage per slide)
   - Should fully saturate GPU
   - Expected: ~70s for 5 slides (if GPU parallelism works)

3. **`run_5_parallel_wsi.sh`** - Full parallel (concurrency=5)
   - All 5 slides run simultaneously
   - Stress test for GPU oversubscription
   - Expected: varies (could be slower if GPU queues)

### Comparison Test

**`compare_sequential_vs_parallel.sh`** - Runs all 3 tests and compares results

## Running Tests

```bash
# First rebuild with the pyramid wait fix
bash /workspace/RRIP/evals/scripts/rebuild_and_test_overlap.sh

# Or run the full comparison
bash /workspace/RRIP/evals/scripts/compare_sequential_vs_parallel.sh
```

## Expected Outcomes

### If Sequential (concurrency=1) wins:
- GPU is already saturated with one slide
- OR pyramid CPU work is the bottleneck (60s pyramid > 5s families)
- **Recommendation**: Current implementation is optimal

### If Concurrent-2 wins:
- GPU has ~50% unused capacity with 1 slide
- 2 concurrent streams achieve 100% GPU utilization
- **Recommendation**: Use concurrency=2 for production

### If Parallel-5 wins:
- GPU can handle >2 parallel streams
- B200 has enough VRAM/compute for 5+ concurrent slides
- **Recommendation**: Increase concurrency further

## Key Metrics

Per slide (200 families):
- **Family encoding**: ~5s on GPU
- **Pyramid generation**: ~60s on CPU (23 cores)
- **GPU utilization**: ~50% per slide (observed)
- **Total sequential time**: 5s + 60s = 65s per slide

Theoretical best case (perfect overlap + concurrency=2):
- 5 slides / 2 = 3 batches (2+2+1)
- Batch time = max(5s GPU, 60s CPU pyramid) = 60s
- Total = 3 × 60s = 180s
- BUT with overlapping pyramids: closer to 90-100s

## Notes

- All tests use `--max-parents 200` (200 families per slide)
- Pyramid generation uses box filter + sharpen (strength 0.25)
- Each slide: 67K×92K pixels → 9 pyramid levels
- B200 GPU: 183GB VRAM, plenty of capacity for concurrent streams
