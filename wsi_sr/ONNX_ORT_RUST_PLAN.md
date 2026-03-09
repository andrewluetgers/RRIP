# ONNX Runtime SR Inference in Rust

## Goal

Replace the luma residual pipeline with a learned 4x SR model running directly in the Rust tile server via ONNX Runtime. The collapsed WSISRX4 model (19,008 params, 77 KB float32 / 32 KB INT8) fits in L1 cache and is ideal for SIMD-accelerated inference.

## Current Pipeline (residual-based)

```
L2 tile (256×256 JPEG) → decode
  → upsample Y 4x (lanczos3) → L0 Y prediction
  → decode fused L0 residual JPEG (1024×1024 grayscale)
  → L0_Y = clamp(pred_Y + residual - 128)
  → upsample Cb/Cr 4x
  → combine Y + CbCr → encode 16 L0 JPEGs
```

**Cost per family**: ~48ms (lanczos3), dominated by upsample + residual decode + 16 JPEG encodes.
**Storage**: L2 JPEG + fused L0 residual JPEG per family.

## Proposed Pipeline (SR model)

```
L2 tile (256×256 JPEG) → decode → RGB float [0,1]
  → SR model (ONNX, 256×256×3 → 1024×1024×3)
  → convert to YCbCr
  → slice into 16 L0 tiles + 4 L1 tiles → JPEG encode
```

**Eliminates**: residual JPEG storage + residual decode + manual upsample.
**Adds**: ~5-10ms ONNX inference (AVX2/NEON, INT8).
**Net effect**: Faster reconstruction, zero residual storage, quality depends on model.

## Architecture

### Model Summary

```
WSISRX4 (collapsed):
  head: Conv2d(3→16, 3×3) + ReLU          # 448 params
  body: 5× Conv2d(16→16, 3×3) + ReLU      # 5 × 2,320 = 11,600 params
  tail: Conv2d(16→48, 3×3)                 # 6,960 params
  PixelShuffle(4)                          # 48 channels → 3×(4×4)
  + bilinear skip connection               # input upsampled 4x, added to output
                                           # Total: 19,008 params, 74.2 KB
```

Input: `[1, 3, 256, 256]` float32 → Output: `[1, 3, 1024, 1024]` float32 `[0, 1]`

### Why This Is Fast

- **7 convolutions total**, all 3×3 with 16 channels
- First 6 operate at **256×256** (65K pixels) — tiny
- Only the pixel shuffle expands to 1024×1024
- The bilinear skip is a simple interpolation
- 19K params × 4 bytes = 74 KB — **fits in L1 cache** (typical L1 = 32-64 KB per core, L2 = 256 KB+)
- INT8 variant is 32 KB — comfortably in L1

### SIMD Acceleration

ONNX Runtime's MLAS kernel library automatically uses:

| Platform | ISA | Key ops | Expected speedup |
|----------|-----|---------|-------------------|
| x86-64 (server) | AVX2 | 8-wide float32 FMA | 4-8x vs scalar |
| x86-64 (server) | AVX-512 | 16-wide float32 FMA | 8-16x vs scalar |
| x86-64 (server) | AVX-512 VNNI | INT8 dot product | 16-32x vs scalar float32 |
| Apple Silicon | NEON | 4-wide float32 FMA | 4x vs scalar |
| Apple Silicon | NEON DOT | INT8 dot product | 8-16x vs scalar float32 |

For 3×3 conv with 16 channels: each output pixel needs 16×9 = 144 MACs.
At 256×256, that's 144 × 65,536 = ~9.4M MACs per layer × 7 layers = **~66M MACs total**.
A modern CPU at 100+ GFLOPS (AVX2) can do this in **< 1ms**.

The bottleneck will be the pixel shuffle + bilinear upsample (memory bandwidth at 1024×1024), not compute.

## Implementation Plan

### Step 1: Add `ort` crate dependency

```toml
# server/Cargo.toml
[dependencies]
ort = { version = "2", features = ["load-dynamic"] }  # or "download-binaries"

[features]
sr-model = []  # feature-gate the SR model path
```

Using `load-dynamic` lets us link to a system-installed ONNX Runtime, avoiding large binary bloat. Alternatively, `download-binaries` auto-fetches the right version at build time.

### Step 2: SR model loader module

New file: `server/src/core/sr_model.rs`

```rust
use ort::{Session, SessionBuilder, GraphOptimizationLevel, Value};
use ndarray::{Array4, ArrayView4};

pub struct SRModel {
    session: Session,
}

impl SRModel {
    /// Load ONNX model from path.
    /// Call once at startup, reuse across requests.
    pub fn load(onnx_path: &str, num_threads: usize) -> Result<Self> {
        let session = SessionBuilder::new()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(num_threads)?
            .commit_from_file(onnx_path)?;
        Ok(Self { session })
    }

    /// Run 4x SR inference.
    /// Input: [1, 3, 256, 256] float32 RGB in [0, 1]
    /// Output: [1, 3, 1024, 1024] float32 RGB in [0, 1]
    pub fn infer(&self, input: ArrayView4<f32>) -> Result<Array4<f32>> {
        let input_value = Value::from_array(input)?;
        let outputs = self.session.run(vec![input_value])?;
        let output = outputs[0].extract_tensor::<f32>()?;
        Ok(output.view().to_owned())
    }
}
```

### Step 3: Integrate into reconstruction pipeline

Modify `server/src/core/reconstruct.rs`:

```rust
// New reconstruction path when SR model is available
pub fn reconstruct_family_sr(
    l2_jpeg: &[u8],
    sr_model: &SRModel,
    encoder: &dyn JpegEncoder,
    quality: u8,
) -> Result<FamilyTiles> {
    // 1. Decode L2 JPEG → RGB u8 [256×256]
    let l2_rgb = decode_jpeg(l2_jpeg)?;

    // 2. Convert to float32 [0,1] tensor [1,3,256,256]
    let input = rgb_to_tensor(&l2_rgb);  // u8 → f32, HWC → BCHW

    // 3. Run SR model → [1,3,1024,1024]
    let output = sr_model.infer(input.view())?;

    // 4. Convert to u8 RGB [1024×1024]
    let l0_rgb = tensor_to_rgb(&output);  // f32 → u8, BCHW → HWC

    // 5. Convert to YCbCr
    let (y, cb, cr) = rgb_to_ycbcr_planes(&l0_rgb);

    // 6. Slice into 16 L0 tiles (256×256 each)
    let l0_tiles = slice_1024_to_256(&y, &cb, &cr);

    // 7. Downsample for 4 L1 tiles (512×512 → 4×256×256)
    let l1_y = downsample_2x(&y);
    let l1_cb = downsample_2x(&cb);
    let l1_cr = downsample_2x(&cr);
    let l1_tiles = slice_512_to_256(&l1_y, &l1_cb, &l1_cr);

    // 8. JPEG encode all 20 tiles (parallel via rayon)
    encode_family_tiles(l0_tiles, l1_tiles, encoder, quality)
}
```

### Step 4: CLI integration

Add to `server/src/serve.rs`:

```
--sr-model <path>     Path to SR ONNX model (enables learned reconstruction)
--sr-threads <n>      ONNX Runtime intra-op threads (default: 4)
--sr-int8             Use INT8 quantized model for faster inference
```

When `--sr-model` is provided, use `reconstruct_family_sr()` instead of the residual-based path. Falls back to residual path if the SR model file doesn't exist or for slides that have residual packs.

### Step 5: Benchmarking

Benchmark against the current pipeline:

```bash
# Current: residual-based (lanczos3)
origami serve --slides-root data --upsample-filter lanczos3

# New: SR model
origami serve --slides-root data --sr-model wsi_sr/model_sr.onnx

# New: SR model INT8
origami serve --slides-root data --sr-model wsi_sr/model_sr_int8.onnx --sr-int8
```

Measure:
- Cold miss latency (family reconstruction)
- Per-tile PSNR/SSIM/Delta E vs ground truth
- Throughput under load (vegeta)

## Expected Performance

### Inference latency (per family, single L2 → 20 tiles)

| Step | Current (residual) | SR model (float32) | SR model (INT8) |
|------|-------------------|--------------------|-----------------|
| Decode L2 JPEG | 1ms | 1ms | 1ms |
| Upsample Y 4x (lanczos3) | 13ms | — | — |
| Upsample CbCr 4x | 32ms | — | — |
| Decode residual JPEG | 2ms | — | — |
| **SR model inference** | — | **3-8ms** | **1-4ms** |
| RGB→YCbCr conversion | — | 2ms | 2ms |
| Slice + JPEG encode 20 tiles | 15ms | 15ms | 15ms |
| **Total** | **~48ms** | **~21-26ms** | **~19-22ms** |

The SR model eliminates the expensive upsample steps (45ms) and replaces them with a single ~5ms inference pass. **Net speedup: ~2x.**

### Storage savings

| Component | Current | SR model |
|-----------|---------|----------|
| L2 baseline JPEG | ~15-25 KB | ~15-25 KB |
| Fused L0 residual | ~8-30 KB | **0 KB** |
| SR model (shared) | 0 KB | 77 KB (amortized over all families) |
| **Per-family cost** | ~23-55 KB | **~15-25 KB** |

The model is loaded once and shared across all slides — its 77 KB cost is amortized to effectively zero per-family.

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Model quality worse than residual | Visual artifacts | Run evaluate.py on diverse tiles before deploying; keep residual fallback |
| Single-WSI model doesn't generalize | Poor quality on new scanners/stains | Train on TCGA multi-WSI dataset (Phase 2) |
| ONNX Runtime binary size | Large server binary | Use `load-dynamic` feature, ship libonnxruntime separately |
| INT8 quantization artifacts | Quality loss | Benchmark INT8 vs float32 quality; use float32 if gap > 0.5 dB |
| Apple Silicon Neural Engine | Not accessible via ORT CPU | Use CoreML execution provider or stick with NEON (still fast) |

## Future: Hybrid Pipeline

Once the SR model is proven, a hybrid approach could combine both:

```
L2 JPEG → SR model → SR prediction (1024×1024)
         → compute residual vs ground truth
         → if residual is small enough: skip storing it
         → if residual is large: store compressed residual for correction
```

This gives the best of both worlds: the SR model handles most of the reconstruction, and a lightweight residual corrects the remaining error for tiles where the model struggles.

## Model Variants

### WSISRX4 (baseline, RGB)
- 19,008 params collapsed, 74.2 KB float32
- Input: RGB 256×256 → Output: RGB 1024×1024
- Global residual: bilinear upsample + model correction
- Stage 1 results: PSNR 30.38, SSIM 0.858, Delta E 2.55

### WSISRX4Dual (Y-heavy + CbCr-light)
- 17,736 params collapsed, 69.3 KB float32
- Internal: RGB → YCbCr, Y branch (5 blocks, 16ch) + CbCr branch (2 blocks, 8ch) → YCbCr → RGB
- Allocates capacity to luma (residual quality) while learning chroma (Delta E)
- Potentially better Delta E at same or smaller model size

### ESPCNR (comparison baseline)
- 37,200 params, 145.3 KB float32
- Classic ESPCN + global residual
- 2x larger than our models — if it doesn't beat WSISRX4, our architecture wins

### Rust Integration Notes

For the dual-branch model, the ONNX graph handles the YCbCr conversion internally — the Rust code just feeds RGB and gets RGB back. No special handling needed. The ONNX export includes the `rgb_to_ycbcr` and `ycbcr_to_rgb` ops as part of the graph.

For INT8 quantization, the dual-branch Y path benefits more from quantization (luma is less sensitive to quantization noise than chroma). Consider quantizing Y branch to INT8 and keeping CbCr branch at float32 — mixed precision.

## Decode Integration Strategy

### Phase 1: Side-by-side comparison
- Add `--sr-model` flag to `origami serve` and `origami decode`
- When both SR model and residual packs exist, run both and compare quality
- Log per-tile PSNR difference: SR-only vs residual-corrected

### Phase 2: SR-only mode
- For slides where SR quality is sufficient (PSNR > threshold), skip residuals entirely
- Zero storage cost per family — only the L2 baseline JPEG
- Massive storage reduction for the entire pyramid

### Phase 3: Hybrid mode
- SR model provides the base prediction
- Lightweight residual corrects only where SR is wrong
- Residuals become much smaller (SR catches most of the detail)
- Best quality with minimal storage

## Dependencies

- `ort` crate v2.x — Rust bindings for ONNX Runtime
- ONNX Runtime 1.17+ — system library or auto-downloaded
- Trained ONNX model — `wsi_sr/model_sr.onnx` (77 KB) or `model_sr_int8.onnx` (32 KB)
- For dual-branch: same ONNX format, YCbCr ops baked into graph
