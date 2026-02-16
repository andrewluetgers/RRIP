/**
 * Unsharp Mask Sharpening Kernel — Interleaved RGB u8
 *
 * Matches the CPU implementation in server/src/core/sharpen.rs exactly:
 * 1. Separable 3×3 Gaussian blur (kernel [0.25, 0.5, 0.25])
 * 2. Sharpen: out = src + strength × (src − blurred), clamped [0, 255]
 *
 * Two-pass implementation using integer/fixed-point math (no intermediate rounding):
 * - Pass 1 (horizontal): src → temp (u16), stores left + 2*center + right (range 0..1020)
 * - Pass 2 (vertical + sharpen): temp (u16) + src → dst (u8), fixed-point 8.8 strength
 *
 * Input:  width × height × 3 interleaved RGB u8
 * Output: width × height × 3 interleaved RGB u8
 */

// Pass 1: Horizontal blur → u16 intermediate (no rounding)
// Stores: left + 2*center + right (range 0..1020)
extern "C" __global__ void unsharp_hblur_kernel(
    const unsigned char* __restrict__ src,
    unsigned short* __restrict__ temp,
    int width, int height
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height * 3;
    if (idx >= total) return;

    int c = idx % 3;
    int tmp = idx / 3;
    int x = tmp % width;
    int y = tmp / width;

    int stride = width * 3;
    int row = y * stride;

    int x0 = (x > 0) ? x - 1 : 0;
    int x2 = (x + 1 < width) ? x + 1 : width - 1;

    unsigned short a = (unsigned short)src[row + x0 * 3 + c];
    unsigned short b = (unsigned short)src[row + x  * 3 + c];
    unsigned short d = (unsigned short)src[row + x2 * 3 + c];

    temp[idx] = a + 2 * b + d;
}

// Pass 2: Vertical blur on u16 hblur + fused sharpen
// Uses same fixed-point math as CPU: strength_i = round(strength * 256)
// blur16 = hblur[y-1] + 2*hblur[y] + hblur[y+1]
// s16 = src << 4
// diff = s16 - blur16
// v = s16 * 256 + strength_i * diff
// out = clamp((v + 2048) >> 12, 0, 255)
extern "C" __global__ void unsharp_vblur_sharpen_kernel(
    const unsigned char* __restrict__ src,
    const unsigned short* __restrict__ temp,
    unsigned char* __restrict__ dst,
    int width, int height, int strength_i
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height * 3;
    if (idx >= total) return;

    int c = idx % 3;
    int tmp = idx / 3;
    int x = tmp % width;
    int y = tmp / width;

    int stride = width * 3;
    int col_idx = x * 3 + c;

    int y0 = (y > 0) ? y - 1 : 0;
    int y2 = (y + 1 < height) ? y + 1 : height - 1;

    // Vertical blur of the horizontal-blurred temp (u16)
    int blur16 = (int)temp[y0 * stride + col_idx]
               + 2 * (int)temp[y  * stride + col_idx]
               + (int)temp[y2 * stride + col_idx];

    // Fixed-point sharpen (matches CPU exactly)
    int s16 = ((int)src[idx]) << 4;          // src × 16
    int diff = s16 - blur16;                  // (src - blurred) × 16
    int v = s16 * 256 + strength_i * diff;    // fixed-point: ×(16×256)
    v = (v + 2048) >> 12;                     // round and shift by 12 (16×256 = 4096)

    // Clamp [0, 255]
    v = max(0, min(255, v));
    dst[idx] = (unsigned char)v;
}
