/**
 * Unsharp Mask Sharpening Kernel — Interleaved RGB u8
 *
 * Matches the CPU implementation in server/src/core/sharpen.rs:
 * 1. Separable 3×3 Gaussian blur (kernel [0.25, 0.5, 0.25])
 * 2. Sharpen: out = src + strength × (src − blurred), clamped [0, 255]
 *
 * Two-pass implementation:
 * - Pass 1 (horizontal): src → temp
 * - Pass 2 (vertical + sharpen): temp → dst, using src for the final sharpening
 *
 * Input:  width × height × 3 interleaved RGB u8
 * Output: width × height × 3 interleaved RGB u8
 */

// Pass 1: Horizontal Gaussian blur [0.25, 0.5, 0.25] on interleaved RGB
extern "C" __global__ void unsharp_hblur_kernel(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ temp,
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

    float v = 0.25f * (float)src[row + x0 * 3 + c]
            + 0.50f * (float)src[row + x  * 3 + c]
            + 0.25f * (float)src[row + x2 * 3 + c];

    temp[idx] = (unsigned char)__float2int_rn(v);
}

// Pass 2: Vertical Gaussian blur [0.25, 0.5, 0.25] + sharpen
// Reads from temp (hblur output), writes sharpened result to dst
extern "C" __global__ void unsharp_vblur_sharpen_kernel(
    const unsigned char* __restrict__ src,
    const unsigned char* __restrict__ temp,
    unsigned char* __restrict__ dst,
    int width, int height, float strength
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

    // Vertical blur of the horizontal-blurred temp
    float blurred = 0.25f * (float)temp[y0 * stride + col_idx]
                  + 0.50f * (float)temp[y  * stride + col_idx]
                  + 0.25f * (float)temp[y2 * stride + col_idx];

    // Sharpen: out = src + strength * (src - blurred)
    float s = (float)src[idx];
    float v = s + strength * (s - blurred);
    v = fmaxf(0.0f, fminf(255.0f, v));

    dst[idx] = (unsigned char)__float2int_rn(v);
}
