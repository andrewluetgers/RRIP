/**
 * 2x Downsample Kernel — Box Filter
 *
 * Downsamples an RGB image by 2× using a simple 2x2 box filter (average).
 * Each output pixel is the average of a 2×2 block in the source image.
 *
 * Input:  [H, W, 3] u8 RGB interleaved
 * Output: [H/2, W/2, 3] u8 RGB interleaved
 *
 * Thread per output pixel.
 */

extern "C" __global__ void downsample_2x_box_kernel(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ dst,
    int src_w, int src_h
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_w = src_w / 2;
    int dst_h = src_h / 2;
    int total = dst_w * dst_h * 3;
    if (idx >= total) return;

    int c = idx % 3;
    int tmp = idx / 3;
    int x = tmp % dst_w;
    int y = tmp / dst_w;

    // Source coordinates (top-left of 2x2 block)
    int sx = x * 2;
    int sy = y * 2;

    // Clamp to avoid out-of-bounds for odd dimensions
    int sx1 = (sx + 1 < src_w) ? sx + 1 : sx;
    int sy1 = (sy + 1 < src_h) ? sy + 1 : sy;

    int stride = src_w * 3;

    // Read 2x2 block
    int v00 = (int)src[(sy  * stride) + (sx  * 3) + c];
    int v10 = (int)src[(sy  * stride) + (sx1 * 3) + c];
    int v01 = (int)src[(sy1 * stride) + (sx  * 3) + c];
    int v11 = (int)src[(sy1 * stride) + (sx1 * 3) + c];

    // Box filter: average of 4 pixels
    int avg = (v00 + v10 + v01 + v11 + 2) / 4;  // +2 for rounding
    dst[idx] = (unsigned char)avg;
}
