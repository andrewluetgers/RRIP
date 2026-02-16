/**
 * Bilinear Upsample / Lanczos Downsample Kernels
 *
 * upsample_bilinear_2x: 2x upsample using bilinear interpolation
 *   Used for L2→L1 and L1→L0 prediction.
 *
 * downsample_lanczos3: Lanczos3 downsample (separable, 2-pass)
 *   Used for L0→L1→L2 ground truth generation from original tiles.
 */

/**
 * Bilinear 2x upsample.
 *
 * Input:  [N, H, W, C] float32
 * Output: [N, 2H, 2W, C] float32
 *
 * One thread per output pixel.
 * Maps output (ox, oy) to source (ox/2.0, oy/2.0) and bilinear interpolates.
 */
extern "C" __global__ void upsample_bilinear_2x_kernel(
    const float* __restrict__ src,   // [N * H * W * C]
    float* __restrict__ dst,         // [N * 2H * 2W * C]
    int N, int H, int W, int C
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int H2 = H * 2;
    int W2 = W * 2;
    int total = N * H2 * W2 * C;
    if (idx >= total) return;

    // Decode output index
    int c = idx % C;
    int tmp = idx / C;
    int ox = tmp % W2;
    tmp /= W2;
    int oy = tmp % H2;
    int n = tmp / H2;

    // Source coordinate (continuous)
    // Using the same mapping as image::FilterType::Triangle:
    // source = (output + 0.5) / 2.0 - 0.5
    float sx = ((float)ox + 0.5f) / 2.0f - 0.5f;
    float sy = ((float)oy + 0.5f) / 2.0f - 0.5f;

    // Clamp to valid source range
    sx = fmaxf(0.0f, fminf((float)(W - 1), sx));
    sy = fmaxf(0.0f, fminf((float)(H - 1), sy));

    int x0 = (int)floorf(sx);
    int y0 = (int)floorf(sy);
    int x1 = min(x0 + 1, W - 1);
    int y1 = min(y0 + 1, H - 1);

    float fx = sx - (float)x0;
    float fy = sy - (float)y0;

    // Bilinear interpolation
    float v00 = src[((n * H + y0) * W + x0) * C + c];
    float v01 = src[((n * H + y0) * W + x1) * C + c];
    float v10 = src[((n * H + y1) * W + x0) * C + c];
    float v11 = src[((n * H + y1) * W + x1) * C + c];

    float val = v00 * (1-fx) * (1-fy) + v01 * fx * (1-fy)
              + v10 * (1-fx) * fy     + v11 * fx * fy;

    dst[idx] = val;
}

/**
 * Lanczos3 horizontal pass.
 *
 * Input:  [N, H, W_src, C] float32
 * Output: [N, H, W_dst, C] float32
 *
 * One thread per output pixel.
 */

__device__ float lanczos3(float x) {
    if (fabsf(x) < 1e-6f) return 1.0f;
    if (fabsf(x) >= 3.0f) return 0.0f;
    float pi_x = 3.14159265358979323846f * x;
    return (sinf(pi_x) / pi_x) * (sinf(pi_x / 3.0f) / (pi_x / 3.0f));
}

extern "C" __global__ void downsample_lanczos3_h_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int N, int H, int W_src, int W_dst, int C,
    float scale  // W_dst / W_src
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * H * W_dst * C;
    if (idx >= total) return;

    int c = idx % C;
    int tmp = idx / C;
    int ox = tmp % W_dst;
    tmp /= W_dst;
    int oy = tmp % H;
    int n = tmp / H;

    // Center of output pixel in source coordinates
    float center = ((float)ox + 0.5f) / scale - 0.5f;
    float support = 3.0f / scale;

    int start = max(0, (int)ceilf(center - support));
    int end = min(W_src - 1, (int)floorf(center + support));

    float sum = 0.0f;
    float weight_sum = 0.0f;

    for (int sx = start; sx <= end; sx++) {
        float w = lanczos3((float)sx - center);
        sum += w * src[((n * H + oy) * W_src + sx) * C + c];
        weight_sum += w;
    }

    dst[idx] = (weight_sum > 0.0f) ? sum / weight_sum : 0.0f;
}

/**
 * Lanczos3 vertical pass.
 *
 * Input:  [N, H_src, W, C] float32
 * Output: [N, H_dst, W, C] float32
 */
extern "C" __global__ void downsample_lanczos3_v_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int N, int H_src, int H_dst, int W, int C,
    float scale  // H_dst / H_src
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * H_dst * W * C;
    if (idx >= total) return;

    int c = idx % C;
    int tmp = idx / C;
    int ox = tmp % W;
    tmp /= W;
    int oy = tmp % H_dst;
    int n = tmp / H_dst;

    float center = ((float)oy + 0.5f) / scale - 0.5f;
    float support = 3.0f / scale;

    int start = max(0, (int)ceilf(center - support));
    int end = min(H_src - 1, (int)floorf(center + support));

    float sum = 0.0f;
    float weight_sum = 0.0f;

    for (int sy = start; sy <= end; sy++) {
        float w = lanczos3((float)sy - center);
        sum += w * src[((n * H_src + sy) * W + ox) * C + c];
        weight_sum += w;
    }

    dst[idx] = (weight_sum > 0.0f) ? sum / weight_sum : 0.0f;
}
