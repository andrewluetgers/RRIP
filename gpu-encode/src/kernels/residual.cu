/**
 * YCbCr Conversion + Residual Computation Kernels
 *
 * Fused kernels for the residual computation pipeline:
 *
 * 1. rgb_to_ycbcr_f32: Convert RGB u8 to float32 YCbCr (BT.601)
 * 2. compute_residual: residual = round(gt_y - pred_y_f32 + 128)
 * 3. reconstruct_y: recon_y = clamp(pred_y + (res - 128), 0, 255)
 * 4. ycbcr_to_rgb: Convert YCbCr to RGB u8
 *
 * All element-wise, trivially parallel.
 */

/**
 * RGB u8 → float32 YCbCr (BT.601)
 *
 * Y  =  0.299*R + 0.587*G + 0.114*B
 * Cb = -0.169*R - 0.331*G + 0.500*B + 128
 * Cr =  0.500*R - 0.419*G - 0.081*B + 128
 */
extern "C" __global__ void rgb_to_ycbcr_f32_kernel(
    const unsigned char* __restrict__ rgb,  // [N * H * W * 3]
    float* __restrict__ y_out,              // [N * H * W]
    float* __restrict__ cb_out,             // [N * H * W]
    float* __restrict__ cr_out,             // [N * H * W]
    int total_pixels                        // N * H * W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pixels) return;

    float r = (float)rgb[idx * 3];
    float g = (float)rgb[idx * 3 + 1];
    float b = (float)rgb[idx * 3 + 2];

    y_out[idx]  =  0.299f * r + 0.587f * g + 0.114f * b;
    cb_out[idx] = -0.168736f * r - 0.331264f * g + 0.500f * b + 128.0f;
    cr_out[idx] =  0.500f * r - 0.418688f * g - 0.081312f * b + 128.0f;
}

/**
 * Compute centered residual from u8 ground truth Y and f32 predicted Y.
 *
 * residual[i] = clamp(round(gt_y[i] - pred_y[i]) + 128, 0, 255)
 */
extern "C" __global__ void compute_residual_kernel(
    const unsigned char* __restrict__ gt_y,    // [N * H * W]
    const float* __restrict__ pred_y,          // [N * H * W]
    unsigned char* __restrict__ residual,       // [N * H * W]
    int total_pixels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pixels) return;

    float diff = (float)gt_y[idx] - pred_y[idx] + 128.0f;
    diff = roundf(diff);
    diff = fmaxf(0.0f, fminf(255.0f, diff));
    residual[idx] = (unsigned char)diff;
}

/**
 * Reconstruct Y channel from prediction + decoded residual.
 *
 * recon[i] = clamp(pred_y[i] + (residual[i] - 128), 0, 255)
 */
extern "C" __global__ void reconstruct_y_kernel(
    const float* __restrict__ pred_y,          // [N * H * W]
    const unsigned char* __restrict__ residual, // [N * H * W]
    float* __restrict__ recon_y,               // [N * H * W]
    int total_pixels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pixels) return;

    float val = pred_y[idx] + ((float)residual[idx] - 128.0f);
    recon_y[idx] = fmaxf(0.0f, fminf(255.0f, val));
}

/**
 * Bulk u8 → f32 conversion.
 */
extern "C" __global__ void u8_to_f32_kernel(
    const unsigned char* __restrict__ src,
    float* __restrict__ dst,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    dst[idx] = (float)src[idx];
}

/**
 * Bulk f32 → u8 conversion (clamp + round).
 */
extern "C" __global__ void f32_to_u8_kernel(
    const float* __restrict__ src,
    unsigned char* __restrict__ dst,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    dst[idx] = (unsigned char)fmaxf(0.0f, fminf(255.0f, roundf(src[idx])));
}

/**
 * Expand grayscale u8 to interleaved RGB u8 (R=G=B=gray).
 * Used to feed single-channel data to nvjpegEncodeImage with NVJPEG_CSS_GRAY.
 */
extern "C" __global__ void gray_to_rgbi_kernel(
    const unsigned char* __restrict__ gray,  // [total_pixels]
    unsigned char* __restrict__ rgb,          // [total_pixels * 3]
    int total_pixels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pixels) return;
    unsigned char v = gray[idx];
    rgb[idx * 3]     = v;
    rgb[idx * 3 + 1] = v;
    rgb[idx * 3 + 2] = v;
}

/**
 * Expand grayscale u8 to interleaved RGB u8 (R=G=B=gray).
 * Used to feed single-channel data to nvjpegEncodeImage with NVJPEG_CSS_GRAY.
 */
extern "C" __global__ void gray_to_rgbi_kernel(
    const unsigned char* __restrict__ gray,  // [total_pixels]
    unsigned char* __restrict__ rgb,          // [total_pixels * 3]
    int total_pixels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pixels) return;
    unsigned char v = gray[idx];
    rgb[idx * 3]     = v;
    rgb[idx * 3 + 1] = v;
    rgb[idx * 3 + 2] = v;
}

/**
 * YCbCr float32 → RGB u8 (BT.601 inverse)
 *
 * R = Y + 1.402 * (Cr - 128)
 * G = Y - 0.344136 * (Cb - 128) - 0.714136 * (Cr - 128)
 * B = Y + 1.772 * (Cb - 128)
 */
extern "C" __global__ void ycbcr_to_rgb_kernel(
    const float* __restrict__ y_in,    // [N * H * W]
    const float* __restrict__ cb_in,   // [N * H * W]
    const float* __restrict__ cr_in,   // [N * H * W]
    unsigned char* __restrict__ rgb,    // [N * H * W * 3]
    int total_pixels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pixels) return;

    float y  = y_in[idx];
    float cb = cb_in[idx] - 128.0f;
    float cr = cr_in[idx] - 128.0f;

    float r = y + 1.402f * cr;
    float g = y - 0.344136f * cb - 0.714136f * cr;
    float b = y + 1.772f * cb;

    rgb[idx * 3]     = (unsigned char)fmaxf(0.0f, fminf(255.0f, roundf(r)));
    rgb[idx * 3 + 1] = (unsigned char)fmaxf(0.0f, fminf(255.0f, roundf(g)));
    rgb[idx * 3 + 2] = (unsigned char)fmaxf(0.0f, fminf(255.0f, roundf(b)));
}
