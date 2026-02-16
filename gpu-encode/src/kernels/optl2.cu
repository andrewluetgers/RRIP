/**
 * OptL2 Gradient Descent Kernel — Batched
 *
 * Processes N families concurrently. Each thread owns one source pixel (one channel).
 *
 * Uses the same center-of-pixel bilinear coordinate mapping as the upsample kernel:
 *   src_coord = (dst_coord + 0.5) / 2.0 - 0.5
 * which is equivalent to:
 *   src_coord = dst_coord * 0.5 - 0.25
 *
 * The gradient is the adjoint (transpose) of this bilinear operator:
 * For each L1 target pixel, we scatter-add the error × weight back to the
 * L2 pixels that contributed to it.
 *
 * Input:  N x H x W x 3 float32 (L2 current, modified in place)
 *         N x H x W x 3 float32 (L2 original, frozen)
 *         N x (2H) x (2W) x 3 float32 (L1 target)
 * Output: N x H x W x 3 float32 (optimized L2, in l2_current)
 */

extern "C" __global__ void optl2_step_kernel(
    float* __restrict__ l2_current,     // [N, H, W, 3]
    const float* __restrict__ l2_orig,  // [N, H, W, 3]
    const float* __restrict__ l1_target,// [N, 2H, 2W, 3]
    int N, int H, int W,
    float lr,
    float max_delta
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * H * W * 3;
    if (idx >= total) return;

    // Decode L2 pixel indices
    int c = idx % 3;
    int tmp = idx / 3;
    int x = tmp % W;
    tmp /= W;
    int y = tmp % H;
    int n = tmp / H;

    int W2 = W * 2;
    int H2 = H * 2;

    // This pixel's original and current value
    float orig = l2_orig[idx];
    float cur = l2_current[idx];

    // Compute gradient by iterating over L1 (target) pixels that this L2 pixel
    // contributes to via bilinear upsample.
    //
    // The upsample mapping is: src_x = (dst_x + 0.5) / 2.0 - 0.5
    // So dst_x maps to src_x. We need to find which dst_x values have
    // this L2 pixel (x, y) as one of their 4 bilinear neighbors.
    //
    // src_x = dst_x * 0.5 - 0.25
    // For L2 pixel x, it's a neighbor when floor(src_x) == x or floor(src_x) == x-1
    // i.e., x <= src_x < x+1 or x-1 <= src_x < x
    // i.e., (x + 0.25) / 0.5 <= dst_x  => dst_x >= 2*x + 0.5
    //       (x + 1.25) / 0.5 >= dst_x  => dst_x <= 2*x + 2.5
    // So dst_x range is [2*x, 2*x+2] (integer values)
    //
    // Similarly for y.

    float grad = 0.0f;

    for (int ty = 2 * y; ty <= min(2 * y + 2, H2 - 1); ty++) {
        // Source coordinate for this target pixel (same as upsample kernel)
        float sy = ((float)ty + 0.5f) / 2.0f - 0.5f;
        sy = fmaxf(0.0f, fminf((float)(H - 1), sy));

        int sy0 = (int)floorf(sy);
        int sy1 = min(sy0 + 1, H - 1);
        float fy = sy - (float)sy0;

        // This L2 pixel's y-weight
        float wy;
        if (y == sy0 && y == sy1) {
            wy = 1.0f;  // Edge case: clamped, this pixel gets all weight
        } else if (y == sy0) {
            wy = 1.0f - fy;
        } else if (y == sy1) {
            wy = fy;
        } else {
            continue;  // This pixel doesn't contribute to this target row
        }

        for (int tx = 2 * x; tx <= min(2 * x + 2, W2 - 1); tx++) {
            float sx = ((float)tx + 0.5f) / 2.0f - 0.5f;
            sx = fmaxf(0.0f, fminf((float)(W - 1), sx));

            int sx0 = (int)floorf(sx);
            int sx1 = min(sx0 + 1, W - 1);
            float fx = sx - (float)sx0;

            // This L2 pixel's x-weight
            float wx;
            if (x == sx0 && x == sx1) {
                wx = 1.0f;
            } else if (x == sx0) {
                wx = 1.0f - fx;
            } else if (x == sx1) {
                wx = fx;
            } else {
                continue;
            }

            float w = wx * wy;
            if (w < 1e-6f) continue;

            // Compute predicted value at (tx, ty, c) via bilinear interpolation
            // of l2_current, using the SAME mapping as the upsample kernel
            float v00 = l2_current[((n * H + sy0) * W + sx0) * 3 + c];
            float v01 = l2_current[((n * H + sy0) * W + sx1) * 3 + c];
            float v10 = l2_current[((n * H + sy1) * W + sx0) * 3 + c];
            float v11 = l2_current[((n * H + sy1) * W + sx1) * 3 + c];

            float predicted = v00 * (1.0f - fx) * (1.0f - fy)
                            + v01 * fx * (1.0f - fy)
                            + v10 * (1.0f - fx) * fy
                            + v11 * fx * fy;

            // Target value
            float target = l1_target[((n * H2 + ty) * W2 + tx) * 3 + c];

            // Error and gradient contribution (adjoint of bilinear)
            // d(||target - predicted||^2)/d(current[x,y]) = -2 * (target - predicted) * w
            // We minimize, so update = +2 * error * w
            float error = target - predicted;
            grad += 2.0f * error * w;
        }
    }

    // Update
    float new_val = cur + lr * grad;

    // Clamp to [original - max_delta, original + max_delta] and [0, 255]
    new_val = fmaxf(new_val, fmaxf(orig - max_delta, 0.0f));
    new_val = fminf(new_val, fminf(orig + max_delta, 255.0f));

    l2_current[idx] = new_val;
}
