/**
 * DICOM Tile Compositing Kernel
 *
 * Copies decoded JPEG tiles (224x224x3) into a larger family canvas (896x896x3).
 * One thread per output pixel.
 *
 * Input:  16 decoded JPEG tiles per family, as a flat array
 *         [N, 16, tile_h, tile_w, 3] u8
 * Output: [N, canvas_h, canvas_w, 3] u8
 *
 * Tile layout within canvas (4x4 grid):
 *   (0,0) (1,0) (2,0) (3,0)
 *   (0,1) (1,1) (2,1) (3,1)
 *   (0,2) (1,2) (2,2) (3,2)
 *   (0,3) (1,3) (2,3) (3,3)
 */

extern "C" __global__ void composite_kernel(
    const unsigned char* __restrict__ tiles,   // [N * 16 * tile_h * tile_w * 3]
    unsigned char* __restrict__ canvas,         // [N * canvas_h * canvas_w * 3]
    int N,
    int tile_w, int tile_h,
    int canvas_w, int canvas_h,
    int tiles_per_row  // 4 for 4x4 grid
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * canvas_h * canvas_w * 3;
    if (idx >= total) return;

    // Decode output index
    int c = idx % 3;
    int tmp = idx / 3;
    int cx = tmp % canvas_w;
    tmp /= canvas_w;
    int cy = tmp % canvas_h;
    int n = tmp / canvas_h;

    // Determine which tile this pixel belongs to
    int tx = cx / tile_w;  // tile column (0..3)
    int ty = cy / tile_h;  // tile row (0..3)
    int lx = cx % tile_w;  // local x within tile
    int ly = cy % tile_h;  // local y within tile

    // Bounds check (for edge tiles)
    if (tx >= tiles_per_row || ty >= tiles_per_row) {
        canvas[idx] = 255; // white fill
        return;
    }

    // Source tile index within the family
    int tile_idx = ty * tiles_per_row + tx;  // 0..15
    int tile_pixels = tile_h * tile_w;

    // Read from source tile
    int src_idx = ((n * 16 + tile_idx) * tile_pixels + ly * tile_w + lx) * 3 + c;
    canvas[idx] = tiles[src_idx];
}
