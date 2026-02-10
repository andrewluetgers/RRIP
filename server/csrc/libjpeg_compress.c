/*
 * Thin C wrapper around the standard libjpeg62 compress API.
 * Compiled by build.rs (via the `cc` crate) and linked against
 * either mozjpeg or jpegli depending on the Cargo feature flag.
 *
 * We use a C file rather than raw Rust FFI to avoid reproducing
 * the internal layout of jpeg_compress_struct, which varies between
 * libjpeg implementations (standard, mozjpeg, jpegli).
 */

#include <stdio.h>
#include <stdlib.h>
#include <jpeglib.h>

/*
 * Compress raw pixels (RGB or grayscale) to JPEG.
 *
 * Returns 0 on success, non-zero on failure.
 * On success, *out_buf is malloc'd and must be freed by the caller.
 */
int libjpeg_compress(
    const unsigned char *pixels,
    int width,
    int height,
    int components,  /* 1 = grayscale, 3 = RGB */
    int quality,
    unsigned char **out_buf,
    unsigned long *out_size
) {
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;

    *out_buf = NULL;
    *out_size = 0;

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);

    jpeg_mem_dest(&cinfo, out_buf, out_size);

    cinfo.image_width = (JDIMENSION)width;
    cinfo.image_height = (JDIMENSION)height;
    cinfo.input_components = components;
    cinfo.in_color_space = (components == 1) ? JCS_GRAYSCALE : JCS_RGB;

    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, quality, TRUE);

    jpeg_start_compress(&cinfo, TRUE);

    int row_stride = width * components;
    while (cinfo.next_scanline < cinfo.image_height) {
        const unsigned char *row = pixels + cinfo.next_scanline * row_stride;
        JSAMPROW row_ptr = (JSAMPROW)row;
        jpeg_write_scanlines(&cinfo, &row_ptr, 1);
    }

    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);

    return 0;
}
