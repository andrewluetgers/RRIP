//! CUDA kernel bindings and GPU context management.
//!
//! This module provides:
//! - `GpuContext`: CUDA device, stream, and loaded kernel modules
//! - Kernel launch wrappers for each .cu file

use std::sync::Arc;

use anyhow::{anyhow, Result};
use cudarc::driver::{
    CudaContext, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use tracing::info;

/// CUDA GPU context — owns device handle, stream, and loaded kernel modules.
pub struct GpuContext {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    mod_upsample: Arc<CudaModule>,
    mod_downsample: Arc<CudaModule>,
    mod_residual: Arc<CudaModule>,
    mod_composite: Arc<CudaModule>,
    mod_optl2: Arc<CudaModule>,
    mod_sharpen: Arc<CudaModule>,
}

impl GpuContext {
    /// Initialize CUDA and select a device. Loads all PTX modules.
    pub fn new(device: usize) -> Result<Self> {
        info!("Initializing CUDA device {}", device);
        let ctx = CudaContext::new(device)
            .map_err(|e| anyhow!("CUDA init failed: {}", e))?;
        let stream = ctx.default_stream();

        let ptx_dir = env!("CUDA_PTX_DIR");
        info!("Loading PTX from {}", ptx_dir);

        let mod_upsample = ctx
            .load_module(Ptx::from_file(format!("{}/upsample.ptx", ptx_dir)))
            .map_err(|e| anyhow!("Failed to load upsample.ptx: {}", e))?;
        let mod_downsample = ctx
            .load_module(Ptx::from_file(format!("{}/downsample.ptx", ptx_dir)))
            .map_err(|e| anyhow!("Failed to load downsample.ptx: {}", e))?;
        let mod_residual = ctx
            .load_module(Ptx::from_file(format!("{}/residual.ptx", ptx_dir)))
            .map_err(|e| anyhow!("Failed to load residual.ptx: {}", e))?;
        let mod_composite = ctx
            .load_module(Ptx::from_file(format!("{}/composite.ptx", ptx_dir)))
            .map_err(|e| anyhow!("Failed to load composite.ptx: {}", e))?;
        let mod_optl2 = ctx
            .load_module(Ptx::from_file(format!("{}/optl2.ptx", ptx_dir)))
            .map_err(|e| anyhow!("Failed to load optl2.ptx: {}", e))?;
        let mod_sharpen = ctx
            .load_module(Ptx::from_file(format!("{}/sharpen.ptx", ptx_dir)))
            .map_err(|e| anyhow!("Failed to load sharpen.ptx: {}", e))?;

        info!("All CUDA kernels loaded");

        Ok(GpuContext {
            ctx,
            stream,
            mod_upsample,
            mod_downsample,
            mod_residual,
            mod_composite,
            mod_optl2,
            mod_sharpen,
        })
    }

    pub fn device(&self) -> usize {
        0
    }

    // -----------------------------------------------------------------------
    // Kernel launch wrappers
    // -----------------------------------------------------------------------

    /// Bilinear 2x upsample on GPU.
    ///
    /// src: [N, H, W, C] f32 on device
    /// Returns: [N, 2H, 2W, C] f32 on device
    pub fn upsample_bilinear_2x(
        &self,
        src: &CudaSlice<f32>,
        n: i32,
        h: i32,
        w: i32,
        c: i32,
    ) -> Result<CudaSlice<f32>> {
        let h2 = h * 2;
        let w2 = w * 2;
        let total = (n * h2 * w2 * c) as usize;
        let mut dst = unsafe { self.stream.alloc::<f32>(total) }
            .map_err(|e| anyhow!("alloc failed: {}", e))?;

        let f = self.mod_upsample
            .load_function("upsample_bilinear_2x_kernel")
            .map_err(|e| anyhow!("load function failed: {}", e))?;

        let cfg = LaunchConfig::for_num_elems(total as u32);
        let mut launch = self.stream.launch_builder(&f);
        launch.arg(src);
        launch.arg(&mut dst);
        launch.arg(&n);
        launch.arg(&h);
        launch.arg(&w);
        launch.arg(&c);
        unsafe { launch.launch(cfg) }
            .map_err(|e| anyhow!("upsample kernel launch failed: {}", e))?;

        Ok(dst)
    }

    /// RGB u8 → float32 YCbCr (BT.601) on GPU.
    ///
    /// rgb: [total_pixels * 3] u8 on device
    /// Returns: (y, cb, cr) each [total_pixels] f32 on device
    pub fn rgb_to_ycbcr_f32(
        &self,
        rgb: &CudaSlice<u8>,
        total_pixels: i32,
    ) -> Result<(CudaSlice<f32>, CudaSlice<f32>, CudaSlice<f32>)> {
        let n = total_pixels as usize;
        let mut y_out = unsafe { self.stream.alloc::<f32>(n) }
            .map_err(|e| anyhow!("alloc y failed: {}", e))?;
        let mut cb_out = unsafe { self.stream.alloc::<f32>(n) }
            .map_err(|e| anyhow!("alloc cb failed: {}", e))?;
        let mut cr_out = unsafe { self.stream.alloc::<f32>(n) }
            .map_err(|e| anyhow!("alloc cr failed: {}", e))?;

        let f = self.mod_residual
            .load_function("rgb_to_ycbcr_f32_kernel")
            .map_err(|e| anyhow!("load function failed: {}", e))?;

        let cfg = LaunchConfig::for_num_elems(total_pixels as u32);
        let mut launch = self.stream.launch_builder(&f);
        launch.arg(rgb);
        launch.arg(&mut y_out);
        launch.arg(&mut cb_out);
        launch.arg(&mut cr_out);
        launch.arg(&total_pixels);
        unsafe { launch.launch(cfg) }
            .map_err(|e| anyhow!("rgb_to_ycbcr kernel launch failed: {}", e))?;

        Ok((y_out, cb_out, cr_out))
    }

    /// Compute centered residual: residual = clamp(round(gt_y - pred_y + 128), 0, 255)
    pub fn compute_residual(
        &self,
        gt_y: &CudaSlice<u8>,
        pred_y: &CudaSlice<f32>,
        total_pixels: i32,
    ) -> Result<CudaSlice<u8>> {
        let mut residual = unsafe { self.stream.alloc::<u8>(total_pixels as usize) }
            .map_err(|e| anyhow!("alloc residual failed: {}", e))?;

        let f = self.mod_residual
            .load_function("compute_residual_kernel")
            .map_err(|e| anyhow!("load function failed: {}", e))?;

        let cfg = LaunchConfig::for_num_elems(total_pixels as u32);
        let mut launch = self.stream.launch_builder(&f);
        launch.arg(gt_y);
        launch.arg(pred_y);
        launch.arg(&mut residual);
        launch.arg(&total_pixels);
        unsafe { launch.launch(cfg) }
            .map_err(|e| anyhow!("compute_residual kernel launch failed: {}", e))?;

        Ok(residual)
    }

    /// Reconstruct Y: recon = clamp(pred_y + (residual - 128), 0, 255)
    pub fn reconstruct_y(
        &self,
        pred_y: &CudaSlice<f32>,
        residual: &CudaSlice<u8>,
        total_pixels: i32,
    ) -> Result<CudaSlice<f32>> {
        let mut recon = unsafe { self.stream.alloc::<f32>(total_pixels as usize) }
            .map_err(|e| anyhow!("alloc recon failed: {}", e))?;

        let f = self.mod_residual
            .load_function("reconstruct_y_kernel")
            .map_err(|e| anyhow!("load function failed: {}", e))?;

        let cfg = LaunchConfig::for_num_elems(total_pixels as u32);
        let mut launch = self.stream.launch_builder(&f);
        launch.arg(pred_y);
        launch.arg(residual);
        launch.arg(&mut recon);
        launch.arg(&total_pixels);
        unsafe { launch.launch(cfg) }
            .map_err(|e| anyhow!("reconstruct_y kernel launch failed: {}", e))?;

        Ok(recon)
    }

    /// YCbCr float32 → RGB u8 (BT.601 inverse)
    pub fn ycbcr_to_rgb(
        &self,
        y_in: &CudaSlice<f32>,
        cb_in: &CudaSlice<f32>,
        cr_in: &CudaSlice<f32>,
        total_pixels: i32,
    ) -> Result<CudaSlice<u8>> {
        let mut rgb = unsafe { self.stream.alloc::<u8>((total_pixels * 3) as usize) }
            .map_err(|e| anyhow!("alloc rgb failed: {}", e))?;

        let f = self.mod_residual
            .load_function("ycbcr_to_rgb_kernel")
            .map_err(|e| anyhow!("load function failed: {}", e))?;

        let cfg = LaunchConfig::for_num_elems(total_pixels as u32);
        let mut launch = self.stream.launch_builder(&f);
        launch.arg(y_in);
        launch.arg(cb_in);
        launch.arg(cr_in);
        launch.arg(&mut rgb);
        launch.arg(&total_pixels);
        unsafe { launch.launch(cfg) }
            .map_err(|e| anyhow!("ycbcr_to_rgb kernel launch failed: {}", e))?;

        Ok(rgb)
    }

    /// Bulk u8 → f32 conversion on GPU.
    pub fn u8_to_f32(
        &self,
        src: &CudaSlice<u8>,
        total: i32,
    ) -> Result<CudaSlice<f32>> {
        let mut dst = unsafe { self.stream.alloc::<f32>(total as usize) }
            .map_err(|e| anyhow!("alloc f32 failed: {}", e))?;

        let f = self.mod_residual
            .load_function("u8_to_f32_kernel")
            .map_err(|e| anyhow!("load function failed: {}", e))?;

        let cfg = LaunchConfig::for_num_elems(total as u32);
        let mut launch = self.stream.launch_builder(&f);
        launch.arg(src);
        launch.arg(&mut dst);
        launch.arg(&total);
        unsafe { launch.launch(cfg) }
            .map_err(|e| anyhow!("u8_to_f32 kernel launch failed: {}", e))?;

        Ok(dst)
    }

    /// Bulk f32 → u8 conversion on GPU (clamp + round).
    pub fn f32_to_u8(
        &self,
        src: &CudaSlice<f32>,
        total: i32,
    ) -> Result<CudaSlice<u8>> {
        let mut dst = unsafe { self.stream.alloc::<u8>(total as usize) }
            .map_err(|e| anyhow!("alloc u8 failed: {}", e))?;

        let f = self.mod_residual
            .load_function("f32_to_u8_kernel")
            .map_err(|e| anyhow!("load function failed: {}", e))?;

        let cfg = LaunchConfig::for_num_elems(total as u32);
        let mut launch = self.stream.launch_builder(&f);
        launch.arg(src);
        launch.arg(&mut dst);
        launch.arg(&total);
        unsafe { launch.launch(cfg) }
            .map_err(|e| anyhow!("f32_to_u8 kernel launch failed: {}", e))?;

        Ok(dst)
    }

    /// Expand grayscale u8 to interleaved RGB u8 (R=G=B=gray) on GPU.
    pub fn gray_to_rgbi(
        &self,
        gray: &CudaSlice<u8>,
        total_pixels: i32,
    ) -> Result<CudaSlice<u8>> {
        let mut rgb = unsafe { self.stream.alloc::<u8>((total_pixels * 3) as usize) }
            .map_err(|e| anyhow!("alloc rgb failed: {}", e))?;

        let f = self.mod_residual
            .load_function("gray_to_rgbi_kernel")
            .map_err(|e| anyhow!("load function failed: {}", e))?;

        let cfg = LaunchConfig::for_num_elems(total_pixels as u32);
        let mut launch = self.stream.launch_builder(&f);
        launch.arg(gray);
        launch.arg(&mut rgb);
        launch.arg(&total_pixels);
        unsafe { launch.launch(cfg) }
            .map_err(|e| anyhow!("gray_to_rgbi kernel launch failed: {}", e))?;

        Ok(rgb)
    }

    /// Composite 16 tiles into a 4x4 family canvas.
    ///
    /// tiles: [N * tiles_per_row^2 * tile_h * tile_w * 3] u8 on device
    /// Returns: [N * canvas_h * canvas_w * 3] u8 on device
    pub fn composite_tiles(
        &self,
        tiles: &CudaSlice<u8>,
        n: i32,
        tile_w: i32,
        tile_h: i32,
        canvas_w: i32,
        canvas_h: i32,
        tiles_per_row: i32,
    ) -> Result<CudaSlice<u8>> {
        let total_out = (n * canvas_h * canvas_w * 3) as usize;
        let mut canvas = self.stream.alloc_zeros::<u8>(total_out)
            .map_err(|e| anyhow!("alloc canvas failed: {}", e))?;

        let f = self.mod_composite
            .load_function("composite_kernel")
            .map_err(|e| anyhow!("load function failed: {}", e))?;

        let total_threads = (n * canvas_h * canvas_w * 3) as u32;
        let cfg = LaunchConfig::for_num_elems(total_threads);
        let mut launch = self.stream.launch_builder(&f);
        launch.arg(tiles);
        launch.arg(&mut canvas);
        launch.arg(&n);
        launch.arg(&tile_w);
        launch.arg(&tile_h);
        launch.arg(&canvas_w);
        launch.arg(&canvas_h);
        launch.arg(&tiles_per_row);
        unsafe { launch.launch(cfg) }
            .map_err(|e| anyhow!("composite kernel launch failed: {}", e))?;

        Ok(canvas)
    }

    /// OptL2 gradient descent step.
    pub fn optl2_step(
        &self,
        l2_current: &mut CudaSlice<f32>,
        l2_orig: &CudaSlice<f32>,
        l1_target: &CudaSlice<f32>,
        n: i32,
        h: i32,
        w: i32,
        lr: f32,
        max_delta: f32,
    ) -> Result<()> {
        let f = self.mod_optl2
            .load_function("optl2_step_kernel")
            .map_err(|e| anyhow!("load function failed: {}", e))?;

        let total = (n * h * w * 3) as u32;
        let cfg = LaunchConfig::for_num_elems(total);
        let mut launch = self.stream.launch_builder(&f);
        launch.arg(l2_current);
        launch.arg(l2_orig);
        launch.arg(l1_target);
        launch.arg(&n);
        launch.arg(&h);
        launch.arg(&w);
        launch.arg(&lr);
        launch.arg(&max_delta);
        unsafe { launch.launch(cfg) }
            .map_err(|e| anyhow!("optl2 kernel launch failed: {}", e))?;

        Ok(())
    }

    /// Lanczos3 downsample (2-pass separable).
    pub fn downsample_lanczos3(
        &self,
        src: &CudaSlice<f32>,
        n: i32,
        h_src: i32,
        w_src: i32,
        h_dst: i32,
        w_dst: i32,
        c: i32,
    ) -> Result<CudaSlice<f32>> {
        // Horizontal pass: [N, H_src, W_src, C] → [N, H_src, W_dst, C]
        let tmp_size = (n * h_src * w_dst * c) as usize;
        let mut tmp = unsafe { self.stream.alloc::<f32>(tmp_size) }
            .map_err(|e| anyhow!("alloc tmp failed: {}", e))?;

        let scale_x = w_dst as f32 / w_src as f32;

        let f_h = self.mod_upsample
            .load_function("downsample_lanczos3_h_kernel")
            .map_err(|e| anyhow!("load h kernel failed: {}", e))?;

        let cfg_h = LaunchConfig::for_num_elems(tmp_size as u32);
        let mut launch_h = self.stream.launch_builder(&f_h);
        launch_h.arg(src);
        launch_h.arg(&mut tmp);
        launch_h.arg(&n);
        launch_h.arg(&h_src);
        launch_h.arg(&w_src);
        launch_h.arg(&w_dst);
        launch_h.arg(&c);
        launch_h.arg(&scale_x);
        unsafe { launch_h.launch(cfg_h) }
            .map_err(|e| anyhow!("lanczos3_h launch failed: {}", e))?;

        // Vertical pass: [N, H_src, W_dst, C] → [N, H_dst, W_dst, C]
        let dst_size = (n * h_dst * w_dst * c) as usize;
        let mut dst = unsafe { self.stream.alloc::<f32>(dst_size) }
            .map_err(|e| anyhow!("alloc dst failed: {}", e))?;

        let scale_y = h_dst as f32 / h_src as f32;

        let f_v = self.mod_upsample
            .load_function("downsample_lanczos3_v_kernel")
            .map_err(|e| anyhow!("load v kernel failed: {}", e))?;

        let cfg_v = LaunchConfig::for_num_elems(dst_size as u32);
        let mut launch_v = self.stream.launch_builder(&f_v);
        launch_v.arg(&tmp);
        launch_v.arg(&mut dst);
        launch_v.arg(&n);
        launch_v.arg(&h_src);
        launch_v.arg(&h_dst);
        launch_v.arg(&w_dst);
        launch_v.arg(&c);
        launch_v.arg(&scale_y);
        unsafe { launch_v.launch(cfg_v) }
            .map_err(|e| anyhow!("lanczos3_v launch failed: {}", e))?;

        Ok(dst)
    }

    /// Box filter 2×2 downsample on RGB u8 interleaved buffer.
    /// Each output pixel = average of 2×2 block in source.
    pub fn downsample_box_2x(
        &self,
        src: &CudaSlice<u8>,
        src_width: u32,
        src_height: u32,
    ) -> Result<CudaSlice<u8>> {
        let dst_width = src_width / 2;
        let dst_height = src_height / 2;
        let dst_total = (dst_width * dst_height * 3) as usize;

        let mut dst = unsafe { self.stream.alloc::<u8>(dst_total) }
            .map_err(|e| anyhow!("downsample dst alloc failed: {}", e))?;

        let f = self.mod_downsample
            .load_function("downsample_2x_box_kernel")
            .map_err(|e| anyhow!("load downsample_2x_box_kernel failed: {}", e))?;

        let cfg = LaunchConfig::for_num_elems(dst_total as u32);
        let mut launch = self.stream.launch_builder(&f);
        launch.arg(src);
        launch.arg(&mut dst);
        launch.arg(&(src_width as i32));
        launch.arg(&(src_height as i32));
        unsafe { launch.launch(cfg) }
            .map_err(|e| anyhow!("downsample_2x_box launch failed: {}", e))?;

        Ok(dst)
    }

    /// Apply unsharp mask sharpening to an interleaved RGB u8 buffer.
    /// Uses same integer/fixed-point math as CPU (server/src/core/sharpen.rs):
    /// Pass 1: horizontal blur → u16 (no rounding), Pass 2: fixed-point 8.8 sharpen.
    pub fn sharpen_l2(
        &self,
        src: &CudaSlice<u8>,
        width: u32,
        height: u32,
        strength: f32,
    ) -> Result<CudaSlice<u8>> {
        let total = (width * height * 3) as u32;
        let cfg = LaunchConfig::for_num_elems(total);

        // Allocate temp buffer for horizontal blur pass (u16, no intermediate rounding)
        let temp: CudaSlice<u16> = unsafe { self.stream.alloc(total as usize) }
            .map_err(|e| anyhow!("sharpen temp alloc failed: {}", e))?;

        // Allocate output buffer
        let dst: CudaSlice<u8> = unsafe { self.stream.alloc(total as usize) }
            .map_err(|e| anyhow!("sharpen dst alloc failed: {}", e))?;

        let w = width as i32;
        let h = height as i32;
        // Fixed-point 8.8 strength (matches CPU: (strength * 256.0).round() as i32)
        let strength_i = (strength * 256.0).round() as i32;

        // Pass 1: horizontal blur → u16
        let f_hblur = self.mod_sharpen
            .load_function("unsharp_hblur_kernel")
            .map_err(|e| anyhow!("load unsharp_hblur_kernel failed: {}", e))?;
        let mut launch_h = self.stream.launch_builder(&f_hblur);
        launch_h.arg(src);
        launch_h.arg(&temp);
        launch_h.arg(&w);
        launch_h.arg(&h);
        unsafe { launch_h.launch(cfg) }
            .map_err(|e| anyhow!("unsharp_hblur launch failed: {}", e))?;

        // Pass 2: vertical blur + fixed-point sharpen
        let f_vblur = self.mod_sharpen
            .load_function("unsharp_vblur_sharpen_kernel")
            .map_err(|e| anyhow!("load unsharp_vblur_sharpen_kernel failed: {}", e))?;
        let mut launch_v = self.stream.launch_builder(&f_vblur);
        launch_v.arg(src);
        launch_v.arg(&temp);
        launch_v.arg(&dst);
        launch_v.arg(&w);
        launch_v.arg(&h);
        launch_v.arg(&strength_i);
        unsafe { launch_v.launch(cfg) }
            .map_err(|e| anyhow!("unsharp_vblur_sharpen launch failed: {}", e))?;

        Ok(dst)
    }

    /// Synchronize the default stream (wait for all GPU work to complete).
    pub fn sync(&self) -> Result<()> {
        self.stream.synchronize()
            .map_err(|e| anyhow!("stream sync failed: {}", e))
    }
}
