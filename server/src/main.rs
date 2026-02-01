use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use axum::extract::{Path as AxumPath, Query, State};
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::get;
use axum::Router;
use bytes::Bytes;
use clap::Parser;
use image::codecs::jpeg::JpegEncoder;
use image::imageops::FilterType;
use image::{DynamicImage, GrayImage, RgbImage};
use moka::sync::Cache;
use memmap2::Mmap;
use sysinfo::System;
use rayon::prelude::*;
use tokio::sync::{mpsc, Semaphore};
use tokio::task;
use std::time::Instant;
use tower_http::trace::TraceLayer;
use tracing::info;

mod fast_upsample_ycbcr;
use fast_upsample_ycbcr::{YCbCrPlanes, upsample_2x_channel, upsample_2x_nearest, upsample_4x_nearest};

mod turbojpeg_optimized;
use turbojpeg_optimized::{load_luma_turbo, decode_luma_turbo, encode_jpeg_turbo, load_rgb_turbo};

#[derive(Parser, Debug)]
#[command(name = "rrip-tile-server")]
struct Args {
    #[arg(long, default_value = "data")]
    slides_root: PathBuf,
    #[arg(long, default_value = "residuals_q32")]
    residuals_dir: String,
    #[arg(long, default_value_t = 90)]
    tile_quality: u8,
    #[arg(long, default_value_t = 2048)]
    cache_entries: usize,
    #[arg(long, default_value_t = 8080)]
    port: u16,
    #[arg(long, default_value_t = false)]
    timing_breakdown: bool,
    #[arg(long)]
    write_generated_dir: Option<PathBuf>,
    #[arg(long, default_value_t = 2048)]
    write_queue_size: usize,
    #[arg(long, default_value_t = 30)]
    metrics_interval_secs: u64,
    #[arg(long, default_value_t = 128)]
    buffer_pool_size: usize,
    #[arg(long)]
    residual_pack_dir: Option<PathBuf>,
    #[arg(long)]
    rayon_threads: Option<usize>,
    #[arg(long, default_value_t = 8)]
    tokio_workers: usize,
    #[arg(long, default_value_t = 32)]
    tokio_blocking_threads: usize,
    #[arg(long, default_value_t = 32)]
    max_inflight_families: usize,
    #[arg(long, default_value_t = false)]
    prewarm_on_l2: bool,
}

#[derive(Clone)]
struct AppState {
    slides: Arc<HashMap<String, Slide>>,
    cache: Arc<Cache<TileKey, Bytes>>,
    tile_quality: u8,
    timing_breakdown: bool,
    writer: Option<mpsc::Sender<WriteJob>>,
    write_generated_dir: Option<PathBuf>,
    metrics: Arc<Mutex<Metrics>>,
    buffer_pool: Arc<BufferPool>,
    pack_dir: Option<PathBuf>,
    inflight: Arc<InflightFamilies>,
    inflight_limit: Arc<Semaphore>,
    prewarm_on_l2: bool,
}

#[derive(serde::Deserialize)]
struct ModeQuery {
    mode: Option<String>,
}

#[derive(Clone, Debug)]
struct Slide {
    slide_id: String,
    dzi_path: PathBuf,
    files_dir: PathBuf,
    residuals_dir: PathBuf,
    tile_size: u32,
    max_level: u32,
    l0: u32,
    l1: u32,
    l2: u32,
}

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
struct TileKey {
    slide_id: String,
    level: u32,
    x: u32,
    y: u32,
}

#[derive(Clone)]
struct WriteJob {
    path: PathBuf,
    bytes: Bytes,
}

struct FamilyResult {
    tiles: HashMap<TileKey, Bytes>,
    stats: Option<FamilyStats>,
}

struct FamilyStats {
    l2_decode_ms: u128,
    l1_resize_ms: u128,
    l1_residual_ms: u128,
    l1_encode_ms: u128,
    l0_resize_ms: u128,
    l0_residual_ms: u128,
    l0_encode_ms: u128,
    total_ms: u128,
    residuals_l1: usize,
    residuals_l0: usize,
    l1_parallel_max: usize,
    l0_parallel_max: usize,
}

struct InflightFamilies {
    current: AtomicUsize,
    max: AtomicUsize,
}

impl InflightFamilies {
    fn new() -> Self {
        Self {
            current: AtomicUsize::new(0),
            max: AtomicUsize::new(0),
        }
    }

    fn enter(&self) -> InflightGuard<'_> {
        let cur = self.current.fetch_add(1, Ordering::SeqCst) + 1;
        self.max.fetch_max(cur, Ordering::SeqCst);
        InflightGuard { stats: self }
    }

    fn take_max(&self) -> usize {
        self.max.swap(self.current.load(Ordering::SeqCst), Ordering::SeqCst)
    }

    fn current(&self) -> usize {
        self.current.load(Ordering::SeqCst)
    }
}

struct InflightGuard<'a> {
    stats: &'a InflightFamilies,
}

impl<'a> Drop for InflightGuard<'a> {
    fn drop(&mut self) {
        self.stats.current.fetch_sub(1, Ordering::SeqCst);
    }
}

struct PackIndexEntry {
    level_kind: u8,
    idx_in_parent: u8,
    offset: u32,
    length: u32,
}

struct BufferPool {
    buffers: Mutex<Vec<Vec<u8>>>,
    total: usize,
    in_use: AtomicUsize,
    in_use_max: AtomicUsize,
}

impl BufferPool {
    fn new(count: usize) -> Self {
        let mut bufs = Vec::with_capacity(count);
        for _ in 0..count {
            bufs.push(Vec::new());
        }
        Self {
            buffers: Mutex::new(bufs),
            total: count,
            in_use: AtomicUsize::new(0),
            in_use_max: AtomicUsize::new(0),
        }
    }

    fn get(&self, len: usize) -> Vec<u8> {
        let mut guard = self.buffers.lock().unwrap();
        let mut buf = guard.pop().unwrap_or_default();
        let in_use = self.in_use.fetch_add(1, Ordering::SeqCst) + 1;
        self.in_use_max.fetch_max(in_use, Ordering::SeqCst);
        if buf.len() != len {
            buf.resize(len, 0);
        }
        buf
    }

    fn put(&self, mut buf: Vec<u8>) {
        buf.clear();
        self.buffers.lock().unwrap().push(buf);
        self.in_use.fetch_sub(1, Ordering::SeqCst);
    }

    fn stats(&self) -> (usize, usize, usize) {
        let available = self.buffers.lock().unwrap().len();
        let in_use = self.in_use.load(Ordering::SeqCst);
        let in_use_max = self.in_use_max.swap(in_use, Ordering::SeqCst);
        (self.total, available, in_use_max)
    }
}

struct ParallelStats {
    current: AtomicUsize,
    max: AtomicUsize,
}

impl ParallelStats {
    fn new() -> Self {
        Self {
            current: AtomicUsize::new(0),
            max: AtomicUsize::new(0),
        }
    }

    fn enter(&self) -> ParallelGuard<'_> {
        let cur = self.current.fetch_add(1, Ordering::SeqCst) + 1;
        self.max.fetch_max(cur, Ordering::SeqCst);
        ParallelGuard { stats: self }
    }

    fn take_max(&self) -> usize {
        self.max.swap(self.current.load(Ordering::SeqCst), Ordering::SeqCst)
    }
}

struct ParallelGuard<'a> {
    stats: &'a ParallelStats,
}

impl<'a> Drop for ParallelGuard<'a> {
    fn drop(&mut self) {
        self.stats.current.fetch_sub(1, Ordering::SeqCst);
    }
}

#[derive(Default, Clone)]
struct Metrics {
    tile_total: u64,
    tile_baseline: u64,
    tile_cache_hit: u64,
    tile_generated: u64,
    tile_fallback: u64,
    tile_residual_view: u64,
    tile_ms_sum: u128,
    tile_ms_max: u128,
    family_generated: u64,
    family_ms_sum: u128,
    family_ms_max: u128,
}

impl Metrics {
    fn record_tile(&mut self, kind: &str, ms: u128) {
        self.tile_total += 1;
        self.tile_ms_sum += ms;
        if ms > self.tile_ms_max {
            self.tile_ms_max = ms;
        }
        match kind {
            "baseline" => self.tile_baseline += 1,
            "cache_hit" => self.tile_cache_hit += 1,
            "generated" => self.tile_generated += 1,
            "fallback" => self.tile_fallback += 1,
            "residual_view" => self.tile_residual_view += 1,
            _ => {}
        }
    }

    fn record_family(&mut self, ms: u128) {
        self.family_generated += 1;
        self.family_ms_sum += ms;
        if ms > self.family_ms_max {
            self.family_ms_max = ms;
        }
    }

    fn take(&mut self) -> Metrics {
        let snapshot = self.clone();
        *self = Metrics::default();
        snapshot
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .init();

    if let Some(threads) = args.rayon_threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .map_err(|e| anyhow!("rayon init failed: {}", e))?;
    }

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(args.tokio_workers)
        .max_blocking_threads(args.tokio_blocking_threads)
        .enable_all()
        .build()?;

    runtime.block_on(async_main(args))
}

async fn async_main(args: Args) -> Result<()> {
    let slides = discover_slides(&args.slides_root, &args.residuals_dir)?;
    if slides.is_empty() {
        return Err(anyhow!(
            "No slides found under {}",
            args.slides_root.display()
        ));
    }
    for slide in slides.values() {
        info!(
            "slide_id={} max_level={} tile_size={} l0={} l1={} l2={}",
            slide.slide_id, slide.max_level, slide.tile_size, slide.l0, slide.l1, slide.l2
        );
    }

    let slides = Arc::new(slides);
    let write_generated_dir = args.write_generated_dir.clone();
    let writer = if let Some(dir) = write_generated_dir.clone() {
        fs::create_dir_all(&dir).ok();
        let (tx, mut rx) = mpsc::channel::<WriteJob>(args.write_queue_size);
        tokio::spawn(async move {
            while let Some(job) = rx.recv().await {
                if let Some(parent) = job.path.parent() {
                    if let Err(err) = fs::create_dir_all(parent) {
                        info!("write error mkdir {}: {}", parent.display(), err);
                        continue;
                    }
                }
                if let Err(err) = fs::write(&job.path, &job.bytes) {
                    info!("write error {}: {}", job.path.display(), err);
                }
            }
        });
        info!("write_generated_dir enabled: {}", dir.display());
        Some(tx)
    } else {
        None
    };
    let cache = Cache::builder()
        .max_capacity(args.cache_entries as u64)
        .time_to_idle(Duration::from_secs(300))  // Expire items after 5 min of inactivity
        .build();
    let inflight = Arc::new(InflightFamilies::new());
    let inflight_limit = Arc::new(Semaphore::new(args.max_inflight_families));
    let state = AppState {
        slides: slides.clone(),
        cache: Arc::new(cache),
        tile_quality: args.tile_quality,
        timing_breakdown: args.timing_breakdown,
        writer,
        write_generated_dir,
        metrics: Arc::new(Mutex::new(Metrics::default())),
        buffer_pool: Arc::new(BufferPool::new(args.buffer_pool_size)),
        pack_dir: args.residual_pack_dir.clone(),
        inflight,
        inflight_limit,
        prewarm_on_l2: args.prewarm_on_l2,
    };

    let metrics = state.metrics.clone();
    let buffer_pool = state.buffer_pool.clone();
    let inflight = state.inflight.clone();
    if args.metrics_interval_secs > 0 {
        let interval = Duration::from_secs(args.metrics_interval_secs);
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(interval).await;
                let snapshot = metrics.lock().unwrap().take();
                if snapshot.tile_total == 0 && snapshot.family_generated == 0 {
                    continue;
                }
                let tile_avg = if snapshot.tile_total > 0 {
                    snapshot.tile_ms_sum / snapshot.tile_total as u128
                } else {
                    0
                };
                let fam_avg = if snapshot.family_generated > 0 {
                    snapshot.family_ms_sum / snapshot.family_generated as u128
                } else {
                    0
                };
                let (pool_total, pool_avail, pool_in_use_max) = buffer_pool.stats();
                let inflight_current = inflight.current();
                let inflight_max = inflight.take_max();
                let mut sys = System::new();
                let (rss_kb, cpu_pct) = if let Ok(pid) = sysinfo::get_current_pid() {
                    sys.refresh_process(pid);
                    sys.process(pid)
                        .map(|p| (p.memory(), p.cpu_usage()))
                        .unwrap_or((0, 0.0))
                } else {
                    (0, 0.0)
                };
                let rss_mb = rss_kb / 1024;
                info!(
                    "metrics tiles_total={} baseline={} cache_hit={} generated={} fallback={} residual_view={} tile_avg_ms={} tile_max_ms={} families={} family_avg_ms={} family_max_ms={} pool_total={} pool_avail={} pool_in_use_max={} inflight_current={} inflight_max={} rss_kb={} rss_mb={} cpu_pct={:.1}",
                    snapshot.tile_total,
                    snapshot.tile_baseline,
                    snapshot.tile_cache_hit,
                    snapshot.tile_generated,
                    snapshot.tile_fallback,
                    snapshot.tile_residual_view,
                    tile_avg,
                    snapshot.tile_ms_max,
                    snapshot.family_generated,
                    fam_avg,
                    snapshot.family_ms_max,
                    pool_total,
                    pool_avail,
                    pool_in_use_max,
                    inflight_current,
                    inflight_max,
                    rss_kb,
                    rss_mb,
                    cpu_pct
                );
            }
        });
    }

    let app = Router::new()
        .route("/healthz", get(healthz))
        .route("/dzi/:slide_id.dzi", get(get_dzi))
        .route("/dzi/:slide_id_files/:level/:tile", get(get_tile_dzi))
        .route("/tiles/:slide_id/:level/:tile", get(get_tile))
        .route(
            "/tiles/:slide_id/residual/:level/:tile",
            get(get_residual_tile),
        )
        .route("/viewer/:slide_id", get(viewer))
        .with_state(state)
        .layer(TraceLayer::new_for_http());

    let addr = format!("0.0.0.0:{}", args.port);
    info!(
        "listening on http://{} (rayon_threads={}, tokio_workers={}, tokio_blocking_threads={}, hw_threads={})",
        addr,
        rayon::current_num_threads(),
        args.tokio_workers,
        args.tokio_blocking_threads,
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0)
    );
    for slide in slides.values() {
        info!("viewer url: http://{}/viewer/{}", addr, slide.slide_id);
    }
    info!("dzi url: http://{}/dzi/<slide_id>.dzi", addr);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn healthz() -> impl IntoResponse {
    "ok"
}

async fn get_dzi(
    State(state): State<AppState>,
    AxumPath(slide_id_raw): AxumPath<String>,
    Query(query): Query<ModeQuery>,
) -> Result<Response, StatusCode> {
    let slide_id = slide_id_raw.trim_end_matches(".dzi");
    let slide = state
        .slides
        .get(slide_id)
        .ok_or(StatusCode::NOT_FOUND)?;
    let mut body = fs::read_to_string(&slide.dzi_path).map_err(|_| StatusCode::NOT_FOUND)?;
    if let Some(mode) = query.mode.as_deref() {
        if mode == "residual" {
            let needle = "Url=\"";
            if let Some(idx) = body.find(needle) {
                let start = idx + needle.len();
                if let Some(end) = body[start..].find('"') {
                    let end = start + end;
                    let replacement = format!("/tiles/{}/residual/", slide_id);
                    body.replace_range(start..end, &replacement);
                }
            }
        }
    }
    let mut resp = Response::new(body.into());
    resp.headers_mut()
        .insert(header::CONTENT_TYPE, HeaderValue::from_static("image/xml"));
    Ok(resp)
}

async fn get_tile(
    State(state): State<AppState>,
    AxumPath((slide_id, level, tile)): AxumPath<(String, u32, String)>,
) -> Result<Response, StatusCode> {
    serve_tile(state, slide_id, level, tile).await
}

async fn get_tile_dzi(
    State(state): State<AppState>,
    AxumPath((slide_id_files, level, tile)): AxumPath<(String, u32, String)>,
) -> Result<Response, StatusCode> {
    let slide_id = slide_id_files.trim_end_matches("_files").to_string();
    serve_tile(state, slide_id, level, tile).await
}

async fn get_residual_tile(
    State(state): State<AppState>,
    AxumPath((slide_id, level, tile)): AxumPath<(String, u32, String)>,
) -> Result<Response, StatusCode> {
    let (x, y) = parse_tile_name(&tile).ok_or(StatusCode::BAD_REQUEST)?;
    let slide = state
        .slides
        .get(&slide_id)
        .ok_or(StatusCode::NOT_FOUND)?;

    if level <= slide.l2 {
        let bytes =
            fs::read(baseline_tile_path(slide, level, x, y)).map_err(|_| StatusCode::NOT_FOUND)?;
        state
            .metrics
            .lock()
            .unwrap()
            .record_tile("residual_view", 0);
        info!(
            "tile residual_view baseline slide_id={} level={} x={} y={}",
            slide_id, level, x, y
        );
        return Ok(jpeg_response(bytes));
    }

    let (x2, y2) = if level == slide.l1 {
        (x >> 1, y >> 1)
    } else if level == slide.l0 {
        (x >> 2, y >> 2)
    } else {
        return Err(StatusCode::BAD_REQUEST);
    };
    let residual_path = residual_tile_path(slide, level, x2, y2, x, y);
    let bytes = fs::read(&residual_path).map_err(|_| StatusCode::NOT_FOUND)?;
    state
        .metrics
        .lock()
        .unwrap()
        .record_tile("residual_view", 0);
    info!(
        "tile residual_view residual slide_id={} level={} x={} y={} path={}",
        slide_id,
        level,
        x,
        y,
        residual_path.display()
    );
    Ok(jpeg_response(bytes))
}

async fn serve_tile(
    state: AppState,
    slide_id: String,
    level: u32,
    tile: String,
) -> Result<Response, StatusCode> {
    let start = Instant::now();
    let (x, y) = parse_tile_name(&tile).ok_or(StatusCode::BAD_REQUEST)?;
    let slide = state
        .slides
        .get(&slide_id)
        .ok_or(StatusCode::NOT_FOUND)?;

    if level <= slide.l2 {
        let bytes =
            fs::read(baseline_tile_path(slide, level, x, y)).map_err(|_| StatusCode::NOT_FOUND)?;
        if level == slide.l2 && state.prewarm_on_l2 {
            spawn_family_prewarm(state.clone(), slide.clone(), x, y);
        }
        state
            .metrics
            .lock()
            .unwrap()
            .record_tile("baseline", start.elapsed().as_millis());
        info!(
            "tile baseline slide_id={} level={} x={} y={} ms={}",
            slide_id,
            level,
            x,
            y,
            start.elapsed().as_millis()
        );
        return Ok(jpeg_response(bytes));
    }

    let key = TileKey {
        slide_id: slide_id.clone(),
        level,
        x,
        y,
    };
    if let Some(bytes) = state.cache.get(&key) {
        state
            .metrics
            .lock()
            .unwrap()
            .record_tile("cache_hit", start.elapsed().as_millis());
        info!(
            "tile cache_hit slide_id={} level={} x={} y={} ms={}",
            slide_id,
            level,
            x,
            y,
            start.elapsed().as_millis()
        );
        return Ok(jpeg_response(bytes.to_vec()));
    }

    let slide = slide.clone();
    let cache = state.cache.clone();
    let quality = state.tile_quality;
    let timing = state.timing_breakdown;
    let writer = state.writer.clone();
    let write_root = state.write_generated_dir.clone();
    let buffer_pool = state.buffer_pool.clone();
    let pack_dir = state.pack_dir.clone();
    let inflight = state.inflight.clone();
    let inflight_limit = state.inflight_limit.clone();
    let permit = inflight_limit.acquire_owned().await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let bytes = task::spawn_blocking(move || {
        let _permit = permit;
        let _inflight = inflight.enter();
        let gen_start = Instant::now();
        let (x2, y2) = if level == slide.l1 {
            (x >> 1, y >> 1)
        } else if level == slide.l0 {
            (x >> 2, y >> 2)
        } else {
            return Err(anyhow!("unsupported level {}", level));
        };
        let result = generate_family(
            &slide,
            x2,
            y2,
            quality,
            timing,
            &writer,
            &write_root,
            &buffer_pool,
            pack_dir.as_deref(),
        );
        match result {
            Ok(result) => {
                let family = result.tiles;
                for (k, v) in family.iter() {
                    cache.insert(k.clone(), v.clone());
                }
                let family_ms = gen_start.elapsed().as_millis();
                if let Some(stats) = result.stats {
                    info!(
                        "family_breakdown [PARALLEL_CHROMA] slide_id={} x2={} y2={} l2_decode_ms={} parallel_chroma_ms={} l1_residual_ms={} l1_encode_ms={} l0_resize_ms={} l0_residual_ms={} l0_encode_ms={} total_ms={} l1_residuals={} l0_residuals={} l1_parallel_max={} l0_parallel_max={}",
                        slide.slide_id,
                        x2,
                        y2,
                        stats.l2_decode_ms,
                        stats.l1_resize_ms,
                        stats.l1_residual_ms,
                        stats.l1_encode_ms,
                        stats.l0_resize_ms,
                        stats.l0_residual_ms,
                        stats.l0_encode_ms,
                        stats.total_ms,
                        stats.residuals_l1,
                        stats.residuals_l0,
                        stats.l1_parallel_max,
                        stats.l0_parallel_max
                    );
                }
                info!(
                    "family_generated slide_id={} x2={} y2={} tiles={} ms={}",
                    slide.slide_id,
                    x2,
                    y2,
                    family.len(),
                    family_ms
                );
                let tile = family
                    .get(&TileKey {
                        slide_id: slide.slide_id.clone(),
                        level,
                        x,
                        y,
                    })
                    .cloned()
                    .ok_or_else(|| anyhow!("tile not generated"))?;
                Ok((tile, Some(family_ms), false))
            }
            Err(err) => {
                info!(
                    "family_error slide_id={} x2={} y2={} err={}",
                    slide.slide_id, x2, y2, err
                );
                let bytes = fs::read(baseline_tile_path(&slide, level, x, y))?;
                Ok((Bytes::from(bytes), None, true))
            }
        }
    })
    .await
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    if let Some(family_ms) = bytes.1 {
        state.metrics.lock().unwrap().record_family(family_ms);
    }
    if bytes.2 {
        state
            .metrics
            .lock()
            .unwrap()
            .record_tile("fallback", start.elapsed().as_millis());
    } else {
        state
            .metrics
            .lock()
            .unwrap()
            .record_tile("generated", start.elapsed().as_millis());
    }
    info!(
        "tile generated slide_id={} level={} x={} y={} ms={}",
        slide_id,
        level,
        x,
        y,
        start.elapsed().as_millis()
    );
    Ok(jpeg_response(bytes.0.to_vec()))
}

fn spawn_family_prewarm(state: AppState, slide: Slide, x2: u32, y2: u32) {
    let cache = state.cache.clone();
    let quality = state.tile_quality;
    let writer = state.writer.clone();
    let write_root = state.write_generated_dir.clone();
    let buffer_pool = state.buffer_pool.clone();
    let pack_dir = state.pack_dir.clone();
    let inflight = state.inflight.clone();
    let inflight_limit = state.inflight_limit.clone();
    tokio::spawn(async move {
        let permit = match inflight_limit.acquire_owned().await {
            Ok(p) => p,
            Err(_) => return,
        };
        let _inflight = inflight.enter();
        let _permit = permit;
        let _ = task::spawn_blocking(move || {
            let result = generate_family(
                &slide,
                x2,
                y2,
                quality,
                false,
                &writer,
                &write_root,
                &buffer_pool,
                pack_dir.as_deref(),
            );
            if let Ok(result) = result {
                for (k, v) in result.tiles.iter() {
                    cache.insert(k.clone(), v.clone());
                }
            }
        })
        .await;
    });
}

async fn viewer(
    State(state): State<AppState>,
    AxumPath(slide_id): AxumPath<String>,
) -> Result<Html<String>, StatusCode> {
    let slide = state
        .slides
        .get(&slide_id)
        .ok_or(StatusCode::NOT_FOUND)?;
    let html = format!(
        r##"<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>WSI Viewer - {slide_id}</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.0/openseadragon.min.js" crossorigin="anonymous" referrerpolicy="no-referrer"></script>
    <style>
      html, body {{ height: 100%; margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif; }}
      #osd {{ width: 100%; height: 100%; }}
    </style>
  </head>
  <body>
    <div id="osd"></div>
    <script>
      OpenSeadragon({{
        id: "osd",
        prefixUrl: "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.0/images/",
        showNavigator: true,
        tileSources: "/dzi/{slide_id}.dzi"
      }});
    </script>
  </body>
</html>"##
    );
    info!(
        "viewer requested slide_id={} tile_size={}",
        slide_id, slide.tile_size
    );
    Ok(Html(html))
}

fn parse_tile_name(name: &str) -> Option<(u32, u32)> {
    let trimmed = name.strip_suffix(".jpg").unwrap_or(name);
    let mut parts = trimmed.split('_');
    let x = parts.next()?.parse().ok()?;
    let y = parts.next()?.parse().ok()?;
    Some((x, y))
}

fn jpeg_response(bytes: Vec<u8>) -> Response {
    let mut resp = Response::new(bytes.into());
    resp.headers_mut()
        .insert(header::CONTENT_TYPE, HeaderValue::from_static("image/jpeg"));
    resp
}

fn baseline_tile_path(slide: &Slide, level: u32, x: u32, y: u32) -> PathBuf {
    slide
        .files_dir
        .join(level.to_string())
        .join(format!("{}_{}.jpg", x, y))
}

fn residual_tile_path(slide: &Slide, level: u32, x2: u32, y2: u32, x: u32, y: u32) -> PathBuf {
    let subdir = if level == slide.l1 { "L1" } else { "L0" };
    slide
        .residuals_dir
        .join(subdir)
        .join(format!("{}_{}", x2, y2))
        .join(format!("{}_{}.jpg", x, y))
}

fn copy_tile_into_mosaic(
    tile: &[u8],
    mosaic: &mut [u8],
    mosaic_width: u32,
    tile_size: u32,
    dx: u32,
    dy: u32,
) {
    let tile_stride = (tile_size * 3) as usize;
    let mosaic_stride = (mosaic_width * 3) as usize;
    let base_x = (dx * tile_size * 3) as usize;
    let base_y = (dy * tile_size) as usize;
    for y in 0..tile_size as usize {
        let tile_off = y * tile_stride;
        let mosaic_off = (base_y + y) * mosaic_stride + base_x;
        let dst = &mut mosaic[mosaic_off..mosaic_off + tile_stride];
        let src = &tile[tile_off..tile_off + tile_stride];
        dst.copy_from_slice(src);
    }
}

fn enqueue_generated(
    writer: &Option<mpsc::Sender<WriteJob>>,
    write_root: &Option<PathBuf>,
    tiles: &HashMap<TileKey, Bytes>,
) {
    let Some(writer) = writer else { return; };
    let Some(root) = write_root else { return; };
    for (key, bytes) in tiles.iter() {
        let mut path = root.clone();
        path.push(&key.slide_id);
        path.push(key.level.to_string());
        path.push(format!("{}_{}.jpg", key.x, key.y));
        let job = WriteJob {
            path,
            bytes: bytes.clone(),
        };
        if writer.try_send(job).is_err() {
            info!("write queue full, dropping tile {}", key_to_string(key));
        }
    }
}

fn key_to_string(key: &TileKey) -> String {
    format!("{}:{}:{}:{}", key.slide_id, key.level, key.x, key.y)
}

struct PackFile {
    _mmap: Mmap,
    data_offset: usize,
    index: Vec<PackIndexEntry>,
}

impl PackFile {
    fn get_residual(&self, level_kind: u8, idx_in_parent: u8) -> Option<&[u8]> {
        let entry = self
            .index
            .iter()
            .find(|e| e.level_kind == level_kind && e.idx_in_parent == idx_in_parent)?;
        let start = self.data_offset + entry.offset as usize;
        let end = start + entry.length as usize;
        Some(&self._mmap[start..end])
    }
}

fn open_pack(pack_dir: &Path, x2: u32, y2: u32) -> Result<PackFile> {
    let path = pack_dir.join(format!("{}_{}.pack", x2, y2));
    let file = fs::File::open(&path)
        .with_context(|| format!("open pack {}", path.display()))?;
    let mmap = unsafe { Mmap::map(&file)? };
    if mmap.len() < 24 {
        return Err(anyhow!("pack too small"));
    }
    if &mmap[0..4] != b"RRIP" {
        return Err(anyhow!("pack magic mismatch"));
    }
    let version = u16::from_le_bytes([mmap[4], mmap[5]]);
    if version != 1 {
        return Err(anyhow!("pack version mismatch"));
    }
    let count = u32::from_le_bytes([mmap[8], mmap[9], mmap[10], mmap[11]]) as usize;
    let index_offset = u32::from_le_bytes([mmap[12], mmap[13], mmap[14], mmap[15]]) as usize;
    let data_offset = u32::from_le_bytes([mmap[16], mmap[17], mmap[18], mmap[19]]) as usize;
    let mut index = Vec::with_capacity(count);
    let mut cursor = index_offset;
    for _ in 0..count {
        let level_kind = mmap[cursor];
        let idx_in_parent = mmap[cursor + 1];
        let offset = u32::from_le_bytes([
            mmap[cursor + 4],
            mmap[cursor + 5],
            mmap[cursor + 6],
            mmap[cursor + 7],
        ]);
        let length = u32::from_le_bytes([
            mmap[cursor + 8],
            mmap[cursor + 9],
            mmap[cursor + 10],
            mmap[cursor + 11],
        ]);
        index.push(PackIndexEntry {
            level_kind,
            idx_in_parent,
            offset,
            length,
        });
        cursor += 16;
    }
    Ok(PackFile {
        _mmap: mmap,
        data_offset,
        index,
    })
}

fn decode_luma_from_bytes(bytes: &[u8]) -> Result<Vec<u8>> {
    let (pixels, _width, _height) = decode_luma_turbo(bytes)?;
    Ok(pixels)
}

fn discover_slides(root: &Path, residuals_dir: &str) -> Result<HashMap<String, Slide>> {
    let mut slides = HashMap::new();
    for entry in fs::read_dir(root).with_context(|| format!("reading {}", root.display()))? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let slide_id = entry.file_name().to_string_lossy().to_string();
        let slide_root = entry.path();
        let dzi_path = slide_root.join("baseline_pyramid.dzi");
        let files_dir = slide_root.join("baseline_pyramid_files");
        if !dzi_path.exists() || !files_dir.exists() {
            continue;
        }
        let tile_size = parse_tile_size(&dzi_path).unwrap_or(256);
        let max_level = max_level_from_files(&files_dir)?;
        let l0 = max_level;
        let l1 = max_level.saturating_sub(1);
        let l2 = max_level.saturating_sub(2);
        let slide = Slide {
            slide_id: slide_id.clone(),
            dzi_path,
            files_dir,
            residuals_dir: slide_root.join(residuals_dir),
            tile_size,
            max_level,
            l0,
            l1,
            l2,
        };
        slides.insert(slide_id, slide);
    }
    Ok(slides)
}

fn parse_tile_size(dzi_path: &Path) -> Option<u32> {
    let contents = fs::read_to_string(dzi_path).ok()?;
    let key = "TileSize=\"";
    let start = contents.find(key)? + key.len();
    let end = contents[start..].find('"')? + start;
    contents[start..end].parse().ok()
}

fn max_level_from_files(files_dir: &Path) -> Result<u32> {
    let mut max_level: Option<u32> = None;
    for entry in fs::read_dir(files_dir)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().to_string();
        if let Ok(level) = name.parse::<u32>() {
            max_level = Some(max_level.map_or(level, |m| m.max(level)));
        }
    }
    max_level.ok_or_else(|| anyhow!("no level directories in {}", files_dir.display()))
}

fn generate_family(
    slide: &Slide,
    x2: u32,
    y2: u32,
    quality: u8,
    timing: bool,
    writer: &Option<mpsc::Sender<WriteJob>>,
    write_root: &Option<PathBuf>,
    buffer_pool: &BufferPool,
    pack_dir: Option<&Path>,
) -> Result<FamilyResult> {
    let total_start = Instant::now();
    let l2_path = baseline_tile_path(slide, slide.l2, x2, y2);
    let l2_start = Instant::now();
    let l2_img = load_rgb(&l2_path)
        .with_context(|| format!("loading baseline L2 tile {}", l2_path.display()))?;
    let l2_decode_ms = if timing { l2_start.elapsed().as_millis() } else { 0 };

    let tile_size = slide.tile_size;

    // OPTIMIZATION: Fast YCbCr upsampling instead of RGB resize
    let parallel_chroma_start = Instant::now();

    // Convert L2 to YCbCr once
    let (l2_y, l2_cb, l2_cr) = ycbcr_planes(&l2_img);
    let l2_width = l2_img.width();
    let l2_height = l2_img.height();

    // Compute L1 and L0 chroma upsampling in parallel
    let ((l1_y, l1_cb, l1_cr), (l0_cb, l0_cr)) = rayon::join(
        || {
            // L2 → L1 upsampling (2x)
            let l1_y = upsample_2x_channel(&l2_y, l2_width as usize, l2_height as usize);
            let l1_cb = upsample_2x_nearest(&l2_cb, l2_width as usize, l2_height as usize);
            let l1_cr = upsample_2x_nearest(&l2_cr, l2_width as usize, l2_height as usize);
            (l1_y, l1_cb, l1_cr)
        },
        || {
            // L2 → L0 chroma direct (4x for chroma only, using nearest neighbor)
            let l0_cb = upsample_4x_nearest(&l2_cb, l2_width as usize, l2_height as usize);
            let l0_cr = upsample_4x_nearest(&l2_cr, l2_width as usize, l2_height as usize);
            (l0_cb, l0_cr)
        }
    );

    let parallel_chroma_ms = if timing { parallel_chroma_start.elapsed().as_millis() } else { 0 };
    let mut out = HashMap::new();
    let tile_buf_len = (tile_size * tile_size * 3) as usize;
    let pack = if let Some(pack_root) = pack_dir {
        open_pack(pack_root, x2, y2).ok()
    } else {
        None
    };
    let mut l1_tile_bufs: Vec<Vec<u8>> = (0..4).map(|_| buffer_pool.get(tile_buf_len)).collect();
    let l1_parallel = ParallelStats::new();
    let l1_results: Result<Vec<_>> = l1_tile_bufs
        .par_iter_mut()
        .enumerate()
        .map(|(idx, buf)| {
            let _guard = l1_parallel.enter();
            let dx = (idx % 2) as u32;
            let dy = (idx / 2) as u32;
            let x1 = x2 * 2 + dx;
            let y1 = y2 * 2 + dy;
            let residual_path = residual_tile_path(slide, slide.l1, x2, y2, x1, y1);
            let res_start = Instant::now();
            let used_residual = if let Some(ref pack) = pack {
                if let Some(bytes) = pack.get_residual(1, (dy * 2 + dx) as u8) {
                    let residual = decode_luma_from_bytes(bytes)?;
                    apply_residual_into(
                        &l1_y,
                        &l1_cb,
                        &l1_cr,
                        tile_size * 2,
                        tile_size * 2,
                        dx * tile_size,
                        dy * tile_size,
                        tile_size,
                        &residual,
                        buf,
                    )?;
                    true
                } else if residual_path.exists() {
                    let residual = load_luma(&residual_path)?;
                    apply_residual_into(
                        &l1_y,
                        &l1_cb,
                        &l1_cr,
                        tile_size * 2,
                        tile_size * 2,
                        dx * tile_size,
                        dy * tile_size,
                        tile_size,
                        &residual,
                        buf,
                    )?;
                    true
                } else {
                    let base = load_rgb(&baseline_tile_path(slide, slide.l1, x1, y1))?;
                    buf.copy_from_slice(base.as_raw());
                    false
                }
            } else if residual_path.exists() {
                let residual = load_luma(&residual_path)?;
                apply_residual_into(
                    &l1_y,
                    &l1_cb,
                    &l1_cr,
                    tile_size * 2,
                    tile_size * 2,
                    dx * tile_size,
                    dy * tile_size,
                    tile_size,
                    &residual,
                    buf,
                )?;
                true
            } else {
                let base = load_rgb(&baseline_tile_path(slide, slide.l1, x1, y1))?;
                buf.copy_from_slice(base.as_raw());
                false
            };
            let res_ms = if timing { res_start.elapsed().as_millis() } else { 0 };
            let enc_start = Instant::now();
            let bytes = encode_jpeg_from_rgb_bytes(buf, tile_size, tile_size, quality)?;
            let enc_ms = if timing { enc_start.elapsed().as_millis() } else { 0 };
            Ok((idx, x1, y1, Bytes::from(bytes), used_residual, res_ms, enc_ms))
        })
        .collect();

    let l1_results = l1_results?;
    let l1_parallel_max = l1_parallel.take_max();
    let mut residuals_l1 = 0usize;
    let mut residuals_l0 = 0usize;
    let mut l1_residual_ms = 0u128;
    let mut l1_encode_ms = 0u128;
    let mut l0_residual_ms = 0u128;
    let mut l0_encode_ms = 0u128;

    for (_idx, x1, y1, bytes, used_residual, res_ms, enc_ms) in l1_results {
        if used_residual {
            residuals_l1 += 1;
        }
        l1_residual_ms += res_ms;
        l1_encode_ms += enc_ms;
        out.insert(
            TileKey {
                slide_id: slide.slide_id.clone(),
                level: slide.l1,
                x: x1,
                y: y1,
            },
            bytes,
        );
    }

    // Build L1 YCbCr mosaic from reconstructed tiles
    let l1_mosaic_w = tile_size * 2;
    let l1_mosaic_size = (l1_mosaic_w * l1_mosaic_w) as usize;
    let mut l1_mosaic_y = vec![0u8; l1_mosaic_size];
    let mut l1_mosaic_cb = vec![0u8; l1_mosaic_size];
    let mut l1_mosaic_cr = vec![0u8; l1_mosaic_size];

    // Convert RGB tiles to YCbCr and copy into mosaic
    for idx in 0..4 {
        let dx = (idx % 2) as u32;
        let dy = (idx / 2) as u32;
        let tile_offset = ((dy * tile_size * l1_mosaic_w) + (dx * tile_size)) as usize;

        // Extract YCbCr from the RGB tile buffer
        for y in 0..tile_size {
            for x in 0..tile_size {
                let src_idx = ((y * tile_size + x) * 3) as usize;
                let dst_idx = tile_offset + (y * l1_mosaic_w + x) as usize;
                let r = l1_tile_bufs[idx][src_idx];
                let g = l1_tile_bufs[idx][src_idx + 1];
                let b = l1_tile_bufs[idx][src_idx + 2];
                let (yy, cb, cr) = rgb_to_ycbcr(r, g, b);
                l1_mosaic_y[dst_idx] = yy;
                l1_mosaic_cb[dst_idx] = cb;
                l1_mosaic_cr[dst_idx] = cr;
            }
        }
    }

    for buf in l1_tile_bufs.drain(..) {
        buffer_pool.put(buf);
    }

    // Fast YCbCr upsampling for L0 prediction (2x from L1 mosaic)
    let l0_resize_start = Instant::now();
    let l0_y = upsample_2x_channel(&l1_mosaic_y, l1_mosaic_w as usize, l1_mosaic_w as usize);
    let l0_resize_ms = if timing { l0_resize_start.elapsed().as_millis() } else { 0 };
    // Use the pre-computed L0 chroma from parallel processing
    let l0_parallel = ParallelStats::new();
    let l0_results: Result<Vec<_>> = (0..16)
        .into_par_iter()
        .map(|idx| {
            let _guard = l0_parallel.enter();
            let dx = (idx % 4) as u32;
            let dy = (idx / 4) as u32;
            let x0 = x2 * 4 + dx;
            let y0 = y2 * 4 + dy;
            let residual_path = residual_tile_path(slide, slide.l0, x2, y2, x0, y0);
            let res_start = Instant::now();
            let mut buf = buffer_pool.get(tile_buf_len);
            let (used_residual, bytes, enc_ms) = {
                let used_residual = if let Some(ref pack) = pack {
                    if let Some(bytes) = pack.get_residual(0, (dy * 4 + dx) as u8) {
                        let residual = decode_luma_from_bytes(bytes)?;
                        apply_residual_into(
                            &l0_y,
                            &l0_cb,
                            &l0_cr,
                            tile_size * 4,
                            tile_size * 4,
                            dx * tile_size,
                            dy * tile_size,
                            tile_size,
                            &residual,
                            &mut buf,
                        )?;
                        true
                    } else if residual_path.exists() {
                        let residual = load_luma(&residual_path)?;
                        apply_residual_into(
                            &l0_y,
                            &l0_cb,
                            &l0_cr,
                            tile_size * 4,
                            tile_size * 4,
                            dx * tile_size,
                            dy * tile_size,
                            tile_size,
                            &residual,
                            &mut buf,
                        )?;
                        true
                    } else {
                        let base = load_rgb(&baseline_tile_path(slide, slide.l0, x0, y0))?;
                        buf.copy_from_slice(base.as_raw());
                        false
                    }
                } else if residual_path.exists() {
                    let residual = load_luma(&residual_path)?;
                    apply_residual_into(
                        &l0_y,
                        &l0_cb,
                        &l0_cr,
                        tile_size * 4,
                        tile_size * 4,
                        dx * tile_size,
                        dy * tile_size,
                        tile_size,
                        &residual,
                        &mut buf,
                    )?;
                    true
                } else {
                    let base = load_rgb(&baseline_tile_path(slide, slide.l0, x0, y0))?;
                    buf.copy_from_slice(base.as_raw());
                    false
                };
                let enc_start = Instant::now();
                let bytes = encode_jpeg_from_rgb_bytes(&buf, tile_size, tile_size, quality)?;
                let enc_ms = if timing { enc_start.elapsed().as_millis() } else { 0 };
                (used_residual, bytes, enc_ms)
            };
            buffer_pool.put(buf);
            let res_ms = if timing { res_start.elapsed().as_millis() } else { 0 };
            Ok((x0, y0, Bytes::from(bytes), used_residual, res_ms, enc_ms))
        })
        .collect();

    let l0_results = l0_results?;
    let l0_parallel_max = l0_parallel.take_max();
    for (x0, y0, bytes, used_residual, res_ms, enc_ms) in l0_results {
        if used_residual {
            residuals_l0 += 1;
        }
        l0_residual_ms += res_ms;
        l0_encode_ms += enc_ms;
        out.insert(
            TileKey {
                slide_id: slide.slide_id.clone(),
                level: slide.l0,
                x: x0,
                y: y0,
            },
            bytes,
        );
    }

    info!(
        "family_residuals slide_id={} x2={} y2={} l1={} l0={}",
        slide.slide_id, x2, y2, residuals_l1, residuals_l0
    );
    enqueue_generated(writer, write_root, &out);
    let stats = if timing {
        Some(FamilyStats {
            l2_decode_ms,
            l1_resize_ms: parallel_chroma_ms, // Now includes parallel chroma
            l1_residual_ms,
            l1_encode_ms,
            l0_resize_ms,
            l0_residual_ms,
            l0_encode_ms,
            total_ms: total_start.elapsed().as_millis(),
            residuals_l1,
            residuals_l0,
            l1_parallel_max,
            l0_parallel_max,
        })
    } else {
        None
    };
    Ok(FamilyResult { tiles: out, stats })
}

fn load_rgb(path: &Path) -> Result<RgbImage> {
    // Use TurboJPEG for 3-5x faster JPEG decoding
    let (pixels, width, height) = load_rgb_turbo(path)?;
    RgbImage::from_raw(width, height, pixels)
        .ok_or_else(|| anyhow!("failed to create RGB image from pixels"))
}

fn load_luma(path: &Path) -> Result<Vec<u8>> {
    let (pixels, _width, _height) = load_luma_turbo(path)?;
    Ok(pixels)
}

fn resize_rgb(img: &RgbImage, width: u32, height: u32) -> RgbImage {
    DynamicImage::ImageRgb8(img.clone())
        .resize_exact(width, height, FilterType::Triangle)
        .to_rgb8()
}

fn encode_jpeg_from_rgb_bytes(
    bytes: &[u8],
    width: u32,
    height: u32,
    quality: u8,
) -> Result<Vec<u8>> {
    // Use TurboJPEG for much faster encoding
    encode_jpeg_turbo(bytes, width, height, quality)
}

fn apply_residual_into(
    y_plane: &[u8],
    cb_plane: &[u8],
    cr_plane: &[u8],
    width: u32,
    height: u32,
    x0: u32,
    y0: u32,
    tile_size: u32,
    residual: &[u8],
    out: &mut [u8],
) -> Result<()> {
    let expected_len = (tile_size * tile_size * 3) as usize;
    if out.len() != expected_len {
        return Err(anyhow!("output buffer size mismatch"));
    }

    // Optimized with chunking for better auto-vectorization
    // Process 8 pixels at a time when possible
    for y in 0..tile_size {
        let py = y0 + y;
        if py >= height {
            continue;
        }

        let row_offset = (y * tile_size) as usize;
        let plane_row_offset = (py * width) as usize;

        // Process main chunks of 8 pixels
        let mut x = 0;
        while x + 8 <= tile_size {
            let px = x0 + x;
            if px + 8 <= width {
                // Process 8 pixels at once for better SIMD utilization
                for dx in 0..8 {
                    let x_pos = x + dx;
                    let px_pos = px + dx;
                    let idx = plane_row_offset + px_pos as usize;
                    let residual_idx = row_offset + x_pos as usize;

                    // Apply residual with optimized arithmetic
                    let y_pred = y_plane[idx] as i16;
                    let res = residual[residual_idx] as i16 - 128;
                    let y_recon = ((y_pred + res).max(0).min(255)) as u8;

                    // Get chroma values
                    let cb_pred = cb_plane[idx];
                    let cr_pred = cr_plane[idx];

                    // Convert to RGB using optimized function
                    let (r_out, g_out, b_out) = ycbcr_to_rgb(y_recon, cb_pred, cr_pred);

                    // Write output
                    let out_idx = (residual_idx * 3) as usize;
                    out[out_idx] = r_out;
                    out[out_idx + 1] = g_out;
                    out[out_idx + 2] = b_out;
                }
                x += 8;
            } else {
                // Handle remaining pixels
                let px_pos = px;
                if px_pos < width {
                    let idx = plane_row_offset + px_pos as usize;
                    let residual_idx = row_offset + x as usize;

                    let y_pred = y_plane[idx] as i16;
                    let res = residual[residual_idx] as i16 - 128;
                    let y_recon = ((y_pred + res).max(0).min(255)) as u8;

                    let cb_pred = cb_plane[idx];
                    let cr_pred = cr_plane[idx];

                    let (r_out, g_out, b_out) = ycbcr_to_rgb(y_recon, cb_pred, cr_pred);
                    let out_idx = (residual_idx * 3) as usize;
                    out[out_idx] = r_out;
                    out[out_idx + 1] = g_out;
                    out[out_idx + 2] = b_out;
                }
                x += 1;
            }
        }

        // Process remaining pixels
        while x < tile_size {
            let px = x0 + x;
            if px < width {
                let idx = plane_row_offset + px as usize;
                let residual_idx = row_offset + x as usize;

                let y_pred = y_plane[idx] as i16;
                let res = residual[residual_idx] as i16 - 128;
                let y_recon = ((y_pred + res).max(0).min(255)) as u8;

                let cb_pred = cb_plane[idx];
                let cr_pred = cr_plane[idx];

                let (r_out, g_out, b_out) = ycbcr_to_rgb(y_recon, cb_pred, cr_pred);
                let out_idx = (residual_idx * 3) as usize;
                out[out_idx] = r_out;
                out[out_idx + 1] = g_out;
                out[out_idx + 2] = b_out;
            }
            x += 1;
        }
    }
    Ok(())
}

fn ycbcr_planes(img: &RgbImage) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let w = img.width();
    let h = img.height();
    let mut y = vec![0u8; (w * h) as usize];
    let mut cb = vec![0u8; (w * h) as usize];
    let mut cr = vec![0u8; (w * h) as usize];
    for yy in 0..h {
        for xx in 0..w {
            let idx = (yy * w + xx) as usize;
            let p = img.get_pixel(xx, yy).0;
            let (yyc, cbc, crc) = rgb_to_ycbcr(p[0], p[1], p[2]);
            y[idx] = yyc;
            cb[idx] = cbc;
            cr[idx] = crc;
        }
    }
    (y, cb, cr)
}

fn rgb_to_ycbcr(r: u8, g: u8, b: u8) -> (u8, u8, u8) {
    // Fixed-point conversion (10-bit precision) - much faster than floats
    let r = r as i32;
    let g = g as i32;
    let b = b as i32;

    // ITU-R BT.601 coefficients scaled by 1024
    let y = ((306 * r + 601 * g + 117 * b) >> 10).clamp(0, 255) as u8;
    let cb = (((-173 * r - 339 * g + 512 * b) >> 10) + 128).clamp(0, 255) as u8;
    let cr = (((512 * r - 429 * g - 83 * b) >> 10) + 128).clamp(0, 255) as u8;

    (y, cb, cr)
}

fn ycbcr_to_rgb(y: u8, cb: u8, cr: u8) -> (u8, u8, u8) {
    // Fixed-point conversion - 20-30% faster than floating point
    let y = y as i32;
    let cb = cb as i32 - 128;
    let cr = cr as i32 - 128;

    // ITU-R BT.601 coefficients scaled by 1024
    let r = (y + ((cr * 1436) >> 10)).clamp(0, 255) as u8;
    let g = (y - ((cb * 352 + cr * 731) >> 10)).clamp(0, 255) as u8;
    let b = (y + ((cb * 1815) >> 10)).clamp(0, 255) as u8;

    (r, g, b)
}
