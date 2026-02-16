use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use axum::extract::{Path as AxumPath, Query, State};
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::get;
use axum::Router;
use bytes::Bytes;
use clap::Args;
use sysinfo::System;
use tokio::sync::{mpsc, Semaphore};
use tokio::task;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use tracing::info;

use crate::core::pack::{open_bundle, BundleFile};
use crate::core::pyramid::{parse_tile_size, max_level_from_files};
use crate::core::reconstruct::{
    BufferPool, OutputFormat, ReconstructInput, ReconstructOpts, reconstruct_family,
};

#[derive(Args, Debug)]
pub struct ServeArgs {
    #[arg(long, default_value = "data")]
    slides_root: PathBuf,
    #[arg(long, default_value = "residuals_q32")]
    residuals_dir: String,
    #[arg(long, default_value = "residual_packs")]
    pack_dir: String,
    #[arg(long, default_value_t = 95)]
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
    rayon_threads: Option<usize>,
    #[arg(long, default_value_t = 8)]
    tokio_workers: usize,
    #[arg(long, default_value_t = 32)]
    tokio_blocking_threads: usize,
    #[arg(long, default_value_t = 32)]
    max_inflight_families: usize,
    #[arg(long, default_value_t = false)]
    prewarm_on_l2: bool,
    #[arg(long, default_value_t = false)]
    grayscale_only: bool,
    /// Cache directory for saving reconstructed tiles to disk
    #[arg(long)]
    cache_dir: Option<String>,
    /// Output format for reconstructed tiles: jpeg or webp
    #[arg(long, default_value = "jpeg")]
    output_format: String,
}

#[derive(Clone)]
struct AppState {
    slides: Arc<HashMap<String, Slide>>,
    cache: Arc<moka::sync::Cache<TileKey, Bytes>>,
    tile_quality: u8,
    timing_breakdown: bool,
    writer: Option<mpsc::Sender<WriteJob>>,
    write_generated_dir: Option<PathBuf>,
    metrics: Arc<Mutex<Metrics>>,
    buffer_pool: Arc<BufferPool>,
    #[allow(dead_code)]
    pack_dir: Option<PathBuf>,
    inflight: Arc<InflightFamilies>,
    inflight_limit: Arc<Semaphore>,
    prewarm_on_l2: bool,
    grayscale_only: bool,
    cache_dir: Option<PathBuf>,
    output_format: OutputFormat,
}

#[derive(serde::Deserialize)]
struct ModeQuery {
    mode: Option<String>,
}

#[derive(Clone)]
struct Slide {
    slide_id: String,
    dzi_path: PathBuf,
    files_dir: PathBuf,
    residuals_dir: PathBuf,
    pack_dir: PathBuf,
    /// mmapped bundle file (preferred over pack_dir for residual serving)
    bundle: Option<Arc<BundleFile>>,
    tile_size: u32,
    max_level: u32,
    l0: u32,
    l1: u32,
    l2: u32,
    /// Human-readable label (e.g. "Origami 60/40" or "JPEG Q98")
    label: String,
    /// Total disk size in MB for all served content
    disk_size_mb: f64,
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

struct ServerFamilyResult {
    tiles: HashMap<TileKey, Bytes>,
    stats: Option<crate::core::reconstruct::FamilyStats>,
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
        self.max
            .swap(self.current.load(Ordering::SeqCst), Ordering::SeqCst)
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

pub fn run(args: ServeArgs) -> Result<()> {
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

async fn async_main(args: ServeArgs) -> Result<()> {
    let slides = discover_slides(&args.slides_root, &args.residuals_dir, &args.pack_dir)?;
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

    let output_format = match args.output_format.as_str() {
        "jpeg" | "jpg" => OutputFormat::Jpeg,
        "webp" => OutputFormat::Webp,
        other => return Err(anyhow!("unknown output format: '{}'. Available: jpeg, webp", other)),
    };

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

    if let Some(ref cache_dir_str) = args.cache_dir {
        let cache_path = PathBuf::from(cache_dir_str);
        if let Err(err) = fs::create_dir_all(&cache_path) {
            info!(
                "Failed to create cache directory {}: {}",
                cache_path.display(),
                err
            );
        } else {
            info!("Cache directory enabled: {}", cache_path.display());
        }
    }

    let cache = moka::sync::Cache::builder()
        .max_capacity(args.cache_entries as u64)
        .time_to_idle(Duration::from_secs(300))
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
        pack_dir: None,
        inflight,
        inflight_limit,
        prewarm_on_l2: args.prewarm_on_l2,
        grayscale_only: args.grayscale_only,
        cache_dir: args.cache_dir.map(PathBuf::from),
        output_format,
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
        .route("/compare/:left/:right", get(compare_viewer))
        .route("/compare4/:a/:b/:c/:d", get(compare4_viewer))
        .route("/compare", get(compare_picker))
        .route("/slides.json", get(list_slides))
        .with_state(state)
        .layer(CorsLayer::new().allow_origin(Any).allow_methods(Any).allow_headers(Any))
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
    info!("compare url: http://{}/compare/<left>/<right>", addr);
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
    let slide = state.slides.get(slide_id).ok_or(StatusCode::NOT_FOUND)?;
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
    let slide = state.slides.get(&slide_id).ok_or(StatusCode::NOT_FOUND)?;

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
    let content_type = match residual_path.extension().and_then(|e| e.to_str()) {
        Some("webp") => "image/webp",
        Some("jxl") => "image/jxl",
        _ => "image/jpeg",
    };
    let mut resp = Response::new(bytes.into());
    resp.headers_mut()
        .insert(header::CONTENT_TYPE, HeaderValue::from_static(content_type));
    Ok(resp)
}

async fn serve_tile(
    state: AppState,
    slide_id: String,
    level: u32,
    tile: String,
) -> Result<Response, StatusCode> {
    let start = Instant::now();
    let (x, y) = parse_tile_name(&tile).ok_or(StatusCode::BAD_REQUEST)?;
    let slide = state.slides.get(&slide_id).ok_or(StatusCode::NOT_FOUND)?;

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
        return Ok(tile_response(bytes.to_vec(), state.output_format));
    }

    let slide = slide.clone();
    let cache = state.cache.clone();
    let quality = state.tile_quality;
    let timing = state.timing_breakdown;
    let writer = state.writer.clone();
    let write_root = state.write_generated_dir.clone();
    let buffer_pool = state.buffer_pool.clone();
    let inflight = state.inflight.clone();
    let inflight_limit = state.inflight_limit.clone();
    let grayscale_only = state.grayscale_only;
    let output_format = state.output_format;
    let cache_dir = state.cache_dir.clone();
    let permit = inflight_limit
        .acquire_owned()
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
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
        let result = generate_family_server(
            &slide, x2, y2, quality, timing, grayscale_only, output_format,
            &writer, &write_root, &buffer_pool, cache_dir.as_deref(),
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
                        slide.slide_id, x2, y2,
                        stats.l2_decode_ms, stats.l1_resize_ms,
                        stats.l1_residual_ms, stats.l1_encode_ms,
                        stats.l0_resize_ms, stats.l0_residual_ms, stats.l0_encode_ms,
                        stats.total_ms, stats.residuals_l1, stats.residuals_l0,
                        stats.l1_parallel_max, stats.l0_parallel_max
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
    Ok(tile_response(bytes.0.to_vec(), state.output_format))
}

fn spawn_family_prewarm(state: AppState, slide: Slide, x2: u32, y2: u32) {
    let cache = state.cache.clone();
    let quality = state.tile_quality;
    let grayscale_only = state.grayscale_only;
    let output_format = state.output_format;
    let writer = state.writer.clone();
    let write_root = state.write_generated_dir.clone();
    let buffer_pool = state.buffer_pool.clone();
    let cache_dir = state.cache_dir.clone();
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
            let result = generate_family_server(
                &slide, x2, y2, quality, false, grayscale_only, output_format,
                &writer, &write_root, &buffer_pool, cache_dir.as_deref(),
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
    let slide = state.slides.get(&slide_id).ok_or(StatusCode::NOT_FOUND)?;
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
        tileSources: "/dzi/{slide_id}.dzi",
        animationTime: 0.6,
        springStiffness: 12,
        visibilityRatio: 1.0,
        constrainDuringPan: false,
        immediateRender: false,
        blendTime: 0.15,
        gestureSettingsMouse: {{
          flickEnabled: true,
          flickMinSpeed: 180,
          flickMomentum: 0.15,
          dragToPan: true,
          scrollToZoom: true,
          clickToZoom: true,
          dblClickToZoom: true,
          pinchToZoom: true
        }},
        gestureSettingsTouch: {{
          flickEnabled: true,
          flickMinSpeed: 120,
          flickMomentum: 0.15,
          dragToPan: true,
          scrollToZoom: false,
          clickToZoom: false,
          dblClickToZoom: true,
          pinchToZoom: true
        }},
        maxZoomPixelRatio: 2,
        pixelsPerWheelLine: 60,
        zoomPerScroll: 1.2,
        zoomPerClick: 2.0,
        minZoomImageRatio: 0.5,
        maxZoomLevel: 40
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

async fn compare_viewer(
    State(state): State<AppState>,
    AxumPath((left, right)): AxumPath<(String, String)>,
) -> Result<Html<String>, StatusCode> {
    let sl = state.slides.get(&left).ok_or(StatusCode::NOT_FOUND)?;
    let sr = state.slides.get(&right).ok_or(StatusCode::NOT_FOUND)?;
    let left_label = format!("{} ({:.0} MB)", sl.label, sl.disk_size_mb);
    let right_label = format!("{} ({:.0} MB)", sr.label, sr.disk_size_mb);

    let html = format!(
        r##"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Compare: {left_label} vs {right_label}</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.0/openseadragon.min.js"
          crossorigin="anonymous" referrerpolicy="no-referrer"></script>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    html, body {{ height: 100%; font-family: ui-sans-serif, system-ui, -apple-system, sans-serif; background: #111; color: #eee; }}
    .container {{ display: flex; height: 100%; }}
    .panel {{ flex: 1; position: relative; border-right: 2px solid #333; }}
    .panel:last-child {{ border-right: none; }}
    .panel .viewer {{ width: 100%; height: 100%; }}
    .label {{ position: absolute; top: 12px; left: 12px; z-index: 100;
              background: rgba(0,0,0,0.7); color: #fff; padding: 6px 14px;
              border-radius: 6px; font-size: 14px; font-weight: 600;
              pointer-events: none; }}
  </style>
</head>
<body>
  <div class="container">
    <div class="panel">
      <div class="label">{left_label}</div>
      <div id="osd-left" class="viewer"></div>
    </div>
    <div class="panel">
      <div class="label">{right_label}</div>
      <div id="osd-right" class="viewer"></div>
    </div>
  </div>
  <script>
    var osdOpts = {{
      prefixUrl: "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.0/images/",
      animationTime: 0.6,
      springStiffness: 12,
      visibilityRatio: 1.0,
      constrainDuringPan: false,
      immediateRender: false,
      blendTime: 0.15,
      maxZoomPixelRatio: 2,
      pixelsPerWheelLine: 60,
      zoomPerScroll: 1.2,
      zoomPerClick: 2.0,
      minZoomImageRatio: 0.5,
      maxZoomLevel: 40,
      gestureSettingsMouse: {{ flickEnabled: true, flickMinSpeed: 180, flickMomentum: 0.15,
        dragToPan: true, scrollToZoom: true, clickToZoom: true, dblClickToZoom: true, pinchToZoom: true }},
      gestureSettingsTouch: {{ flickEnabled: true, flickMinSpeed: 120, flickMomentum: 0.15,
        dragToPan: true, scrollToZoom: false, clickToZoom: false, dblClickToZoom: true, pinchToZoom: true }},
    }};

    var left = OpenSeadragon(Object.assign({{ id: "osd-left", showNavigator: true,
      tileSources: "/dzi/{left}.dzi" }}, osdOpts));
    var right = OpenSeadragon(Object.assign({{ id: "osd-right", showNavigator: false,
      tileSources: "/dzi/{right}.dzi" }}, osdOpts));

    var syncing = false;
    function syncTo(src, dst) {{
      if (syncing) return;
      syncing = true;
      dst.viewport.panTo(src.viewport.getCenter(false), false);
      dst.viewport.zoomTo(src.viewport.getZoom(false), null, false);
      syncing = false;
    }}

    left.addHandler('zoom',   function() {{ syncTo(left, right); }});
    left.addHandler('pan',    function() {{ syncTo(left, right); }});
    right.addHandler('zoom',  function() {{ syncTo(right, left); }});
    right.addHandler('pan',   function() {{ syncTo(right, left); }});
  </script>
</body>
</html>"##
    );

    info!("compare viewer requested left={} right={}", left, right);
    Ok(Html(html))
}

async fn compare4_viewer(
    State(state): State<AppState>,
    AxumPath((a, b, c, d)): AxumPath<(String, String, String, String)>,
) -> Result<Html<String>, StatusCode> {
    let slides: Vec<_> = [&a, &b, &c, &d].iter().map(|id| {
        state.slides.get(id.as_str()).ok_or(StatusCode::NOT_FOUND)
    }).collect::<Result<Vec<_>, _>>()?;
    let labels: Vec<String> = slides.iter().map(|s| {
        format!("{} ({:.0} MB)", s.label, s.disk_size_mb)
    }).collect();
    let ids = [&a, &b, &c, &d];

    let html = format!(
        r##"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Compare 4-up</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.0/openseadragon.min.js"
          crossorigin="anonymous" referrerpolicy="no-referrer"></script>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    html, body {{ height: 100%; font-family: ui-sans-serif, system-ui, -apple-system, sans-serif; background: #111; color: #eee; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; grid-template-rows: 1fr 1fr; height: 100%; gap: 2px; background: #333; }}
    .panel {{ position: relative; background: #111; overflow: hidden; }}
    .panel .viewer {{ width: 100%; height: 100%; }}
    .label {{ position: absolute; top: 8px; left: 8px; z-index: 100;
              background: rgba(0,0,0,0.75); color: #fff; padding: 5px 12px;
              border-radius: 5px; font-size: 13px; font-weight: 600;
              pointer-events: none; }}
  </style>
</head>
<body>
  <div class="grid">
    <div class="panel">
      <div class="label">{label0}</div>
      <div id="osd-0" class="viewer"></div>
    </div>
    <div class="panel">
      <div class="label">{label1}</div>
      <div id="osd-1" class="viewer"></div>
    </div>
    <div class="panel">
      <div class="label">{label2}</div>
      <div id="osd-2" class="viewer"></div>
    </div>
    <div class="panel">
      <div class="label">{label3}</div>
      <div id="osd-3" class="viewer"></div>
    </div>
  </div>
  <script>
    var osdOpts = {{
      prefixUrl: "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.0/images/",
      showNavigator: false,
      animationTime: 0.6,
      springStiffness: 12,
      visibilityRatio: 1.0,
      constrainDuringPan: false,
      immediateRender: false,
      blendTime: 0.15,
      maxZoomPixelRatio: 2,
      pixelsPerWheelLine: 60,
      zoomPerScroll: 1.2,
      zoomPerClick: 2.0,
      minZoomImageRatio: 0.5,
      maxZoomLevel: 40,
      gestureSettingsMouse: {{ flickEnabled: true, flickMinSpeed: 180, flickMomentum: 0.15,
        dragToPan: true, scrollToZoom: true, clickToZoom: true, dblClickToZoom: true, pinchToZoom: true }},
      gestureSettingsTouch: {{ flickEnabled: true, flickMinSpeed: 120, flickMomentum: 0.15,
        dragToPan: true, scrollToZoom: false, clickToZoom: false, dblClickToZoom: true, pinchToZoom: true }},
    }};

    var viewers = [
      OpenSeadragon(Object.assign({{ id: "osd-0", showNavigator: true, tileSources: "/dzi/{id0}.dzi" }}, osdOpts)),
      OpenSeadragon(Object.assign({{ id: "osd-1", tileSources: "/dzi/{id1}.dzi" }}, osdOpts)),
      OpenSeadragon(Object.assign({{ id: "osd-2", tileSources: "/dzi/{id2}.dzi" }}, osdOpts)),
      OpenSeadragon(Object.assign({{ id: "osd-3", tileSources: "/dzi/{id3}.dzi" }}, osdOpts)),
    ];

    var syncing = false;
    function syncFrom(src) {{
      if (syncing) return;
      syncing = true;
      var center = src.viewport.getCenter(false);
      var zoom = src.viewport.getZoom(false);
      for (var i = 0; i < viewers.length; i++) {{
        if (viewers[i] !== src) {{
          viewers[i].viewport.panTo(center, false);
          viewers[i].viewport.zoomTo(zoom, null, false);
        }}
      }}
      syncing = false;
    }}

    for (var i = 0; i < viewers.length; i++) {{
      (function(v) {{
        v.addHandler('zoom', function() {{ syncFrom(v); }});
        v.addHandler('pan',  function() {{ syncFrom(v); }});
      }})(viewers[i]);
    }}
  </script>
</body>
</html>"##,
        label0 = labels[0], label1 = labels[1], label2 = labels[2], label3 = labels[3],
        id0 = ids[0], id1 = ids[1], id2 = ids[2], id3 = ids[3],
    );

    info!("compare4 viewer requested: {} / {} / {} / {}", a, b, c, d);
    Ok(Html(html))
}

async fn list_slides(
    State(state): State<AppState>,
) -> impl IntoResponse {
    let mut slides: Vec<serde_json::Value> = state.slides.iter().map(|(id, s)| {
        serde_json::json!({
            "id": id,
            "label": s.label,
            "disk_size_mb": s.disk_size_mb,
        })
    }).collect();
    slides.sort_by(|a, b| a["label"].as_str().unwrap_or("").cmp(b["label"].as_str().unwrap_or("")));
    axum::Json(slides)
}

async fn compare_picker(
    State(state): State<AppState>,
) -> Html<String> {
    // Build sorted slide list for initial rendering
    let mut slide_ids: Vec<&str> = state.slides.keys().map(|s| s.as_str()).collect();
    slide_ids.sort();

    let html = r##"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Compare Slides</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.0/openseadragon.min.js"
          crossorigin="anonymous" referrerpolicy="no-referrer"></script>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    html, body { height: 100%; font-family: ui-sans-serif, system-ui, -apple-system, sans-serif; background: #111; color: #eee; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; grid-template-rows: 1fr 1fr; height: 100%; gap: 2px; background: #333; }
    .panel { position: relative; background: #111; overflow: hidden; }
    .panel .viewer { width: 100%; height: 100%; }
    .panel-header { position: absolute; top: 8px; left: 8px; z-index: 100; display: flex; align-items: center; gap: 6px; }
    .panel-header select {
      background: rgba(0,0,0,0.85); color: #fff; border: 1px solid #555;
      padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: 600;
      cursor: pointer; max-width: 300px;
    }
    .panel-header select:hover { border-color: #888; }
    .size-label {
      background: rgba(0,0,0,0.75); color: #aaa; padding: 4px 8px;
      border-radius: 4px; font-size: 11px; pointer-events: none;
    }
  </style>
</head>
<body>
  <div class="grid">
    <div class="panel">
      <div class="panel-header"><select id="sel-0"></select><span id="size-0" class="size-label"></span></div>
      <div id="osd-0" class="viewer"></div>
    </div>
    <div class="panel">
      <div class="panel-header"><select id="sel-1"></select><span id="size-1" class="size-label"></span></div>
      <div id="osd-1" class="viewer"></div>
    </div>
    <div class="panel">
      <div class="panel-header"><select id="sel-2"></select><span id="size-2" class="size-label"></span></div>
      <div id="osd-2" class="viewer"></div>
    </div>
    <div class="panel">
      <div class="panel-header"><select id="sel-3"></select><span id="size-3" class="size-label"></span></div>
      <div id="osd-3" class="viewer"></div>
    </div>
  </div>
  <script>
    var slides = [];
    var viewers = [null, null, null, null];
    var syncing = false;

    var osdOpts = {
      prefixUrl: "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.0/images/",
      showNavigator: false,
      animationTime: 0.6,
      springStiffness: 12,
      visibilityRatio: 1.0,
      constrainDuringPan: false,
      immediateRender: false,
      blendTime: 0.15,
      maxZoomPixelRatio: 2,
      pixelsPerWheelLine: 60,
      zoomPerScroll: 1.2,
      zoomPerClick: 2.0,
      minZoomImageRatio: 0.5,
      maxZoomLevel: 40,
      gestureSettingsMouse: { flickEnabled: true, flickMinSpeed: 180, flickMomentum: 0.15,
        dragToPan: true, scrollToZoom: true, clickToZoom: true, dblClickToZoom: true, pinchToZoom: true },
      gestureSettingsTouch: { flickEnabled: true, flickMinSpeed: 120, flickMomentum: 0.15,
        dragToPan: true, scrollToZoom: false, clickToZoom: false, dblClickToZoom: true, pinchToZoom: true },
    };

    function syncFrom(src) {
      if (syncing) return;
      syncing = true;
      var center = src.viewport.getCenter(false);
      var zoom = src.viewport.getZoom(false);
      for (var i = 0; i < viewers.length; i++) {
        if (viewers[i] && viewers[i] !== src) {
          viewers[i].viewport.panTo(center, false);
          viewers[i].viewport.zoomTo(zoom, null, false);
        }
      }
      syncing = false;
    }

    function createViewer(idx, slideId) {
      if (viewers[idx]) {
        viewers[idx].destroy();
      }
      var v = OpenSeadragon(Object.assign({
        id: "osd-" + idx,
        showNavigator: idx === 0,
        tileSources: "/dzi/" + slideId + ".dzi"
      }, osdOpts));
      v.addHandler('zoom', function() { syncFrom(v); });
      v.addHandler('pan',  function() { syncFrom(v); });
      viewers[idx] = v;

      // Sync new viewer to existing position
      for (var i = 0; i < viewers.length; i++) {
        if (i !== idx && viewers[i]) {
          v.addOnceHandler('open', function() {
            var ref = null;
            for (var j = 0; j < viewers.length; j++) {
              if (j !== idx && viewers[j]) { ref = viewers[j]; break; }
            }
            if (ref) {
              syncing = true;
              v.viewport.panTo(ref.viewport.getCenter(false), true);
              v.viewport.zoomTo(ref.viewport.getZoom(false), null, true);
              syncing = false;
            }
          });
          break;
        }
      }

      // Update size label
      var slide = slides.find(function(s) { return s.id === slideId; });
      document.getElementById("size-" + idx).textContent =
        slide ? slide.disk_size_mb.toFixed(0) + " MB" : "";
    }

    function populateSelectors() {
      for (var idx = 0; idx < 4; idx++) {
        var sel = document.getElementById("sel-" + idx);
        sel.innerHTML = "";
        for (var j = 0; j < slides.length; j++) {
          var opt = document.createElement("option");
          opt.value = slides[j].id;
          opt.textContent = slides[j].label;
          sel.appendChild(opt);
        }
      }
    }

    function onSelectChange(idx) {
      var sel = document.getElementById("sel-" + idx);
      createViewer(idx, sel.value);
      // Save to URL hash
      var ids = [];
      for (var i = 0; i < 4; i++) {
        ids.push(document.getElementById("sel-" + i).value);
      }
      window.location.hash = ids.join("/");
    }

    // Load slides and initialize
    fetch("/slides.json").then(function(r) { return r.json(); }).then(function(data) {
      slides = data;
      populateSelectors();

      // Parse initial selection from URL hash or use first 4 slides
      var hash = window.location.hash.replace("#", "");
      var initial = hash ? hash.split("/") : [];
      for (var i = 0; i < 4; i++) {
        var sel = document.getElementById("sel-" + i);
        if (initial[i] && slides.find(function(s) { return s.id === initial[i]; })) {
          sel.value = initial[i];
        } else if (slides[i]) {
          sel.value = slides[Math.min(i, slides.length - 1)].id;
        }
        sel.onchange = (function(idx) { return function() { onSelectChange(idx); }; })(i);
        createViewer(i, sel.value);
      }
    });
  </script>
</body>
</html>"##.to_string();

    info!("compare picker viewer requested");
    Html(html)
}

fn parse_tile_name(name: &str) -> Option<(u32, u32)> {
    let trimmed = name
        .strip_suffix(".jpg")
        .or_else(|| name.strip_suffix(".webp"))
        .unwrap_or(name);
    let mut parts = trimmed.split('_');
    let x = parts.next()?.parse().ok()?;
    let y = parts.next()?.parse().ok()?;
    Some((x, y))
}

fn tile_response(bytes: Vec<u8>, format: OutputFormat) -> Response {
    let mut resp = Response::new(bytes.into());
    resp.headers_mut()
        .insert(header::CONTENT_TYPE, HeaderValue::from_static(format.content_type()));
    resp
}

fn jpeg_response(bytes: Vec<u8>) -> Response {
    tile_response(bytes, OutputFormat::Jpeg)
}

fn baseline_tile_path(slide: &Slide, level: u32, x: u32, y: u32) -> PathBuf {
    slide
        .files_dir
        .join(level.to_string())
        .join(format!("{}_{}.jpg", x, y))
}

fn residual_tile_path(slide: &Slide, level: u32, x2: u32, y2: u32, x: u32, y: u32) -> PathBuf {
    let subdir = if level == slide.l1 { "L1" } else { "L0" };
    let parent = slide
        .residuals_dir
        .join(subdir)
        .join(format!("{}_{}", x2, y2));
    for ext in &[".jpg", ".webp", ".jxl"] {
        let p = parent.join(format!("{}_{}{}", x, y, ext));
        if p.exists() {
            return p;
        }
    }
    parent.join(format!("{}_{}.jpg", x, y))
}

fn enqueue_generated(
    writer: &Option<mpsc::Sender<WriteJob>>,
    write_root: &Option<PathBuf>,
    tiles: &HashMap<TileKey, Bytes>,
    format: OutputFormat,
) {
    let Some(writer) = writer else { return };
    let Some(root) = write_root else { return };
    let ext = format.extension();
    for (key, bytes) in tiles.iter() {
        let mut path = root.clone();
        path.push(&key.slide_id);
        path.push(key.level.to_string());
        path.push(format!("{}_{}{}", key.x, key.y, ext));
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

fn discover_slides(
    root: &Path,
    residuals_dir: &str,
    pack_dir: &str,
) -> Result<HashMap<String, Slide>> {
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

        // Compute label and disk size from summary.json
        let summary_path = slide_root.join("summary.json");
        let (label, disk_size_mb) = if let Ok(data) = fs::read_to_string(&summary_path) {
            if let Ok(summary) = serde_json::from_str::<serde_json::Value>(&data) {
                let mode = summary.get("mode").and_then(|v| v.as_str()).unwrap_or("");
                let lbl = if mode == "ingest-jpeg-only" {
                    let q = summary.get("baseq").and_then(|v| v.as_u64()).unwrap_or(0);
                    format!("JPEG Q{}", q)
                } else {
                    let baseq = summary.get("baseq").and_then(|v| v.as_u64()).unwrap_or(0);
                    let l1q = summary.get("l1q").and_then(|v| v.as_u64()).unwrap_or(0);
                    let l0q = summary.get("l0q").and_then(|v| v.as_u64()).unwrap_or(0);
                    format!("Origami {}/{}/{}", baseq, l1q, l0q)
                };
                let size = if mode == "ingest-jpeg-only" {
                    // JPEG-only: total_bytes covers everything
                    let total = summary.get("total_bytes").and_then(|v| v.as_u64()).unwrap_or(0);
                    total as f64 / 1_048_576.0
                } else {
                    // Origami: l2_bytes + residual_bytes (pack files)
                    let l2 = summary.get("l2_bytes").and_then(|v| v.as_u64()).unwrap_or(0);
                    let res = summary.get("residual_bytes").and_then(|v| v.as_u64()).unwrap_or(0);
                    (l2 + res) as f64 / 1_048_576.0
                };
                (lbl, size)
            } else {
                (slide_id.clone(), 0.0)
            }
        } else {
            (slide_id.clone(), 0.0)
        };

        // Try to open a bundle file (preferred over individual .pack files)
        let pack_dir_path = slide_root.join(pack_dir);
        let bundle_path = pack_dir_path.join("residuals.bundle");
        let bundle = if bundle_path.exists() {
            match open_bundle(&bundle_path) {
                Ok(b) => {
                    info!("  opened bundle: {} ({}x{})", bundle_path.display(), b.grid_cols(), b.grid_rows());
                    Some(Arc::new(b))
                }
                Err(e) => {
                    info!("  bundle open failed ({}), falling back to individual packs", e);
                    None
                }
            }
        } else {
            None
        };

        let slide = Slide {
            slide_id: slide_id.clone(),
            dzi_path,
            files_dir,
            residuals_dir: slide_root.join(residuals_dir),
            pack_dir: pack_dir_path,
            bundle,
            tile_size,
            max_level,
            l0,
            l1,
            l2,
            label,
            disk_size_mb,
        };
        slides.insert(slide_id, slide);
    }
    Ok(slides)
}

// ---------------------------------------------------------------------------
// generate_family_server — thin adapter from Slide to core::reconstruct
// ---------------------------------------------------------------------------

/// Server-side adapter: builds ReconstructInput from Slide, calls the shared
/// pipeline, then maps the result into server types (TileKey → Bytes HashMap).
fn generate_family_server(
    slide: &Slide,
    x2: u32,
    y2: u32,
    quality: u8,
    timing: bool,
    grayscale_only: bool,
    output_format: OutputFormat,
    writer: &Option<mpsc::Sender<WriteJob>>,
    write_root: &Option<PathBuf>,
    buffer_pool: &BufferPool,
    cache_dir: Option<&Path>,
) -> Result<ServerFamilyResult> {
    let input = ReconstructInput {
        files_dir: &slide.files_dir,
        residuals_dir: Some(&slide.residuals_dir),
        pack_dir: Some(&slide.pack_dir),
        bundle: slide.bundle.as_deref(),
        tile_size: slide.tile_size,
        l0: slide.l0,
        l1: slide.l1,
        l2: slide.l2,
    };
    let opts = ReconstructOpts {
        quality,
        timing,
        grayscale_only,
        output_format,
    };

    let core_result = reconstruct_family(&input, x2, y2, &opts, buffer_pool)?;

    // Map CoreFamilyResult → server HashMap<TileKey, Bytes>
    let mut out = HashMap::new();
    for (x1, y1, bytes) in &core_result.l1 {
        out.insert(
            TileKey {
                slide_id: slide.slide_id.clone(),
                level: slide.l1,
                x: *x1,
                y: *y1,
            },
            Bytes::from(bytes.clone()),
        );
    }
    for (x0, y0, bytes) in &core_result.l0 {
        out.insert(
            TileKey {
                slide_id: slide.slide_id.clone(),
                level: slide.l0,
                x: *x0,
                y: *y0,
            },
            Bytes::from(bytes.clone()),
        );
    }

    // Enqueue async disk writes if configured
    enqueue_generated(writer, write_root, &out, output_format);

    // Write to cache dir if configured
    if let Some(cache_dir) = cache_dir {
        let ext = output_format.extension();
        for (tile_key, tile_data) in &out {
            let tile_path = cache_dir
                .join(&slide.slide_id)
                .join(format!("{}", tile_key.level))
                .join(format!("{}_{}{}", tile_key.x, tile_key.y, ext));
            if let Some(parent) = tile_path.parent() {
                if let Err(err) = fs::create_dir_all(parent) {
                    info!("Failed to create cache dir {}: {}", parent.display(), err);
                    continue;
                }
            }
            if let Err(err) = fs::write(&tile_path, tile_data) {
                info!(
                    "Failed to write cached tile {}: {}",
                    tile_path.display(),
                    err
                );
            }
        }
    }

    Ok(ServerFamilyResult {
        tiles: out,
        stats: core_result.stats,
    })
}
