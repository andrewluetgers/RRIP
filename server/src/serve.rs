use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use parking_lot::Mutex as PLMutex;

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
use crate::core::ResampleFilter;

// ---------------------------------------------------------------------------
// RocksDB tile cache
// ---------------------------------------------------------------------------

struct TileCache {
    db: rocksdb::DB,
}

impl TileCache {
    fn open(path: &Path, block_cache_mb: usize) -> Result<Self> {
        let mut block_opts = rocksdb::BlockBasedOptions::default();
        let cache = rocksdb::Cache::new_lru_cache(block_cache_mb * 1024 * 1024);
        block_opts.set_block_cache(&cache);
        // JPEG/WebP bytes are already compressed — skip RocksDB compression
        let mut opts = rocksdb::Options::default();
        opts.create_if_missing(true);
        opts.set_block_based_table_factory(&block_opts);
        opts.set_compression_type(rocksdb::DBCompressionType::None);
        // WAL disabled — cache data is regenerable
        let mut write_opts = rocksdb::WriteOptions::default();
        write_opts.disable_wal(true);
        let db = rocksdb::DB::open(&opts, path)
            .with_context(|| format!("opening RocksDB at {}", path.display()))?;
        Ok(Self { db })
    }

    fn get(&self, key: &TileKey) -> Option<Vec<u8>> {
        let k = format!("tile:{}:{}:{}:{}", key.slide_id, key.level, key.x, key.y);
        self.db.get(k.as_bytes()).ok().flatten()
    }

    fn put_family(&self, tiles: &HashMap<TileKey, Bytes>) {
        let mut batch = rocksdb::WriteBatch::default();
        for (key, data) in tiles {
            let k = format!("tile:{}:{}:{}:{}", key.slide_id, key.level, key.x, key.y);
            batch.put(k.as_bytes(), data.as_ref());
        }
        let mut write_opts = rocksdb::WriteOptions::default();
        write_opts.disable_wal(true);
        if let Err(e) = self.db.write_opt(batch, &write_opts) {
            tracing::warn!("RocksDB WriteBatch failed: {}", e);
        }
    }
}

#[derive(Args, Debug)]
pub struct ServeArgs {
    #[arg(long, default_value = "data")]
    slides_root: PathBuf,
    #[arg(long, default_value = "residual_packs", hide = true)]
    residuals_dir: String, // legacy, unused in v2 pipeline
    #[arg(long, default_value = "residual_packs")]
    pack_dir: String,
    #[arg(long, default_value_t = 95)]
    tile_quality: u8,
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
    /// RocksDB directory for tile cache (persistent across restarts)
    #[arg(long)]
    cache_dir: Option<String>,
    /// RocksDB block cache size in MB (in-memory hot tiles)
    #[arg(long, default_value_t = 256)]
    cache_block_mb: usize,
    /// Output format for reconstructed tiles: jpeg or webp
    #[arg(long, default_value = "jpeg")]
    output_format: String,
    /// Unsharp mask strength to apply to L2 tiles before upsampling (decode-time sharpening)
    #[arg(long)]
    sharpen: Option<f32>,
    /// Upsample filter for predictions: bilinear, bicubic, lanczos3 (default: lanczos3)
    #[arg(long, default_value = "lanczos3")]
    upsample_filter: String,
    /// Enable HTTPS with auto-generated self-signed cert (enables HTTP/2 multiplexing)
    #[arg(long, default_value_t = false)]
    tls: bool,
    /// Serve evals viewer static files from this directory (e.g. evals/viewer/public)
    #[arg(long)]
    viewer_dir: Option<PathBuf>,
    /// Serve evals run images from this directory (e.g. evals/runs)
    #[arg(long)]
    runs_dir: Option<PathBuf>,
}

#[derive(Clone)]
struct AppState {
    slides: Arc<HashMap<String, Slide>>,
    tile_cache: Option<Arc<TileCache>>,
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
    output_format: OutputFormat,
    sharpen: Option<f32>,
    upsample_filter: ResampleFilter,
    singleflight: Arc<InflightMap>,
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

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
struct FamilyKey {
    slide_id: String,
    x2: u32,
    y2: u32,
}

/// Singleflight map: each in-flight family has a watch channel.
/// Leader creates the channel and sends the result; waiters subscribe and await.
type InflightMap = PLMutex<HashMap<FamilyKey, tokio::sync::watch::Receiver<Option<bool>>>>;

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
    tile_singleflight_hit: u64,
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
            "singleflight_hit" => self.tile_singleflight_hit += 1,
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

    let tile_cache = if let Some(ref cache_dir_str) = args.cache_dir {
        let cache_path = PathBuf::from(cache_dir_str);
        match TileCache::open(&cache_path, args.cache_block_mb) {
            Ok(tc) => {
                info!("RocksDB tile cache opened: {} (block_cache={}MB)", cache_path.display(), args.cache_block_mb);
                Some(Arc::new(tc))
            }
            Err(e) => {
                info!("Failed to open RocksDB tile cache at {}: {} — running without cache", cache_path.display(), e);
                None
            }
        }
    } else {
        None
    };
    let inflight = Arc::new(InflightFamilies::new());
    let inflight_limit = Arc::new(Semaphore::new(args.max_inflight_families));
    let state = AppState {
        slides: slides.clone(),
        tile_cache,
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
        output_format,
        sharpen: args.sharpen,
        upsample_filter: args.upsample_filter.parse::<ResampleFilter>()
            .unwrap_or(ResampleFilter::Lanczos3),
        singleflight: Arc::new(PLMutex::new(HashMap::new())),
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
                    "metrics tiles_total={} baseline={} cache_hit={} generated={} fallback={} singleflight_hit={} residual_view={} tile_avg_ms={} tile_max_ms={} families={} family_avg_ms={} family_max_ms={} pool_total={} pool_avail={} pool_in_use_max={} inflight_current={} inflight_max={} rss_kb={} rss_mb={} cpu_pct={:.1}",
                    snapshot.tile_total,
                    snapshot.tile_baseline,
                    snapshot.tile_cache_hit,
                    snapshot.tile_generated,
                    snapshot.tile_fallback,
                    snapshot.tile_singleflight_hit,
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
        .route("/compare", get(compare_dynamic))
        .route("/compare/:left/:right", get(compare_viewer))
        .route("/compare/:a/:b/:c/:d", get(compare4_viewer))
        .route("/slides.json", get(list_slides))
        .with_state(state)
        .layer(CorsLayer::new().allow_origin(Any).allow_methods(Any).allow_headers(Any))
        .layer(TraceLayer::new_for_http());

    // Optionally serve evals viewer static files and run images
    let app = if let Some(ref viewer_dir) = args.viewer_dir {
        use tower_http::services::ServeDir;
        let mut app = app;
        if let Some(ref runs_dir) = args.runs_dir {
            info!("serving run images from {}", runs_dir.display());
            app = app.nest_service("/run-image", ServeDir::new(runs_dir));
        }
        info!("serving viewer from {}", viewer_dir.display());
        app.fallback_service(ServeDir::new(viewer_dir))
    } else {
        app
    };

    let bind_addr = format!("0.0.0.0:{}", args.port);
    let scheme = if args.tls { "https" } else { "http" };
    let display_addr = format!("localhost:{}", args.port);
    info!(
        "listening on {} (rayon_threads={}, tokio_workers={}, tokio_blocking_threads={}, hw_threads={})",
        bind_addr,
        rayon::current_num_threads(),
        args.tokio_workers,
        args.tokio_blocking_threads,
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0)
    );
    println!("\n  ORIGAMI tile server: {}://{}\n", scheme, display_addr);

    if args.tls {
        // Generate self-signed cert for HTTP/2
        let cert = rcgen::generate_simple_self_signed(vec!["localhost".to_string()])
            .map_err(|e| anyhow!("cert generation failed: {}", e))?;
        let cert_pem = cert.cert.pem();
        let key_pem = cert.key_pair.serialize_pem();

        let rustls_config = axum_server::tls_rustls::RustlsConfig::from_pem(
            cert_pem.into_bytes(),
            key_pem.into_bytes(),
        ).await.map_err(|e| anyhow!("TLS config failed: {}", e))?;

        info!("HTTPS/2 enabled with self-signed cert (accept browser warning)");
        axum_server::bind_rustls(bind_addr.parse()?, rustls_config)
            .serve(app.into_make_service())
            .await?;
    } else {
        let listener = tokio::net::TcpListener::bind(&bind_addr).await?;
        axum::serve(listener, app).await?;
    }
    Ok(())
}

async fn healthz() -> impl IntoResponse {
    "ok"
}

async fn homepage(State(state): State<AppState>) -> Html<String> {
    let mut slide_ids: Vec<&str> = state.slides.keys().map(|s| s.as_str()).collect();
    slide_ids.sort();

    let slides_html: String = slide_ids.iter().map(|id| {
        let slide = &state.slides[*id];
        let size = if slide.disk_size_mb > 0.0 {
            format!(" ({:.0} MB)", slide.disk_size_mb)
        } else {
            String::new()
        };
        format!(r#"<li><a href="/viewer/{id}">{label}{size}</a></li>"#,
            id = id, label = slide.label, size = size)
    }).collect::<Vec<_>>().join("\n        ");

    let html = format!(r##"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>ORIGAMI Tile Server</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; background: #111; color: #eee; }}
    a {{ color: #6bf; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    h1 {{ margin-bottom: 8px; }}
    .subtitle {{ color: #888; margin-bottom: 24px; }}
    ul {{ list-style: none; padding: 0; }}
    li {{ padding: 6px 0; }}
    li a {{ font-size: 16px; }}
    section {{ margin-bottom: 32px; }}
    h2 {{ font-size: 18px; color: #aaa; margin-bottom: 12px; border-bottom: 1px solid #333; padding-bottom: 6px; }}
    .compare-link {{ display: inline-block; padding: 8px 16px; background: #234; border-radius: 6px; margin: 4px 0; }}
    video {{ max-width: 100%; border-radius: 8px; margin: 12px 0; }}
  </style>
</head>
<body>
  <h1>ORIGAMI</h1>
  <p class="subtitle">Residual-Pyramid Image Processor &mdash; {count} slides loaded</p>

  <section>
    <h2>Compare</h2>
    <a class="compare-link" href="/compare">
      Side-by-side WSI viewer
    </a>
    <br/>
    <a class="compare-link" href="https://localhost:8084/comparison.html">
      Eval comparison viewer
    </a>
  </section>

  <section>
    <h2>Individual Viewers</h2>
    <ul>
        {slides}
    </ul>
  </section>
</body>
</html>"##,
        count = slide_ids.len(),
        slides = slides_html,
    );
    Html(html)
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
        // For L2 tiles, try loading from bundle first
        if level == slide.l2 {
            if let Some(ref bundle) = slide.bundle {
                if let Ok(pack) = bundle.get_pack(x, y) {
                    if let Some(l2_bytes) = pack.get_residual(2, 0) {
                        let jpeg = ensure_l2_jpeg(l2_bytes).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
                        state
                            .metrics
                            .lock()
                            .unwrap()
                            .record_tile("residual_view", 0);
                        info!(
                            "tile residual_view baseline(bundle) slide_id={} level={} x={} y={}",
                            slide_id, level, x, y
                        );
                        return Ok(jpeg_response(jpeg));
                    }
                }
            }
        }
        // Fallback to filesystem
        let path = baseline_tile_path(slide, level, x, y);
        let bytes = read_baseline_as_jpeg(&path).map_err(|_| StatusCode::NOT_FOUND)?;
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

    // V2 pipeline: no per-tile loose residuals, only fused L0 in packs
    Err(StatusCode::NOT_FOUND)
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
        // For L2 tiles, try loading from pack (bundle or individual .pack file)
        if level == slide.l2 {
            let pack = if let Some(ref bundle) = slide.bundle {
                bundle.get_pack(x, y).ok()
            } else {
                crate::core::pack::open_pack(&slide.pack_dir, x, y).ok()
            };
            if let Some(pack) = pack {
                if let Some(l2_bytes) = pack.get_l2() {
                    let jpeg = ensure_l2_jpeg(l2_bytes).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
                    if state.prewarm_on_l2 {
                        spawn_family_prewarm(state.clone(), slide.clone(), x, y);
                    }
                    state
                        .metrics
                        .lock()
                        .unwrap()
                        .record_tile("baseline", start.elapsed().as_millis());
                    info!(
                        "tile baseline(pack) slide_id={} level={} x={} y={} ms={}",
                        slide_id, level, x, y, start.elapsed().as_millis()
                    );
                    return Ok(jpeg_response(jpeg));
                }
            }
        }
        // Fallback to filesystem (for levels below L2, e.g. thumbnail pyramid)
        let path = baseline_tile_path(slide, level, x, y);
        let bytes = read_baseline_as_jpeg(&path).map_err(|_| StatusCode::NOT_FOUND)?;
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
    if let Some(ref tc) = state.tile_cache {
        if let Some(bytes) = tc.get(&key) {
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
            return Ok(tile_response(bytes, state.output_format));
        }
    }

    let slide = slide.clone();
    let (x2, y2) = if level == slide.l1 {
        (x >> 1, y >> 1)
    } else if level == slide.l0 {
        (x >> 2, y >> 2)
    } else {
        return Err(StatusCode::BAD_REQUEST);
    };

    let family_key = FamilyKey {
        slide_id: slide_id.clone(),
        x2,
        y2,
    };

    // Singleflight: check-then-register under a single lock acquisition.
    // We either become a waiter (existing entry) or the leader (insert new entry).
    let (watch_tx, waiter_rx) = {
        let mut map = state.singleflight.lock();
        if let Some(rx) = map.get(&family_key) {
            // Another task is already decoding this family — become a waiter
            (None, Some(rx.clone()))
        } else {
            // We are the leader — create watch channel and register
            let (tx, rx) = tokio::sync::watch::channel(None);
            map.insert(family_key.clone(), rx);
            (Some(Arc::new(tx)), None)
        }
    };
    // Lock is dropped here — safe to await below

    if let Some(mut rx) = waiter_rx {
        // Wait for the leader to finish
        let _ = rx.changed().await;
        let success = rx.borrow().unwrap_or(false);
        if success {
            if let Some(ref tc) = state.tile_cache {
                if let Some(bytes) = tc.get(&key) {
                    state
                        .metrics
                        .lock()
                        .unwrap()
                        .record_tile("singleflight_hit", start.elapsed().as_millis());
                    info!(
                        "tile singleflight_hit slide_id={} level={} x={} y={} ms={}",
                        slide_id, level, x, y, start.elapsed().as_millis()
                    );
                    return Ok(tile_response(bytes, state.output_format));
                }
            }
        }
        // Leader failed or tile not in cache — fall through to own decode as a new leader
    }

    // At this point we are the leader (or a waiter that fell through).
    // If we fell through, we need to register ourselves.
    let watch_tx = match watch_tx {
        Some(tx) => tx,
        None => {
            // Waiter fell through — register as new leader
            let (tx, rx) = tokio::sync::watch::channel(None);
            let tx = Arc::new(tx);
            state.singleflight.lock().insert(family_key.clone(), rx);
            tx
        }
    };

    let tile_cache = state.tile_cache.clone();
    let quality = state.tile_quality;
    let timing = state.timing_breakdown;
    let writer = state.writer.clone();
    let write_root = state.write_generated_dir.clone();
    let buffer_pool = state.buffer_pool.clone();
    let inflight = state.inflight.clone();
    let inflight_limit = state.inflight_limit.clone();
    let grayscale_only = state.grayscale_only;
    let output_format = state.output_format;
    let sharpen = state.sharpen;
    let upsample_filter = state.upsample_filter;
    let singleflight = state.singleflight.clone();
    let family_key_cleanup = family_key.clone();
    let watch_tx_clone = watch_tx.clone();

    let permit = inflight_limit
        .acquire_owned()
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let bytes = task::spawn_blocking(move || {
        let _permit = permit;
        let _inflight = inflight.enter();
        let gen_start = Instant::now();
        let result = generate_family_server(
            &slide, x2, y2, quality, timing, grayscale_only, output_format,
            sharpen, upsample_filter, &writer, &write_root, &buffer_pool,
        );
        match result {
            Ok(result) => {
                let family = result.tiles;
                if let Some(ref tc) = tile_cache {
                    tc.put_family(&family);
                }
                let family_ms = gen_start.elapsed().as_millis();
                if let Some(stats) = result.stats {
                    info!(
                        "family_breakdown slide_id={} x2={} y2={} l2_decode={}ms upsample={}ms l0_res={}ms l0_enc={}ms l1_ds={}ms l1_enc={}ms total={}ms l0_par={}",
                        slide.slide_id, x2, y2,
                        stats.l2_decode_ms, stats.upsample_ms,
                        stats.l0_residual_ms, stats.l0_encode_ms,
                        stats.l1_downsample_ms, stats.l1_encode_ms,
                        stats.total_ms, stats.l0_parallel_max
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
                // Signal success to waiters
                let _ = watch_tx_clone.send(Some(true));
                singleflight.lock().remove(&family_key_cleanup);

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
                // Signal failure to waiters
                let _ = watch_tx_clone.send(Some(false));
                singleflight.lock().remove(&family_key_cleanup);

                let path = baseline_tile_path(&slide, level, x, y);
                let bytes = read_baseline_as_jpeg(&path)?;
                Ok((Bytes::from(bytes), None, true))
            }
        }
    })
    .await
    .map_err(|e| {
        // Task panicked — clean up singleflight entry and notify waiters
        let _ = watch_tx.send(Some(false));
        state.singleflight.lock().remove(&family_key);
        info!("family_panic slide_id={} x2={} y2={} err={}", slide_id, x2, y2, e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?
    .map_err(|_: anyhow::Error| StatusCode::INTERNAL_SERVER_ERROR)?;

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
    let family_key = FamilyKey {
        slide_id: slide.slide_id.clone(),
        x2,
        y2,
    };

    // Check if this family is already in-flight — if so, skip prewarm
    {
        let map = state.singleflight.lock();
        if map.contains_key(&family_key) {
            return;
        }
    }

    // Register in singleflight so explicit tile requests can wait on us
    let (watch_tx, watch_rx) = tokio::sync::watch::channel(None);
    let watch_tx = Arc::new(watch_tx);
    {
        let mut map = state.singleflight.lock();
        // Double-check after acquiring lock
        if map.contains_key(&family_key) {
            return;
        }
        map.insert(family_key.clone(), watch_rx);
    }

    let tile_cache = state.tile_cache.clone();
    let quality = state.tile_quality;
    let grayscale_only = state.grayscale_only;
    let output_format = state.output_format;
    let sharpen = state.sharpen;
    let upsample_filter = state.upsample_filter;
    let writer = state.writer.clone();
    let write_root = state.write_generated_dir.clone();
    let buffer_pool = state.buffer_pool.clone();
    let inflight = state.inflight.clone();
    let inflight_limit = state.inflight_limit.clone();
    let singleflight = state.singleflight.clone();
    let family_key_cleanup = family_key;

    tokio::spawn(async move {
        let permit = match inflight_limit.acquire_owned().await {
            Ok(p) => p,
            Err(_) => {
                let _ = watch_tx.send(Some(false));
                singleflight.lock().remove(&family_key_cleanup);
                return;
            }
        };
        let _inflight = inflight.enter();
        let _permit = permit;
        let singleflight2 = singleflight.clone();
        let family_key2 = family_key_cleanup.clone();
        let watch_tx2 = watch_tx.clone();
        let _ = task::spawn_blocking(move || {
            let result = generate_family_server(
                &slide, x2, y2, quality, false, grayscale_only, output_format,
                sharpen, upsample_filter, &writer, &write_root, &buffer_pool,
            );
            match result {
                Ok(result) => {
                    if let Some(ref tc) = tile_cache {
                        tc.put_family(&result.tiles);
                    }
                    let _ = watch_tx2.send(Some(true));
                }
                Err(_) => {
                    let _ = watch_tx2.send(Some(false));
                }
            }
            singleflight2.lock().remove(&family_key2);
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

async fn compare_dynamic(
    State(state): State<AppState>,
) -> Html<String> {
    // Build JSON array of available slides for the dropdowns
    let mut slides: Vec<serde_json::Value> = state.slides.iter().map(|(id, s)| {
        serde_json::json!({
            "id": id,
            "label": format!("{} ({:.0} MB)", s.label, s.disk_size_mb),
        })
    }).collect();
    slides.sort_by(|a, b| a["label"].as_str().unwrap_or("").cmp(b["label"].as_str().unwrap_or("")));
    let slides_json = serde_json::to_string(&slides).unwrap_or_else(|_| "[]".to_string());

    let html = format!(
        r##"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>ORIGAMI Compare</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.0/openseadragon.min.js"
          crossorigin="anonymous" referrerpolicy="no-referrer"></script>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    html, body {{ height: 100%; font-family: ui-sans-serif, system-ui, -apple-system, sans-serif; background: #111; color: #eee; }}
    .container {{ display: flex; height: 100%; }}
    .panel {{ flex: 1; position: relative; border-right: 2px solid #333; }}
    .panel:last-child {{ border-right: none; }}
    .panel .viewer {{ width: 100%; height: 100%; }}
    .picker {{ position: absolute; top: 10px; left: 10px; z-index: 100; }}
    .picker select {{
      background: rgba(0,0,0,0.75); color: #fff; border: 1px solid #555;
      padding: 6px 12px; border-radius: 6px; font-size: 13px; font-weight: 600;
      cursor: pointer; outline: none;
    }}
    .picker select:hover {{ border-color: #888; }}
    .picker select option {{ background: #222; }}
  </style>
</head>
<body>
  <div class="container">
    <div class="panel">
      <div class="picker"><select id="sel-0"></select></div>
      <div id="osd-0" class="viewer"></div>
    </div>
    <div class="panel">
      <div class="picker"><select id="sel-1"></select></div>
      <div id="osd-1" class="viewer"></div>
    </div>
    <div class="panel">
      <div class="picker"><select id="sel-2"></select></div>
      <div id="osd-2" class="viewer"></div>
    </div>
  </div>
  <script>
    var slides = {slides_json};
    var viewers = [null, null, null];
    var currentIds = [null, null, null];

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

    // Populate dropdowns
    for (var i = 0; i < 3; i++) {{
      var sel = document.getElementById('sel-' + i);
      slides.forEach(function(s) {{
        var opt = document.createElement('option');
        opt.value = s.id;
        opt.textContent = s.label;
        sel.appendChild(opt);
      }});
    }}

    // Pick defaults: try to assign different slides to each panel
    var defaults = slides.map(function(s) {{ return s.id; }});
    for (var i = 0; i < 3; i++) {{
      var sel = document.getElementById('sel-' + i);
      if (defaults[i]) sel.value = defaults[i];
    }}

    function createViewer(idx) {{
      var id = document.getElementById('sel-' + idx).value;
      if (currentIds[idx] === id && viewers[idx]) return;

      // Save viewport state from an existing viewer
      var savedCenter = null, savedZoom = null;
      for (var j = 0; j < 3; j++) {{
        if (viewers[j]) {{
          savedCenter = viewers[j].viewport.getCenter(false);
          savedZoom = viewers[j].viewport.getZoom(false);
          break;
        }}
      }}

      if (viewers[idx]) viewers[idx].destroy();
      currentIds[idx] = id;
      var v = OpenSeadragon(Object.assign({{
        id: 'osd-' + idx,
        showNavigator: idx === 0,
        tileSources: '/dzi/' + id + '.dzi',
      }}, osdOpts));

      if (savedCenter && savedZoom) {{
        v.addOnceHandler('open', function() {{
          v.viewport.panTo(savedCenter, true);
          v.viewport.zoomTo(savedZoom, null, true);
        }});
      }}

      v.addHandler('zoom', function() {{ syncFrom(v); }});
      v.addHandler('pan',  function() {{ syncFrom(v); }});
      viewers[idx] = v;
    }}

    var syncing = false;
    function syncFrom(src) {{
      if (syncing) return;
      syncing = true;
      var center = src.viewport.getCenter(false);
      var zoom = src.viewport.getZoom(false);
      for (var i = 0; i < 3; i++) {{
        if (viewers[i] && viewers[i] !== src) {{
          viewers[i].viewport.panTo(center, false);
          viewers[i].viewport.zoomTo(zoom, null, false);
        }}
      }}
      syncing = false;
    }}

    // Wire up dropdown changes
    for (var i = 0; i < 3; i++) {{
      (function(idx) {{
        document.getElementById('sel-' + idx).addEventListener('change', function() {{
          createViewer(idx);
        }});
      }})(i);
    }}

    // Initialize all three viewers
    for (var i = 0; i < 3; i++) createViewer(i);
  </script>
</body>
</html>"##,
        slides_json = slides_json,
    );

    info!("compare_dynamic viewer requested");
    Html(html)
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
    // Check for JXL first (L2 may be stored as JXL for space savings)
    let jxl = slide.files_dir.join(level.to_string()).join(format!("{}_{}.jxl", x, y));
    if jxl.exists() {
        return jxl;
    }
    slide
        .files_dir
        .join(level.to_string())
        .join(format!("{}_{}.jpg", x, y))
}

/// Ensure L2 bytes from a pack are servable as JPEG.
/// If the bytes are already JPEG (0xFF 0xD8 magic), return as-is.
/// If JXL, reconstruct or transcode to JPEG.
fn ensure_l2_jpeg(data: &[u8]) -> anyhow::Result<Vec<u8>> {
    // JPEG magic: 0xFF 0xD8
    if data.len() >= 2 && data[0] == 0xFF && data[1] == 0xD8 {
        return Ok(data.to_vec());
    }
    // JXL magic
    let is_jxl = (data.len() >= 2 && data[0] == 0xFF && data[1] == 0x0A)
        || (data.len() >= 12
            && data[..12]
                == [0x00, 0x00, 0x00, 0x0C, 0x4A, 0x58, 0x4C, 0x20, 0x0D, 0x0A, 0x87, 0x0A]);
    if is_jxl {
        return ensure_l2_jxl_to_jpeg(data);
    }
    anyhow::bail!("L2 pack data has unknown format (not JPEG or JXL)")
}

#[cfg(feature = "jpegxl")]
fn ensure_l2_jxl_to_jpeg(data: &[u8]) -> anyhow::Result<Vec<u8>> {
    let decoder = jpegxl_rs::decode::decoder_builder()
        .build()
        .map_err(|e| anyhow::anyhow!("JXL decoder build failed: {e:?}"))?;
    let (_meta, result) = decoder.reconstruct(data)
        .map_err(|e| anyhow::anyhow!("JXL reconstruct failed: {e:?}"))?;
    match result {
        jpegxl_rs::decode::Data::Jpeg(jpeg_bytes) => Ok(jpeg_bytes),
        jpegxl_rs::decode::Data::Pixels(pixels) => {
            let raw = match pixels {
                jpegxl_rs::decode::Pixels::Uint8(v) => v,
                _ => anyhow::bail!("JXL L2 decoded to non-u8 pixel type"),
            };
            let w = _meta.width as usize;
            let h = _meta.height as usize;
            let mut compressor = turbojpeg::Compressor::new()
                .map_err(|e| anyhow::anyhow!("turbojpeg compressor failed: {e}"))?;
            compressor.set_quality(95);
            compressor.set_subsamp(turbojpeg::Subsamp::None);
            let image = turbojpeg::Image {
                pixels: raw.as_slice(),
                width: w,
                pitch: w * 3,
                height: h,
                format: turbojpeg::PixelFormat::RGB,
            };
            let jpeg = compressor.compress_to_vec(image)
                .map_err(|e| anyhow::anyhow!("JPEG re-encode failed: {e}"))?;
            Ok(jpeg)
        }
    }
}

#[cfg(not(feature = "jpegxl"))]
fn ensure_l2_jxl_to_jpeg(_data: &[u8]) -> anyhow::Result<Vec<u8>> {
    anyhow::bail!("L2 in bundle is JXL but built without --features jpegxl");
}

/// Read a baseline tile and return JPEG bytes.
/// If the file is `.jxl`, reconstruct the original JPEG (lossless transcode)
/// or decode to pixels and re-encode as JPEG (lossy JXL).
fn read_baseline_as_jpeg(path: &Path) -> anyhow::Result<Vec<u8>> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    if ext == "jxl" {
        read_jxl_as_jpeg(path)
    } else {
        Ok(fs::read(path)?)
    }
}

#[cfg(feature = "jpegxl")]
fn read_jxl_as_jpeg(path: &Path) -> anyhow::Result<Vec<u8>> {
    let data = fs::read(path)?;
    let decoder = jpegxl_rs::decode::decoder_builder()
        .build()
        .map_err(|e| anyhow::anyhow!("JXL decoder build failed: {e:?}"))?;
    let (_meta, result) = decoder.reconstruct(&data)
        .map_err(|e| anyhow::anyhow!("JXL reconstruct failed: {e:?}"))?;

    match result {
        jpegxl_rs::decode::Data::Jpeg(jpeg_bytes) => {
            // Lossless path: bit-identical original JPEG
            Ok(jpeg_bytes)
        }
        jpegxl_rs::decode::Data::Pixels(pixels) => {
            // Lossy path: decode to RGB, re-encode as JPEG
            let raw = match pixels {
                jpegxl_rs::decode::Pixels::Uint8(v) => v,
                _ => anyhow::bail!("JXL L2 decoded to non-u8 pixel type"),
            };
            let w = _meta.width as usize;
            let h = _meta.height as usize;
            let mut compressor = turbojpeg::Compressor::new()
                .map_err(|e| anyhow::anyhow!("turbojpeg compressor failed: {e}"))?;
            compressor.set_quality(95);
            compressor.set_subsamp(turbojpeg::Subsamp::None);
            let image = turbojpeg::Image {
                pixels: raw.as_slice(),
                width: w,
                pitch: w * 3,
                height: h,
                format: turbojpeg::PixelFormat::RGB,
            };
            let jpeg = compressor.compress_to_vec(image)
                .map_err(|e| anyhow::anyhow!("JPEG re-encode failed: {e}"))?;
            Ok(jpeg)
        }
    }
}

#[cfg(not(feature = "jpegxl"))]
fn read_jxl_as_jpeg(path: &Path) -> anyhow::Result<Vec<u8>> {
    anyhow::bail!("JXL baseline tile at {} but built without --features jpegxl", path.display());
}

// residual_tile_path removed in v2 pipeline (no per-tile loose residuals)

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
    _residuals_dir: &str, // unused in v2 pipeline
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
                } else if summary.get("baseq").is_some() {
                    // baseq may be a number or a string (v2 pipeline)
                    let baseq = summary.get("baseq")
                        .and_then(|v| v.as_u64().or_else(|| v.as_str().and_then(|s| s.parse().ok())))
                        .unwrap_or(0);
                    let l0q = summary.get("l0q").and_then(|v| v.as_u64()).unwrap_or(0);
                    format!("Origami {}/{}", baseq, l0q)
                } else if summary.get("residual_jpeg_q_L").is_some() {
                    // Legacy Python pipeline format
                    let q = summary.get("residual_jpeg_q_L").and_then(|v| v.as_u64()).unwrap_or(0);
                    format!("Origami Q{}", q)
                } else {
                    slide_id.clone()
                };
                let size = if let Some(total) = summary.get("total_bytes").and_then(|v| v.as_u64()) {
                    // V2 pipeline and ingest both use total_bytes
                    total as f64 / 1_048_576.0
                } else if summary.get("l2_bytes").is_some() {
                    let l2 = summary.get("l2_bytes").and_then(|v| v.as_u64()).unwrap_or(0);
                    let res = summary.get("residual_bytes").and_then(|v| v.as_u64()).unwrap_or(0);
                    (l2 + res) as f64 / 1_048_576.0
                } else if summary.get("proposed_bytes").is_some() {
                    // Legacy Python pipeline format
                    let total = summary.get("proposed_bytes").and_then(|v| v.as_u64()).unwrap_or(0);
                    total as f64 / 1_048_576.0
                } else {
                    0.0
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
    sharpen: Option<f32>,
    upsample_filter: ResampleFilter,
    writer: &Option<mpsc::Sender<WriteJob>>,
    write_root: &Option<PathBuf>,
    buffer_pool: &BufferPool,
) -> Result<ServerFamilyResult> {
    let input = ReconstructInput {
        files_dir: &slide.files_dir,
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
        sharpen,
        upsample_filter,
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

    Ok(ServerFamilyResult {
        tiles: out,
        stats: core_result.stats,
    })
}
