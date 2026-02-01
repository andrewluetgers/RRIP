pub mod optimized;
pub mod metrics_optimized;
// pub mod parallel_generation;  // Will integrate directly into main.rs

// Re-exports for easy access
pub use optimized::{FixedBufferPool, TieredBufferPools, TurboJpegPool};
pub use metrics_optimized::{LockFreeMetrics, MetricsSnapshot};
// pub use parallel_generation::generate_family_optimized;