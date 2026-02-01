// OPTIMIZATION 4: Lock-free metrics using atomics
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

pub struct LockFreeMetrics {
    // Counters
    pub tile_total: AtomicU64,
    pub tile_baseline: AtomicU64,
    pub tile_cache_hit: AtomicU64,
    pub tile_generated: AtomicU64,
    pub tile_fallback: AtomicU64,
    pub tile_residual_view: AtomicU64,
    pub family_generated: AtomicU64,

    // Timings (stored as microseconds to fit in u64)
    pub tile_latency_sum_us: AtomicU64,
    pub tile_latency_max_us: AtomicU64,
    pub family_latency_sum_us: AtomicU64,
    pub family_latency_max_us: AtomicU64,
}

impl LockFreeMetrics {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            tile_total: AtomicU64::new(0),
            tile_baseline: AtomicU64::new(0),
            tile_cache_hit: AtomicU64::new(0),
            tile_generated: AtomicU64::new(0),
            tile_fallback: AtomicU64::new(0),
            tile_residual_view: AtomicU64::new(0),
            family_generated: AtomicU64::new(0),
            tile_latency_sum_us: AtomicU64::new(0),
            tile_latency_max_us: AtomicU64::new(0),
            family_latency_sum_us: AtomicU64::new(0),
            family_latency_max_us: AtomicU64::new(0),
        })
    }

    #[inline(always)]
    pub fn record_tile(&self, kind: &str, latency_ms: u128) {
        self.tile_total.fetch_add(1, Ordering::Relaxed);

        // Record specific counter
        match kind {
            "baseline" => self.tile_baseline.fetch_add(1, Ordering::Relaxed),
            "cache_hit" => self.tile_cache_hit.fetch_add(1, Ordering::Relaxed),
            "generated" => self.tile_generated.fetch_add(1, Ordering::Relaxed),
            "fallback" => self.tile_fallback.fetch_add(1, Ordering::Relaxed),
            "residual_view" => self.tile_residual_view.fetch_add(1, Ordering::Relaxed),
            _ => 0,
        };

        // Record timing (convert ms to us for precision)
        let latency_us = (latency_ms * 1000) as u64;
        self.tile_latency_sum_us.fetch_add(latency_us, Ordering::Relaxed);

        // Update max using CAS loop
        let mut current = self.tile_latency_max_us.load(Ordering::Relaxed);
        while latency_us > current {
            match self.tile_latency_max_us.compare_exchange_weak(
                current,
                latency_us,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => current = x,
            }
        }
    }

    #[inline(always)]
    pub fn record_family(&self, latency_ms: u128) {
        self.family_generated.fetch_add(1, Ordering::Relaxed);

        let latency_us = (latency_ms * 1000) as u64;
        self.family_latency_sum_us.fetch_add(latency_us, Ordering::Relaxed);

        // Update max
        let mut current = self.family_latency_max_us.load(Ordering::Relaxed);
        while latency_us > current {
            match self.family_latency_max_us.compare_exchange_weak(
                current,
                latency_us,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => current = x,
            }
        }
    }

    pub fn snapshot_and_reset(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            tile_total: self.tile_total.swap(0, Ordering::Relaxed),
            tile_baseline: self.tile_baseline.swap(0, Ordering::Relaxed),
            tile_cache_hit: self.tile_cache_hit.swap(0, Ordering::Relaxed),
            tile_generated: self.tile_generated.swap(0, Ordering::Relaxed),
            tile_fallback: self.tile_fallback.swap(0, Ordering::Relaxed),
            tile_residual_view: self.tile_residual_view.swap(0, Ordering::Relaxed),
            family_generated: self.family_generated.swap(0, Ordering::Relaxed),
            tile_latency_sum_us: self.tile_latency_sum_us.swap(0, Ordering::Relaxed),
            tile_latency_max_us: self.tile_latency_max_us.swap(0, Ordering::Relaxed),
            family_latency_sum_us: self.family_latency_sum_us.swap(0, Ordering::Relaxed),
            family_latency_max_us: self.family_latency_max_us.swap(0, Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    pub tile_total: u64,
    pub tile_baseline: u64,
    pub tile_cache_hit: u64,
    pub tile_generated: u64,
    pub tile_fallback: u64,
    pub tile_residual_view: u64,
    pub family_generated: u64,
    pub tile_latency_sum_us: u64,
    pub tile_latency_max_us: u64,
    pub family_latency_sum_us: u64,
    pub family_latency_max_us: u64,
}

impl MetricsSnapshot {
    pub fn tile_avg_ms(&self) -> f64 {
        if self.tile_total > 0 {
            (self.tile_latency_sum_us as f64 / self.tile_total as f64) / 1000.0
        } else {
            0.0
        }
    }

    pub fn tile_max_ms(&self) -> f64 {
        self.tile_latency_max_us as f64 / 1000.0
    }

    pub fn family_avg_ms(&self) -> f64 {
        if self.family_generated > 0 {
            (self.family_latency_sum_us as f64 / self.family_generated as f64) / 1000.0
        } else {
            0.0
        }
    }

    pub fn family_max_ms(&self) -> f64 {
        self.family_latency_max_us as f64 / 1000.0
    }
}