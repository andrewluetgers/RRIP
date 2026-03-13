//! Wavelet-domain residual denoising and noise synthesis.
//!
//! Implements configurable wavelet basis (default: db4) 2-level DWT soft thresholding
//! (VisuShrink) for separating structure from noise in centered residual images,
//! plus decode-time noise synthesis from 16-byte synthesis parameters stored in
//! the pack file.
//!
//! Algorithm:
//!   Encode: residual → DWT → MAD σ estimate → soft threshold → IDWT → denoised residual
//!           Also: measure removed noise per-subband σ → pack as SynthesisParams (16 bytes)
//!   Decode: SynthesisParams → sample Laplacian per subband → IDWT → add to reconstructed Y

use std::f64::consts::SQRT_2;
use std::fmt;
use std::str::FromStr;

// ---------------------------------------------------------------------------
// Wavelet basis selection
// ---------------------------------------------------------------------------

/// Available wavelet bases. Default is Db4 (Daubechies-4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaveletBasis {
    /// Daubechies-2 (4 taps) — shortest, fastest, less frequency selectivity
    Db2,
    /// Daubechies-4 (8 taps) — good balance of speed and quality (default)
    Db4,
    /// Daubechies-6 (12 taps) — better frequency selectivity, slower
    Db6,
    /// Symlet-4 (8 taps) — near-symmetric variant of db4
    Sym4,
    /// Coiflet-2 (12 taps) — near-symmetric with vanishing moments
    Coif2,
}

impl Default for WaveletBasis {
    fn default() -> Self {
        WaveletBasis::Db4
    }
}

impl fmt::Display for WaveletBasis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WaveletBasis::Db2 => write!(f, "db2"),
            WaveletBasis::Db4 => write!(f, "db4"),
            WaveletBasis::Db6 => write!(f, "db6"),
            WaveletBasis::Sym4 => write!(f, "sym4"),
            WaveletBasis::Coif2 => write!(f, "coif2"),
        }
    }
}

impl FromStr for WaveletBasis {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "db2" => Ok(WaveletBasis::Db2),
            "db4" => Ok(WaveletBasis::Db4),
            "db6" => Ok(WaveletBasis::Db6),
            "sym4" => Ok(WaveletBasis::Sym4),
            "coif2" => Ok(WaveletBasis::Coif2),
            _ => Err(format!(
                "unknown wavelet '{}'. Available: db2, db4, db6, sym4, coif2", s
            )),
        }
    }
}

/// Wavelet filter bank: decomposition and reconstruction filters.
struct FilterBank {
    lo_d: &'static [f64],
    hi_d: Vec<f64>,
    lo_r: Vec<f64>,
    hi_r: Vec<f64>,
}

impl WaveletBasis {
    /// Get the filter bank for this wavelet basis.
    fn filters(&self) -> FilterBank {
        let lo_d: &'static [f64] = match self {
            WaveletBasis::Db2 => &DB2_LO_D,
            WaveletBasis::Db4 => &DB4_LO_D,
            WaveletBasis::Db6 => &DB6_LO_D,
            WaveletBasis::Sym4 => &SYM4_LO_D,
            WaveletBasis::Coif2 => &COIF2_LO_D,
        };
        // Derive other filters from lo_d:
        // hi_d[k] = (-1)^k * lo_d[N-1-k]  (alternating flip)
        // lo_r = reverse(lo_d)
        // hi_r = reverse(hi_d)
        let n = lo_d.len();
        let hi_d: Vec<f64> = (0..n)
            .map(|k| if k % 2 == 0 { -lo_d[n - 1 - k] } else { lo_d[n - 1 - k] })
            .collect();
        let lo_r: Vec<f64> = lo_d.iter().rev().copied().collect();
        let hi_r: Vec<f64> = hi_d.iter().rev().copied().collect();
        FilterBank { lo_d, hi_d, lo_r, hi_r }
    }

    /// Map to a u8 for pack storage (fits in the SynthesisParams seed field's spare bits,
    /// or could use a dedicated byte). Currently we encode this in the top 3 bits of seed.
    pub fn to_id(&self) -> u8 {
        match self {
            WaveletBasis::Db2 => 0,
            WaveletBasis::Db4 => 1,
            WaveletBasis::Db6 => 2,
            WaveletBasis::Sym4 => 3,
            WaveletBasis::Coif2 => 4,
        }
    }

    /// Reconstruct from id byte.
    pub fn from_id(id: u8) -> Self {
        match id {
            0 => WaveletBasis::Db2,
            1 => WaveletBasis::Db4,
            2 => WaveletBasis::Db6,
            3 => WaveletBasis::Sym4,
            4 => WaveletBasis::Coif2,
            _ => WaveletBasis::Db4, // fallback
        }
    }
}

// ---------------------------------------------------------------------------
// Filter coefficient tables (from PyWavelets / standard reference)
// ---------------------------------------------------------------------------

// Coefficients from PyWavelets (pywt.Wavelet(name).dec_lo)
static DB2_LO_D: [f64; 4] = [
    -0.12940952255126037,
     0.2241438680420134,
     0.8365163037378079,
     0.48296291314453416,
];

static DB4_LO_D: [f64; 8] = [
    -0.010597401785069032,
     0.0328830116668852,
     0.030841381835560764,
    -0.18703481171909309,
    -0.027983769416859854,
     0.6308807679298589,
     0.7148465705529157,
     0.2303778133088965,
];

static DB6_LO_D: [f64; 12] = [
    -0.0010773010853084796,
     0.004777257510945511,
     0.0005538422011614961,
    -0.03158203931748603,
     0.027522865530305727,
     0.09750160558732304,
    -0.12976686756726194,
    -0.22626469396543983,
     0.31525035170919763,
     0.7511339080210954,
     0.49462389039845306,
     0.11154074335010947,
];

static SYM4_LO_D: [f64; 8] = [
    -0.07576571478927333,
    -0.02963552764599851,
     0.49761866763201545,
     0.8037387518059161,
     0.29785779560527736,
    -0.09921954357684722,
    -0.012603967262037833,
     0.0322231006040427,
];

static COIF2_LO_D: [f64; 12] = [
    -0.000720549445520347,
    -0.0018232088709110323,
     0.005611434819368834,
     0.02368017194684777,
    -0.05943441864643109,
    -0.07648859907828076,
     0.4170051844232391,
     0.8127236354494135,
     0.3861100668227629,
    -0.0673725547237256,
    -0.04146493678687178,
     0.01638733646320364,
];

// ---------------------------------------------------------------------------
// 1D convolution + downsampling / upsampling + convolution
// ---------------------------------------------------------------------------

/// Analysis: convolve with reversed filter and downsample by 2, periodic extension.
///
/// Implements: periodically extend signal, convolve with time-reversed filter
/// (i.e. cross-correlate with filter), take every 2nd sample starting at flen-1.
///
/// cA[n] = Σ_k h[flen-1-k] * x_ext[2n + k]  for k = 0..flen-1
///       = Σ_k h[k] * x_ext[2n + flen - 1 - k]
fn convolve_downsample(signal: &[f64], filter: &[f64]) -> Vec<f64> {
    let n = signal.len();
    let flen = filter.len();
    let out_len = (n + 1) / 2;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let mut sum = 0.0;
        for k in 0..flen {
            // Cross-correlation: filter[k] * signal[(2i + k) mod n]
            // But we want convolution, so use filter reversed:
            // conv[i] = Σ_k filter[flen-1-k] * signal[(2i + k) mod n]
            //         = Σ_k filter[k] * signal[(2i + flen - 1 - k) mod n]
            let idx = (2 * i + flen - 1 - k) % n;
            sum += filter[k] * signal[idx];
        }
        out.push(sum);
    }
    out
}

/// Synthesis: upsample by 2, then convolve with filter, periodic extension.
///
/// Creates upsampled signal u[j] = coeffs[j/2] if j even, 0 if odd (period = 2*nc).
/// Then convolves: x[n] = Σ_j filter[j] * u[(n - j) mod N]
/// Only j where (n-j) is even contribute: (n-j) = 2m → j = n - 2m
fn upsample_convolve(coeffs: &[f64], filter: &[f64], output_len: usize) -> Vec<f64> {
    let nc = coeffs.len();
    let flen = filter.len();
    let mut out = vec![0.0; output_len];
    // x[n] = Σ_j filter[j] * u[(n-j) mod (2*nc)]
    // u[2m] = coeffs[m], u[2m+1] = 0
    // So only terms where (n-j) mod (2*nc) is even contribute.
    // Let (n-j) mod (2*nc) = 2m, so j = n - 2m (mod 2*nc), and filter index j must be in [0, flen)
    for n_idx in 0..output_len {
        let mut sum = 0.0;
        for m in 0..nc {
            let j = (n_idx as isize - 2 * m as isize).rem_euclid(2 * nc as isize) as usize;
            if j < flen {
                sum += filter[j] * coeffs[m];
            }
        }
        out[n_idx] = sum;
    }
    out
}

// ---------------------------------------------------------------------------
// 2D DWT (rows then columns)
// ---------------------------------------------------------------------------

/// Apply 1-level 2D DWT decomposition.
fn dwt2_one_level(
    data: &[f64], rows: usize, cols: usize,
    fb: &FilterBank,
) -> (Vec<f64>, (Vec<f64>, Vec<f64>, Vec<f64>), usize, usize) {
    let half_cols = (cols + 1) / 2;
    let half_rows = (rows + 1) / 2;

    // Filter rows
    let mut lo_rows = vec![0.0; rows * half_cols];
    let mut hi_rows = vec![0.0; rows * half_cols];
    for r in 0..rows {
        let row = &data[r * cols..(r + 1) * cols];
        let lo = convolve_downsample(row, fb.lo_d);
        let hi = convolve_downsample(row, &fb.hi_d);
        lo_rows[r * half_cols..(r + 1) * half_cols].copy_from_slice(&lo);
        hi_rows[r * half_cols..(r + 1) * half_cols].copy_from_slice(&hi);
    }

    // Filter columns
    let mut approx = vec![0.0; half_rows * half_cols];
    let mut detail_h = vec![0.0; half_rows * half_cols];
    let mut detail_v = vec![0.0; half_rows * half_cols];
    let mut detail_d = vec![0.0; half_rows * half_cols];

    for c in 0..half_cols {
        let col_lo: Vec<f64> = (0..rows).map(|r| lo_rows[r * half_cols + c]).collect();
        let lo_lo = convolve_downsample(&col_lo, fb.lo_d);
        let hi_lo = convolve_downsample(&col_lo, &fb.hi_d);

        let col_hi: Vec<f64> = (0..rows).map(|r| hi_rows[r * half_cols + c]).collect();
        let lo_hi = convolve_downsample(&col_hi, fb.lo_d);
        let hi_hi = convolve_downsample(&col_hi, &fb.hi_d);

        for r in 0..half_rows {
            approx[r * half_cols + c] = lo_lo[r];
            detail_h[r * half_cols + c] = hi_lo[r];
            detail_v[r * half_cols + c] = lo_hi[r];
            detail_d[r * half_cols + c] = hi_hi[r];
        }
    }

    (approx, (detail_h, detail_v, detail_d), half_rows, half_cols)
}

/// Apply 1-level 2D inverse DWT reconstruction.
fn idwt2_one_level(
    approx: &[f64], detail_h: &[f64], detail_v: &[f64], detail_d: &[f64],
    half_rows: usize, half_cols: usize,
    out_rows: usize, out_cols: usize,
    fb: &FilterBank,
) -> Vec<f64> {
    let mut lo_rows = vec![0.0; out_rows * half_cols];
    let mut hi_rows = vec![0.0; out_rows * half_cols];

    for c in 0..half_cols {
        let col_ll: Vec<f64> = (0..half_rows).map(|r| approx[r * half_cols + c]).collect();
        let col_lh: Vec<f64> = (0..half_rows).map(|r| detail_h[r * half_cols + c]).collect();
        let col_hl: Vec<f64> = (0..half_rows).map(|r| detail_v[r * half_cols + c]).collect();
        let col_hh: Vec<f64> = (0..half_rows).map(|r| detail_d[r * half_cols + c]).collect();

        let up_ll = upsample_convolve(&col_ll, &fb.lo_r, out_rows);
        let up_lh = upsample_convolve(&col_lh, &fb.hi_r, out_rows);
        let up_hl = upsample_convolve(&col_hl, &fb.lo_r, out_rows);
        let up_hh = upsample_convolve(&col_hh, &fb.hi_r, out_rows);

        for r in 0..out_rows {
            lo_rows[r * half_cols + c] = up_ll[r] + up_lh[r];
            hi_rows[r * half_cols + c] = up_hl[r] + up_hh[r];
        }
    }

    let mut output = vec![0.0; out_rows * out_cols];
    for r in 0..out_rows {
        let lo_row = &lo_rows[r * half_cols..(r + 1) * half_cols];
        let hi_row = &hi_rows[r * half_cols..(r + 1) * half_cols];
        let up_lo = upsample_convolve(lo_row, &fb.lo_r, out_cols);
        let up_hi = upsample_convolve(hi_row, &fb.hi_r, out_cols);
        for c in 0..out_cols {
            output[r * out_cols + c] = up_lo[c] + up_hi[c];
        }
    }

    output
}

// ---------------------------------------------------------------------------
// Multi-level DWT
// ---------------------------------------------------------------------------

/// DWT coefficients for a multi-level decomposition.
pub struct DwtCoeffs {
    pub approx: Vec<f64>,
    pub approx_rows: usize,
    pub approx_cols: usize,
    /// Detail coefficients per level: (detail_h, detail_v, detail_d, rows, cols).
    /// Index 0 = first decomposition (coarsest detail).
    pub details: Vec<(Vec<f64>, Vec<f64>, Vec<f64>, usize, usize)>,
    /// Original dimensions at each level for reconstruction.
    pub sizes: Vec<(usize, usize)>,
}

/// Multi-level 2D DWT decomposition with configurable wavelet basis.
pub fn wavedec2(data: &[f64], rows: usize, cols: usize, level: usize, basis: WaveletBasis) -> DwtCoeffs {
    let fb = basis.filters();
    let mut sizes = vec![(rows, cols)];
    let mut details = Vec::with_capacity(level);
    let mut current = data.to_vec();
    let mut cur_rows = rows;
    let mut cur_cols = cols;

    for _ in 0..level {
        let (approx, (dh, dv, dd), hr, hc) = dwt2_one_level(&current, cur_rows, cur_cols, &fb);
        details.push((dh, dv, dd, hr, hc));
        sizes.push((hr, hc));
        current = approx;
        cur_rows = hr;
        cur_cols = hc;
    }

    DwtCoeffs {
        approx: current,
        approx_rows: cur_rows,
        approx_cols: cur_cols,
        details,
        sizes,
    }
}

/// Multi-level 2D inverse DWT reconstruction with configurable wavelet basis.
pub fn waverec2(coeffs: &DwtCoeffs, basis: WaveletBasis) -> Vec<f64> {
    let fb = basis.filters();
    let mut current = coeffs.approx.clone();
    let mut cur_rows = coeffs.approx_rows;
    let mut cur_cols = coeffs.approx_cols;

    for i in (0..coeffs.details.len()).rev() {
        let (ref dh, ref dv, ref dd, _hr, _hc) = coeffs.details[i];
        let (out_rows, out_cols) = coeffs.sizes[i];
        current = idwt2_one_level(&current, dh, dv, dd, cur_rows, cur_cols, out_rows, out_cols, &fb);
        cur_rows = out_rows;
        cur_cols = out_cols;
    }

    current
}

// ---------------------------------------------------------------------------
// Noise estimation and soft thresholding
// ---------------------------------------------------------------------------

/// Estimate noise sigma via MAD on finest-level horizontal detail coefficients.
fn estimate_sigma_mad(coeffs: &DwtCoeffs) -> f64 {
    let finest = &coeffs.details[coeffs.details.len() - 1];
    let dh = &finest.0;
    let mut abs_vals: Vec<f64> = dh.iter().map(|v| v.abs()).collect();
    abs_vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let median = if abs_vals.is_empty() {
        0.0
    } else if abs_vals.len() % 2 == 0 {
        (abs_vals[abs_vals.len() / 2 - 1] + abs_vals[abs_vals.len() / 2]) / 2.0
    } else {
        abs_vals[abs_vals.len() / 2]
    };
    median / 0.6745
}

/// Soft threshold: sign(x) * max(|x| - t, 0)
fn soft_threshold(coeffs: &mut [f64], threshold: f64) {
    for v in coeffs.iter_mut() {
        let abs = v.abs();
        if abs <= threshold {
            *v = 0.0;
        } else {
            *v = v.signum() * (abs - threshold);
        }
    }
}

// ---------------------------------------------------------------------------
// Synthesis parameters (16 bytes packed into pack file)
// ---------------------------------------------------------------------------

/// Noise synthesis parameters stored in the pack file (16 bytes).
///
/// Layout: 7 x f16 sigmas (14 bytes) + 1 x u8 wavelet_id + 1 x u8 seed_lo = 16 bytes
///   [0..2]   approx_sigma (f16)
///   [2..4]   level1_h sigma (f16)   — coarsest detail level
///   [4..6]   level1_v sigma (f16)
///   [6..8]   level1_d sigma (f16)
///   [8..10]  level2_h sigma (f16)   — finest detail level
///   [10..12] level2_v sigma (f16)
///   [12..14] level2_d sigma (f16)
///   [14]     wavelet_id (u8): 0=db2, 1=db4, 2=db6, 3=sym4, 4=coif2
///   [15]     seed (u8)
#[derive(Debug, Clone, Copy)]
pub struct SynthesisParams {
    pub approx_sigma: f32,
    /// Per-level subband sigmas: [level][h,v,d]. Index 0 = coarsest.
    pub subband_sigmas: [[f32; 3]; 2],
    pub basis: WaveletBasis,
    pub seed: u8,
}

impl SynthesisParams {
    /// Serialize to 16 bytes for pack storage.
    pub fn to_bytes(&self) -> [u8; 16] {
        let mut buf = [0u8; 16];
        buf[0..2].copy_from_slice(&half::f16::from_f32(self.approx_sigma).to_le_bytes());
        buf[2..4].copy_from_slice(&half::f16::from_f32(self.subband_sigmas[0][0]).to_le_bytes());
        buf[4..6].copy_from_slice(&half::f16::from_f32(self.subband_sigmas[0][1]).to_le_bytes());
        buf[6..8].copy_from_slice(&half::f16::from_f32(self.subband_sigmas[0][2]).to_le_bytes());
        buf[8..10].copy_from_slice(&half::f16::from_f32(self.subband_sigmas[1][0]).to_le_bytes());
        buf[10..12].copy_from_slice(&half::f16::from_f32(self.subband_sigmas[1][1]).to_le_bytes());
        buf[12..14].copy_from_slice(&half::f16::from_f32(self.subband_sigmas[1][2]).to_le_bytes());
        buf[14] = self.basis.to_id();
        buf[15] = self.seed;
        buf
    }

    /// Deserialize from 16 bytes.
    pub fn from_bytes(buf: &[u8; 16]) -> Self {
        Self {
            approx_sigma: half::f16::from_le_bytes([buf[0], buf[1]]).to_f32(),
            subband_sigmas: [
                [
                    half::f16::from_le_bytes([buf[2], buf[3]]).to_f32(),
                    half::f16::from_le_bytes([buf[4], buf[5]]).to_f32(),
                    half::f16::from_le_bytes([buf[6], buf[7]]).to_f32(),
                ],
                [
                    half::f16::from_le_bytes([buf[8], buf[9]]).to_f32(),
                    half::f16::from_le_bytes([buf[10], buf[11]]).to_f32(),
                    half::f16::from_le_bytes([buf[12], buf[13]]).to_f32(),
                ],
            ],
            basis: WaveletBasis::from_id(buf[14]),
            seed: buf[15],
        }
    }
}

// ---------------------------------------------------------------------------
// Public API: denoise + measure synthesis params
// ---------------------------------------------------------------------------

/// Result of denoising a residual.
pub struct DenoiseResult {
    /// Denoised residual (u8, centered at 128), ready for encoding.
    pub denoised: Vec<u8>,
    /// Noise that was removed (u8, centered at 128), for debug visualization.
    pub removed_noise: Vec<u8>,
    /// Synthesis parameters (16 bytes) for decode-time noise recovery.
    pub synth_params: SynthesisParams,
}

/// Denoise a centered residual (u8, centered at 128) using wavelet soft thresholding
/// with VisuShrink. Wavelet basis and decomposition level are configurable.
///
/// Returns DenoiseResult with denoised residual, removed noise map, and synthesis params.
///
/// `weight` controls how much denoising is applied (0.0 = none, 1.0 = full).
/// At weight=0 the residual is unchanged but synth params are still computed,
/// enabling decode-time noise synthesis without encode-time denoising.
pub fn denoise_residual(
    residual: &[u8],
    width: u32,
    height: u32,
    sigma_multiplier: f32,
    basis: WaveletBasis,
    level: usize,
    weight: f32,
) -> DenoiseResult {
    let w = width as usize;
    let h = height as usize;

    // Convert to centered float
    let centered: Vec<f64> = residual.iter().map(|&v| v as f64 - 128.0).collect();

    // Multi-level DWT decomposition
    let mut coeffs = wavedec2(&centered, h, w, level, basis);

    // Estimate noise sigma via MAD
    let sigma = estimate_sigma_mad(&coeffs);

    // VisuShrink threshold
    let n = (w * h) as f64;
    let threshold = sigma * (2.0 * n.ln()).sqrt() * sigma_multiplier as f64;

    // Soft threshold detail coefficients
    for detail in coeffs.details.iter_mut() {
        soft_threshold(&mut detail.0, threshold);
        soft_threshold(&mut detail.1, threshold);
        soft_threshold(&mut detail.2, threshold);
    }

    // Reconstruct denoised signal
    let denoised_f64 = waverec2(&coeffs, basis);

    // Compute what was removed (noise = original - denoised)
    let noise_f64: Vec<f64> = centered.iter().zip(denoised_f64.iter())
        .map(|(o, d)| o - d).collect();

    // Decompose the removed noise to get per-subband sigmas
    let noise_coeffs = wavedec2(&noise_f64, h, w, level, basis);
    let approx_sigma = std_dev(&noise_coeffs.approx);
    let mut subband_sigmas = [[0.0f32; 3]; 2];
    for (i, detail) in noise_coeffs.details.iter().enumerate() {
        if i < 2 {
            subband_sigmas[i][0] = std_dev(&detail.0) as f32;
            subband_sigmas[i][1] = std_dev(&detail.1) as f32;
            subband_sigmas[i][2] = std_dev(&detail.2) as f32;
        }
    }

    // Blend: output = original * (1 - weight) + denoised * weight
    let w_f64 = (weight as f64).clamp(0.0, 1.0);
    let blended: Vec<f64> = centered.iter().zip(denoised_f64.iter())
        .map(|(&orig, &den)| orig * (1.0 - w_f64) + den * w_f64)
        .collect();

    // Convert back to u8
    let denoised: Vec<u8> = blended.iter()
        .map(|&v| (v + 128.0).clamp(0.0, 255.0) as u8)
        .collect();

    // Removed noise visualization (absolute magnitude, white on black, 4x amplified)
    // Shows what would be fully removed (weight=1), regardless of actual weight
    let removed_noise: Vec<u8> = noise_f64.iter()
        .map(|&v| (v.abs() * 4.0).clamp(0.0, 255.0) as u8)
        .collect();

    let params = SynthesisParams {
        approx_sigma: approx_sigma as f32,
        subband_sigmas,
        basis,
        seed: 42,
    };

    DenoiseResult {
        denoised,
        removed_noise,
        synth_params: params,
    }
}

/// Standard deviation of a slice.
fn std_dev(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let var = data.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / data.len() as f64;
    var.sqrt()
}

// ---------------------------------------------------------------------------
// Public API: noise synthesis (decode-time)
// ---------------------------------------------------------------------------

/// Synthesize noise from SynthesisParams and add it to the reconstructed Y plane.
///
/// Called at decode time after applying the (denoised) residual. Generates
/// wavelet-domain noise with Laplacian-distributed coefficients matched to
/// per-subband sigmas, then inverse-DWTs to get spatially correlated noise.
///
/// `strength` controls what fraction of the measured noise is synthesized (0.0 = none, 1.0 = full).
/// Default 0.5 is recommended since the denoiser removes some real signal along with noise,
/// so synthesizing 100% back adds too much random energy.
pub fn synthesize_and_apply_noise(
    y_plane: &mut [u8],
    width: u32,
    height: u32,
    params: &SynthesisParams,
    strength: f32,
) {
    if strength <= 0.0 { return; }
    let s = strength as f64;

    let w = width as usize;
    let h = height as usize;
    let basis = params.basis;

    // Create a zero buffer decomposition to get coefficient shapes
    let zeros = vec![0.0f64; w * h];
    let template = wavedec2(&zeros, h, w, 2, basis);

    let mut rng = XorShift64::new(params.seed as u64 ^ 0xDEADBEEF);

    // Generate approximation coefficients (scaled by strength)
    let approx_b = params.approx_sigma as f64 * s / SQRT_2;
    let synth_approx: Vec<f64> = (0..template.approx.len())
        .map(|_| rng.laplace(approx_b))
        .collect();

    // Generate detail coefficients (scaled by strength)
    let mut synth_details = Vec::with_capacity(2);
    for (i, detail) in template.details.iter().enumerate() {
        if i >= 2 { break; }
        let sigmas = &params.subband_sigmas[i];
        let dh: Vec<f64> = (0..detail.0.len())
            .map(|_| rng.laplace(sigmas[0] as f64 * s / SQRT_2))
            .collect();
        let dv: Vec<f64> = (0..detail.1.len())
            .map(|_| rng.laplace(sigmas[1] as f64 * s / SQRT_2))
            .collect();
        let dd: Vec<f64> = (0..detail.2.len())
            .map(|_| rng.laplace(sigmas[2] as f64 * s / SQRT_2))
            .collect();
        synth_details.push((dh, dv, dd, detail.3, detail.4));
    }

    let synth_coeffs = DwtCoeffs {
        approx: synth_approx,
        approx_rows: template.approx_rows,
        approx_cols: template.approx_cols,
        details: synth_details,
        sizes: template.sizes,
    };

    let noise = waverec2(&synth_coeffs, basis);

    // Add noise to Y plane
    for (y_val, n) in y_plane.iter_mut().zip(noise.iter()) {
        let result = *y_val as f64 + n;
        *y_val = result.clamp(0.0, 255.0) as u8;
    }
}

// ---------------------------------------------------------------------------
// Minimal Laplacian PRNG (no external dependency)
// ---------------------------------------------------------------------------

struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self { state: if seed == 0 { 1 } else { seed } }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Laplacian(0, b) sample via inverse CDF.
    fn laplace(&mut self, b: f64) -> f64 {
        if b < 1e-10 {
            return 0.0;
        }
        let u = self.uniform() - 0.5;
        -b * u.signum() * (1.0 - 2.0 * u.abs()).ln()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthesis_params_roundtrip() {
        let params = SynthesisParams {
            approx_sigma: 1.234,
            subband_sigmas: [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]],
            basis: WaveletBasis::Db4,
            seed: 123,
        };
        let bytes = params.to_bytes();
        assert_eq!(bytes.len(), 16);
        let decoded = SynthesisParams::from_bytes(&bytes);
        assert!((decoded.approx_sigma - 1.234).abs() < 0.01);
        assert_eq!(decoded.seed, 123);
        assert_eq!(decoded.basis, WaveletBasis::Db4);
        assert!((decoded.subband_sigmas[1][2] - 7.0).abs() < 0.1);
    }

    #[test]
    fn test_synthesis_params_basis_roundtrip() {
        for basis in [WaveletBasis::Db2, WaveletBasis::Db4, WaveletBasis::Db6,
                      WaveletBasis::Sym4, WaveletBasis::Coif2] {
            let params = SynthesisParams {
                approx_sigma: 1.0,
                subband_sigmas: [[1.0; 3]; 2],
                basis,
                seed: 42,
            };
            let decoded = SynthesisParams::from_bytes(&params.to_bytes());
            assert_eq!(decoded.basis, basis, "basis roundtrip failed for {}", basis);
        }
    }

    #[test]
    fn test_dwt_idwt_roundtrip_db4() {
        let rows = 32;
        let cols = 32;
        let data: Vec<f64> = (0..rows * cols).map(|i| (i as f64).sin() * 50.0).collect();

        let coeffs = wavedec2(&data, rows, cols, 2, WaveletBasis::Db4);
        let reconstructed = waverec2(&coeffs, WaveletBasis::Db4);

        let max_err: f64 = data.iter().zip(reconstructed.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(max_err < 1e-8, "DWT roundtrip error too large: {}", max_err);
    }

    #[test]
    fn test_dwt_idwt_roundtrip_all_bases() {
        // Use 64x64 to ensure signal is large enough for 12-tap filters at 2 levels
        let rows = 64;
        let cols = 64;
        let data: Vec<f64> = (0..rows * cols).map(|i| (i as f64).sin() * 50.0).collect();

        for basis in [WaveletBasis::Db2, WaveletBasis::Db4, WaveletBasis::Db6,
                      WaveletBasis::Sym4, WaveletBasis::Coif2] {
            let coeffs = wavedec2(&data, rows, cols, 2, basis);
            let reconstructed = waverec2(&coeffs, basis);
            let max_err: f64 = data.iter().zip(reconstructed.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            assert!(max_err < 1e-6, "{}: DWT roundtrip error too large: {}", basis, max_err);
        }
    }

    #[test]
    fn test_denoise_residual_basic() {
        let w = 64u32;
        let h = 64u32;
        let mut residual = vec![128u8; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                let val = 128.0 + (x as f64 - 32.0) * 0.5;
                residual[(y * w + x) as usize] = val.clamp(0.0, 255.0) as u8;
            }
        }

        let result = denoise_residual(&residual, w, h, 0.25, WaveletBasis::Db4, 2, 1.0);
        assert_eq!(result.denoised.len(), (w * h) as usize);
        assert_eq!(result.removed_noise.len(), (w * h) as usize);
        assert_eq!(result.synth_params.basis, WaveletBasis::Db4);
        assert!(result.synth_params.seed > 0);
    }

    #[test]
    fn test_synthesize_noise() {
        let w = 64u32;
        let h = 64u32;
        let mut y_plane = vec![128u8; (w * h) as usize];
        let params = SynthesisParams {
            approx_sigma: 0.5,
            subband_sigmas: [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
            basis: WaveletBasis::Db4,
            seed: 42,
        };

        synthesize_and_apply_noise(&mut y_plane, w, h, &params, 1.0);

        let min = *y_plane.iter().min().unwrap();
        let max = *y_plane.iter().max().unwrap();
        assert!(max > min, "noise synthesis should produce variation");

        // Test strength=0.5 produces less variation
        let mut y_half = vec![128u8; (w * h) as usize];
        synthesize_and_apply_noise(&mut y_half, w, h, &params, 0.5);
        let var_full: f64 = y_plane.iter().map(|&v| (v as f64 - 128.0).powi(2)).sum::<f64>() / y_plane.len() as f64;
        let var_half: f64 = y_half.iter().map(|&v| (v as f64 - 128.0).powi(2)).sum::<f64>() / y_half.len() as f64;
        assert!(var_half < var_full, "half strength should have less variance");

        // Test strength=0 produces no change
        let mut y_zero = vec![128u8; (w * h) as usize];
        synthesize_and_apply_noise(&mut y_zero, w, h, &params, 0.0);
        assert!(y_zero.iter().all(|&v| v == 128), "zero strength should produce no change");
    }

    #[test]
    fn test_xorshift_laplace_distribution() {
        let mut rng = XorShift64::new(42);
        let b = 1.0;
        let n = 10000;
        let samples: Vec<f64> = (0..n).map(|_| rng.laplace(b)).collect();
        let mean = samples.iter().sum::<f64>() / n as f64;
        let variance = samples.iter().map(|s| (s - mean) * (s - mean)).sum::<f64>() / n as f64;
        assert!(mean.abs() < 0.1, "mean should be ~0, got {}", mean);
        assert!((variance - 2.0).abs() < 0.3, "variance should be ~2, got {}", variance);
    }

    #[test]
    fn test_dwt_idwt_roundtrip_1level_db4() {
        let rows = 16;
        let cols = 16;
        let data: Vec<f64> = (0..rows * cols).map(|i| (i as f64).sin() * 50.0).collect();
        let coeffs = wavedec2(&data, rows, cols, 1, WaveletBasis::Db4);
        let reconstructed = waverec2(&coeffs, WaveletBasis::Db4);
        let max_err: f64 = data.iter().zip(reconstructed.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(max_err < 1e-8, "1-level DWT roundtrip error too large: {}", max_err);
    }

    #[test]
    fn test_1d_roundtrip_db4() {
        let fb = WaveletBasis::Db4.filters();
        let signal: Vec<f64> = (0..16).map(|i| (i as f64).sin() * 10.0).collect();
        let lo = convolve_downsample(&signal, fb.lo_d);
        let hi = convolve_downsample(&signal, &fb.hi_d);
        let rec_lo = upsample_convolve(&lo, &fb.lo_r, signal.len());
        let rec_hi = upsample_convolve(&hi, &fb.hi_r, signal.len());
        let reconstructed: Vec<f64> = rec_lo.iter().zip(rec_hi.iter()).map(|(a, b)| a + b).collect();
        let max_err: f64 = signal.iter().zip(reconstructed.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(max_err < 1e-8, "db4 1D roundtrip error: {}", max_err);
    }

    #[test]
    fn test_wavelet_basis_from_str() {
        assert_eq!("db4".parse::<WaveletBasis>().unwrap(), WaveletBasis::Db4);
        assert_eq!("sym4".parse::<WaveletBasis>().unwrap(), WaveletBasis::Sym4);
        assert_eq!("coif2".parse::<WaveletBasis>().unwrap(), WaveletBasis::Coif2);
        assert!("invalid".parse::<WaveletBasis>().is_err());
    }
}
