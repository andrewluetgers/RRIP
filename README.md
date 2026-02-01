# RRIP: Residual Reconstruction from Interpolated Priors

[![Build and Test](https://github.com/andrewluetgers/RRIP/actions/workflows/build-and-test.yml/badge.svg)](https://github.com/andrewluetgers/RRIP/actions/workflows/build-and-test.yml)

A high-performance tile server for gigapixel whole-slide images (WSI) that achieves 80-95% storage reduction through intelligent residual compression while maintaining visual quality.

## 🎯 Overview

RRIP is a serving-oriented compression system designed for pathology and satellite imagery that leverages a key insight: **the finest two pyramid levels (L0 and L1) typically account for 80-95% of storage in tiled image pyramids**.

Instead of storing these levels as full-quality JPEG tiles, RRIP:
- Keeps conventional high-quality tiles for L2 and coarser levels
- Stores L0/L1 as compact residuals relative to interpolated predictions
- Reconstructs tiles on-demand with optimized CPU operations
- Caches tile families for efficient serving

### Key Features

- **80-95% storage reduction** for gigapixel images
- **Sub-200ms tile reconstruction** with TurboJPEG and SIMD optimizations
- **Compatible with existing viewers** (OpenSeadragon, QuPath, etc.)
- **CPU-only operation** - no GPU required
- **Multi-architecture support** (x86_64 with AVX2/SSE, ARM64 with NEON)

## 🚀 Quick Start

### Prerequisites

```bash
# macOS
brew install jpeg-turbo

# Linux
apt-get install libturbojpeg0-dev
```

### Build and Run

```bash
# Clone the repository
git clone https://github.com/andrewluetgers/RRIP.git
cd RRIP/server

# Build optimized release version
./build.sh

# Run the server
./run-server.sh

# Access the viewer
open http://localhost:3007/viewer/demo_out
```

## 🏗️ Architecture

### How It Works

RRIP implements a hierarchical reconstruction strategy:

```
L2 (Base) → Upsample 2x → L1 Prediction → Add Residual → L1 Reconstructed
           ↓                              ↓
           → Upsample 4x → L0 Prediction → Add Residual → L0 Reconstructed
```

1. **Base Layer (L2)**: Stored as conventional high-quality JPEG tiles
2. **Residual Encoding**: L1/L0 stored as grayscale residuals (luma only)
3. **Chroma Inheritance**: Color information carried from L2 (4:2:0 subsampling across pyramid levels)
4. **Family Generation**: When any L0/L1 tile is requested, generate all 20 tiles in the family (4 L1 + 16 L0)

### Performance Optimizations

- **TurboJPEG**: 3-5x faster JPEG encoding/decoding
- **SIMD Processing**: Platform-specific optimizations (AVX2, SSE2, NEON)
- **Fixed-Point Math**: Eliminates floating-point overhead in color conversion
- **Memory Pooling**: Reduces allocation overhead
- **Dual-Tier Cache**: Hot (in-memory LRU) + Warm (RocksDB persistent)

## 📁 Project Structure

```
RRIP/
├── server/                 # High-performance Rust tile server
│   ├── src/
│   │   ├── main.rs        # Server core with tile reconstruction
│   │   ├── fast_upsample_ycbcr.rs    # SIMD upsampling
│   │   └── turbojpeg_optimized.rs    # TurboJPEG integration
│   ├── build.sh           # Build script with optimizations
│   ├── run-server.sh      # Run script with auto-configuration
│   └── tests/             # Integration tests
│
├── cli/                   # Python preprocessing tools
│   ├── wsi_residual_tool.py    # Generate residual pyramids
│   └── requirements.txt
│
├── notebooks/            # Research and analysis
│   └── wsi_residual_pyramid_tool.ipynb
│
├── paper/               # Research documentation
│   └── rrip-paper.md   # Method description and evaluation
│
└── data/               # Sample data (not in repo)
    └── demo_out/       # Example slide with residuals
```

## 🛠️ Usage

### Server Options

```bash
# Basic usage
./run-server.sh

# Custom port
./run-server.sh --port 8080

# Debug mode with timing
./run-server.sh --debug --timing

# See all options
./run-server.sh --help
```

### Build Options

```bash
# Build release version (default)
./build.sh

# Build debug with tests
./build.sh --debug --test

# Clean build
./build.sh --clean --release

# Build Docker image
./build.sh --docker
```

### Preprocessing Pipeline

Generate residual pyramids from existing slides:

```bash
# Install Python dependencies
pip install -r cli/requirements.txt

# Generate residuals for a slide
python cli/wsi_residual_tool.py \
  --input /path/to/slide.svs \
  --output /path/to/output \
  --quality 32

# Optional: Pack residuals for faster serving
python cli/wsi_residual_tool.py pack \
  --residuals /path/to/residuals \
  --out /path/to/packs
```

## 📊 Performance

Typical performance on modern hardware:

| Metric | Value |
|--------|-------|
| **Storage Reduction** | 80-95% vs JPEG pyramid |
| **L0 Tile Generation** | 150-300ms (full family) |
| **Cache Hit Latency** | <10ms (hot), <25ms (warm) |
| **Throughput** | 500-1000 tiles/sec |
| **Memory Usage** | 200-500MB typical |

### Benchmark Results

On Apple M1 Pro (ARM64):
- Family generation: ~266ms (20 tiles)
- Parallel chroma: ~17ms
- L0 resize: ~85ms
- L0 residual: ~12ms per tile

## 🐳 Docker Deployment

```bash
# Build Docker image
docker build -t rrip-server server/

# Run with Docker
docker run -p 3007:3007 \
  -v $(pwd)/data:/data \
  rrip-server

# Or use Docker Compose
docker-compose up
```

## 🧪 Testing

```bash
# Run unit tests
cargo test

# Run integration tests
cargo test --test integration_test

# Run with coverage
cargo tarpaulin --out Html
```

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

1. Fork and clone the repository
2. Install Rust (1.70+) and Python (3.8+)
3. Install TurboJPEG for your platform
4. Run tests: `cargo test`
5. Submit PR with tests

## 📚 Research

RRIP is based on research in multi-resolution image compression and perceptual coding. The method exploits:

1. **Pyramid byte dominance**: L0+L1 comprise majority of storage
2. **Perceptual redundancy**: Chroma can be subsampled more aggressively
3. **Spatial correlation**: Interpolation captures most image structure
4. **Access patterns**: Viewers typically request spatially-coherent tiles

See [paper/rrip-paper.md](paper/rrip-paper.md) for detailed methodology.

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [OpenSeadragon](https://openseadragon.github.io/) for the viewer
- [libjpeg-turbo](https://libjpeg-turbo.org/) for fast JPEG operations
- [RocksDB](https://rocksdb.org/) for persistent caching

## 📧 Contact

- **Author**: Andrew Luetgers
- **GitHub**: [@andrewluetgers](https://github.com/andrewluetgers)
- **Issues**: [GitHub Issues](https://github.com/andrewluetgers/RRIP/issues)
