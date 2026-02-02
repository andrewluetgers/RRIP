# Multi-stage, multi-architecture Dockerfile for ORIGAMI tile server
# Supports AMD64 and ARM64 with architecture-specific optimizations

# Build stage
FROM --platform=$BUILDPLATFORM rust:1.75-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    libturbojpeg0-dev \
    clang \
    && rm -rf /var/lib/apt/lists/*

# Set up cargo for cross-compilation
ARG TARGETARCH
ARG BUILDARCH

# Map Docker's TARGETARCH to Rust's target triple
RUN case "$TARGETARCH" in \
    amd64) echo "x86_64-unknown-linux-gnu" > /target.txt ;; \
    arm64) echo "aarch64-unknown-linux-gnu" > /target.txt ;; \
    *) echo "Unsupported architecture: $TARGETARCH" && exit 1 ;; \
    esac

# Install cross-compilation tools if needed
RUN if [ "$BUILDARCH" != "$TARGETARCH" ]; then \
    rustup target add $(cat /target.txt); \
    fi

WORKDIR /build

# Copy manifests first for better caching
COPY Cargo.toml Cargo.lock ./

# Create src directory with dummy main to cache dependencies
RUN mkdir src && echo "fn main() {}" > src/main.rs

# Build dependencies only (cached layer)
RUN if [ "$BUILDARCH" = "$TARGETARCH" ]; then \
        cargo build --release; \
    else \
        cargo build --release --target $(cat /target.txt); \
    fi

# Copy actual source code
COPY src ./src

# Clean and rebuild with actual source
RUN rm -rf target/release/deps/origami* target/$(cat /target.txt)/release/deps/origami* && \
    if [ "$BUILDARCH" = "$TARGETARCH" ]; then \
        RUSTFLAGS="-C target-cpu=native" cargo build --release; \
        cp target/release/origami-tile-server /origami-tile-server; \
    else \
        cargo build --release --target $(cat /target.txt); \
        cp target/$(cat /target.txt)/release/origami-tile-server /origami-tile-server; \
    fi

# Runtime stage - minimal image
FROM --platform=$TARGETPLATFORM debian:bookworm-slim

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libturbojpeg0 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1001 -s /bin/bash origami

# Copy binary from builder
COPY --from=builder /origami-tile-server /usr/local/bin/origami-tile-server
RUN chmod +x /usr/local/bin/origami-tile-server

# Create data directory
RUN mkdir -p /data && chown origami:origami /data

USER origami
WORKDIR /data

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/healthz || exit 1

# Default configuration
ENV RUST_LOG=info
ENV RUST_BACKTRACE=1

# Expose default port
EXPOSE 8080

# Default command with sensible defaults
ENTRYPOINT ["/usr/local/bin/origami-tile-server"]
CMD ["--slides-root", "/data", "--port", "8080"]