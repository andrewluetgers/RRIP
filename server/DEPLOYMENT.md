# RRIP Tile Server Deployment Guide

## Multi-Architecture Build & Deployment

This guide covers building and deploying the RRIP tile server for both AMD64 and ARM64 architectures.

## Prerequisites

1. **Depot Account** (for fast multi-arch builds)
   - Sign up at https://depot.dev
   - Create a project and note your Project ID

2. **GitHub Repository Secrets**
   Add these secrets to your GitHub repository:
   - `DEPOT_PROJECT_ID`: Your Depot project ID
   - `DEPOT_TOKEN`: Your Depot API token

## Architecture-Specific Optimizations

The server includes optimizations for both architectures:

### AMD64 (x86_64)
- **AVX2 SIMD**: 32-byte vector operations for modern CPUs
- **SSE2 SIMD**: 16-byte vector operations for older CPUs
- **TurboJPEG**: Hardware-accelerated JPEG encoding/decoding
- **Fixed-point math**: Integer arithmetic for color conversion

### ARM64 (Apple M-series, AWS Graviton)
- **NEON SIMD**: ARM's 16-byte vector operations
- **TurboJPEG**: Hardware-accelerated JPEG on ARM
- **Optimized bilinear interpolation**: For image upsampling

## Local Development

### Building for your architecture
```bash
# AMD64
docker build --platform linux/amd64 -t rrip-server:amd64 .

# ARM64 (Apple Silicon)
docker build --platform linux/arm64 -t rrip-server:arm64 .

# Multi-arch with Docker Buildx
docker buildx build --platform linux/amd64,linux/arm64 -t rrip-server:latest .
```

### Running with Docker Compose
```bash
# Start the server
docker-compose up -d

# View logs
docker-compose logs -f rrip-server

# Stop the server
docker-compose down
```

## CI/CD with GitHub Actions

The workflow automatically:
1. Tests on native architecture
2. Builds multi-arch images with Depot
3. Pushes to GitHub Container Registry
4. Runs performance benchmarks

### Manual trigger
```bash
# Push to main branch
git push origin main

# Or create a release
git tag v1.0.0
git push origin v1.0.0
```

## Deployment Options

### 1. Kubernetes (EKS, GKE, AKS)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rrip-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rrip-server
  template:
    metadata:
      labels:
        app: rrip-server
    spec:
      nodeSelector:
        # For AMD64 nodes
        kubernetes.io/arch: amd64
        # For ARM64 nodes (uncomment if using Graviton)
        # kubernetes.io/arch: arm64
      containers:
      - name: rrip-server
        image: ghcr.io/yourusername/rrip:latest
        ports:
        - containerPort: 8080
        resources:
          requests:
            cpu: "2"
            memory: "2Gi"
          limits:
            cpu: "4"
            memory: "4Gi"
        volumeMounts:
        - name: tile-data
          mountPath: /data
          readOnly: true
        env:
        - name: RUST_LOG
          value: "info"
      volumes:
      - name: tile-data
        persistentVolumeClaim:
          claimName: tile-data-pvc
```

### 2. AWS ECS with Fargate

```json
{
  "family": "rrip-server",
  "requiresCompatibilities": ["FARGATE"],
  "networkMode": "awsvpc",
  "cpu": "4096",
  "memory": "8192",
  "runtimePlatform": {
    "cpuArchitecture": "X86_64"
  },
  "containerDefinitions": [{
    "name": "rrip-server",
    "image": "ghcr.io/yourusername/rrip:latest",
    "portMappings": [{
      "containerPort": 8080,
      "protocol": "tcp"
    }],
    "environment": [
      {"name": "RUST_LOG", "value": "info"}
    ],
    "mountPoints": [{
      "sourceVolume": "tile-data",
      "containerPath": "/data",
      "readOnly": true
    }],
    "healthCheck": {
      "command": ["CMD-SHELL", "curl -f http://localhost:8080/healthz || exit 1"],
      "interval": 30,
      "timeout": 5,
      "retries": 3
    }
  }]
}
```

### 3. Docker Swarm

```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml rrip

# Scale service
docker service scale rrip_rrip-server=5
```

## Performance Tuning

### Environment Variables
```bash
# Thread configuration
RAYON_NUM_THREADS=8        # Parallel processing threads
TOKIO_WORKER_THREADS=4     # Async runtime workers

# Cache configuration
CACHE_ENTRIES=8192          # In-memory cache size
ROCKSDB_CACHE_SIZE=1073741824  # 1GB RocksDB cache

# Rust optimizations
RUST_LOG=warn               # Reduce logging overhead
RUSTFLAGS="-C target-cpu=native -C opt-level=3"
```

### Architecture-specific tuning

#### AMD64
```bash
# Enable AVX2 if available
RUSTFLAGS="-C target-cpu=x86-64-v3"  # Includes AVX2

# For older CPUs (SSE2 only)
RUSTFLAGS="-C target-cpu=x86-64"
```

#### ARM64
```bash
# Apple M-series
RUSTFLAGS="-C target-cpu=apple-m1"

# AWS Graviton
RUSTFLAGS="-C target-cpu=neoverse-n1"
```

## Monitoring

### Prometheus metrics
The server exposes metrics at `/metrics`:
- Request latency histograms
- Cache hit rates
- Family generation timing
- Memory usage

### Example Grafana dashboard
Import the dashboard from `monitoring/grafana-dashboard.json`

## Troubleshooting

### Check SIMD support
```bash
# AMD64
docker run --rm rrip-server:latest sh -c "cat /proc/cpuinfo | grep -E 'avx2|sse2'"

# ARM64
docker run --rm rrip-server:latest sh -c "cat /proc/cpuinfo | grep -E 'neon|asimd'"
```

### Performance testing
```bash
# Load test with wrk
wrk -t12 -c400 -d30s --latency http://localhost:8080/tiles/demo_out/14/100_100.jpg

# Check timing breakdown
curl http://localhost:8080/tiles/demo_out/14/100_100.jpg -H "X-Debug: timing"
```

### Container resource limits
```bash
# Check current limits
docker stats rrip-server

# Adjust in docker-compose.yml or K8s deployment
```

## Cost Optimization

### AWS Graviton (ARM64)
- 40% better price-performance than x86
- Use with ECS Fargate or EC2

### Spot Instances
- Use for non-critical workloads
- Configure with interruption handling

### Auto-scaling
```yaml
# HPA for Kubernetes
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rrip-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rrip-server
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## Security

### Running as non-root
The container runs as UID 1001 by default.

### Read-only filesystem
```yaml
# Docker Compose
read_only: true
tmpfs:
  - /tmp

# Kubernetes
securityContext:
  readOnlyRootFilesystem: true
  runAsNonRoot: true
  runAsUser: 1001
```

### Network policies
Restrict ingress to port 8080 only.

## Support

For issues or questions:
- GitHub Issues: [your-repo/issues]
- Performance problems: Check `/metrics` endpoint first
- Architecture-specific issues: Include CPU info in bug reports