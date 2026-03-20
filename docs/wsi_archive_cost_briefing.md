the # WSI Tile Serving from Archive Storage: Cost & Latency

## Why Serve Directly from Archive?

- Pre-generating tiles duplicates storage costs
- Most tiles will never be viewed by a human
- A ~1GB slide viewed in a session uses only a tiny fraction of tiles
- Archive: no latency penalty for storage class, just pay more per byte served
- Cost scales with actual usage, not dataset size

## Cost Per Tile (25KB from Archive)

| Component | Rate | Cost |
|-----------|------|------|
| Retrieval | $0.05/GB | $0.00000125 |
| Request | $0.0069/1K requests | $0.00000690 |
| **Total** | | **$0.00000815** |

**Sources:**
- Retrieval: https://cloud.google.com/healthcare-api/pricing#dicom_data_retrieval
- Request: https://cloud.google.com/healthcare-api/pricing#request_volume

### Scaled Monthly Costs

| Tiles/Month | Monthly Cost |
|-------------|--------------|
| 1 million | $8.15 |
| 10 million | $81.50 |
| 100 million | $815 |
| 1 billion | $8,150 |

### Example Usage Calculation

| Assumption | Value |
|------------|-------|
| Tiles per screen | ~50 |
| Screens per session (zoom/pan) | ~20 |
| **Tiles per session** | **1,000** |

| Scenario | Daily Users | Tiles/Day | Monthly Cost |
|----------|-------------|-----------|--------------|
| Light (research) | 100 | 100K | ~$25 |
| Moderate | 500 | 500K | ~$125 |
| Heavy (clinical + research) | 1,000 | 1M | ~$250 |

(Cache hits on shared tiles would reduce costs)

## Latency

| Storage Class | Expected Latency |
|---------------|------------------|
| Standard | 10-50ms |
| Archive | 50-100ms+ |

**Sources:**
- Latency best practices: https://cloud.google.com/healthcare-api/docs/best-practices-network-latency

## Scalability

**Key finding:** The ~100 stream limit is HTTP/2 protocol (per TCP connection), NOT an API limit.

Google states the Healthcare API "can scale to thousands of requests per second." Quotas are per-project, per-region, per-minute and adjustable by contacting Google Cloud support.

**Sources:**
- https://docs.cloud.google.com/healthcare-api/docs/introduction
- https://docs.cloud.google.com/healthcare-api/quotas

**Implication:** A caching tier may be valuable for cost optimization (avoid re-fetching same tiles), but is NOT required for concurrency scaling.

## Caching Tier Analysis

### When Does Caching Make Sense?

**Cost per tile from Archive:** $0.00000815

**Note:** Browser caching with long Cache-Control headers handles repeat requests within a session. The caching tier question is about cross-session and cross-user scenarios.

### Break-Even Example

**Scenario:** 500 users/day, 1,000 tiles/session, some users revisit same slides

Without cache, all requests go to Archive:

| Metric | Value |
|--------|-------|
| Total tile requests/day | 500K |
| Archive cost/month | 500K × 30 days × $0.00000815 = **$122** |

**With caching tier ($170/mo n2-standard-4):**

| Cache Hit Rate | Requests to Archive | Archive Cost | Total Cost | vs No Cache |
|----------------|---------------------|--------------|------------|-------------|
| 0% | 500K/day | $122/mo | $292/mo | +$170 |
| 30% | 350K/day | $85/mo | $255/mo | +$133 |
| 50% | 250K/day | $61/mo | $231/mo | +$109 |
| 70% | 150K/day | $37/mo | $207/mo | +$85 |
| 90% | 50K/day | $12/mo | $182/mo | +$60 |

**At 500K tiles/day, caching never pays for itself on cost alone.**

### When Caching Does Pay Off

**Higher volume scenario:** 2M unique tiles/day (1,000 heavy users)

| Cache Hit Rate | Archive Cost (no cache) | With $170 Cache | Savings |
|----------------|-------------------------|-----------------|---------|
| 0% | $489/mo | $659/mo | -$170 |
| 50% | $489/mo | $414/mo | **+$75** |
| 70% | $489/mo | $316/mo | **+$173** |
| 90% | $489/mo | $219/mo | **+$270** |

**Break-even formula:**
```
Cache pays off when: Cache Hit Rate > 1 - (Archive Cost - Cache Cost) / Archive Cost
```

At 2M tiles/day ($489/mo Archive), a $170 cache breaks even at ~35% hit rate.

### Decision Matrix

| Daily Tile Volume | Break-even Hit Rate | Recommendation |
|-------------------|---------------------|----------------|
| 100K | 170% (impossible) | No cache |
| 500K | 86% | Cache only for latency |
| 1M | 65% | Cache if hit rate > 65% |
| 2M | 35% | Cache likely worthwhile |
| 5M+ | < 15% | Cache definitely worthwhile |

**Bottom line:** Caching is a latency play at low-medium volume. It becomes a cost play at high volume (2M+ tiles/day) with reasonable hit rates.

### Sizing by Retention Period

| Retention | Tiles (1K users/day) | Storage Size |
|-----------|---------------------|--------------|
| 1 day | 1M | 25 GB |
| 1 week | 7M | 175 GB |
| 30 days | 30M | 750 GB |

### Why RocksDB?

RocksDB is an embedded key-value store designed for fast storage (SSD/NVMe). Key properties:

- **Built-in LRU eviction** - Automatically evicts oldest entries when storage fills; no manual cleanup needed
- **SSD-optimized** - Designed for datasets larger than RAM; uses SSD as primary storage with RAM for hot indexes and bloom filters
- **Tunable RAM/SSD ratio** - Can run with modest RAM (8-16GB) since NVMe SSD provides ~100K IOPS random reads
- **Single binary** - Embeds directly into a proxy service, no separate database process to manage
- **Proven at scale** - Powers storage at Facebook, used as storage engine for MySQL (MyRocks) and MongoDB (MongoRocks)

For tile caching: tiles are stored as key-value pairs (tile coordinate → image bytes). RocksDB's LRU eviction naturally retains frequently-accessed tiles and ages out cold ones.

### GCP VM Cost Estimates

| Config | vCPU | RAM | Local SSD | Monthly Cost |
|--------|------|-----|-----------|--------------|
| n2-standard-2 + 375GB SSD | 2 | 8 GB | 375 GB | ~$100 |
| n2-standard-4 + 375GB SSD | 4 | 16 GB | 375 GB | ~$170 |
| n2-standard-4 + 750GB SSD | 4 | 16 GB | 750 GB | ~$200 |

*Pricing: n2-standard-4 ~$142/mo, Local SSD ~$0.08/GB/mo ($30/375GB)*

## Open Questions

1. **Validate archive latency** - Confirm 50-100ms in practice
2. **Quantify access patterns** - Expected daily/monthly usage for clinical vs. research
3. **Clinical vs. research split** - What % of slides need low-latency clinical access vs. occasional research?
