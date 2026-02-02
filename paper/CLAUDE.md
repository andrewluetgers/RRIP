# CLAUDE.md - ORIGAMI Paper Writing Guidelines

This document provides specific guidelines for writing and maintaining consistency in the ORIGAMI paper.

## Critical Writing Rules

### 1. Compression Ratio Terminology

**NEVER USE:**
- "2× compression" - ambiguous
- "6× reduction" - unclear what's being reduced
- "82% storage reduction" without context
- Mixing percentages and ratios

**ALWAYS USE:**
Clear, unambiguous ratios with explicit comparison:

#### Correct Examples:
- "Storage ratio: 6:1 (JPEG Q90 : ORIGAMI)"
- "ORIGAMI uses 1/6 the storage of JPEG Q90"
- "Compression ratio: 55:1 (raw pixels : ORIGAMI)"
- "ORIGAMI achieves 82% storage savings compared to JPEG Q90 (uses 18% of original size)"

#### Standard Format:
```
[Method A] : [Method B] = [ratio]
where Method A is the reference (larger)
and Method B is the comparison (smaller)
```

### 2. Quality Context for All Comparisons

**EVERY compression comparison MUST include:**

1. **Quality metrics at that compression level:**
   - PSNR (dB) relative to reference
   - SSIM value
   - Visual quality assessment

2. **Reference baseline:**
   - What are we comparing against? (raw, JPEG Q90, etc.)
   - Why this reference?

3. **Measurement methodology:**
   - How was quality measured?
   - What test set was used?
   - Statistical significance if applicable

#### Template for Compression Claims:
```
"At PSNR 49.8 dB relative to JPEG Q90 (SSIM 0.98), ORIGAMI achieves a
storage ratio of 5.5:1, using only 18% of the baseline storage."
```

### 3. Consistent Terminology

**Storage Metrics:**
- **Storage ratio**: X:1 (baseline : method)
- **Storage fraction**: Uses 1/X of baseline
- **Storage savings**: Y% reduction from baseline (= 100% - 100%/X)

**Quality Metrics:**
- **PSNR**: Always in dB, always specify reference
- **SSIM**: Value between 0-1, always specify reference
- **Visual quality**: Use standard terms (excellent >45dB, very good >40dB, good >35dB)

**Never mix contexts:**
- Don't compare "compression from raw" with "compression from JPEG"
- Always specify both endpoints of comparison

### 4. Notation Consistency

**Pyramid Levels:**
- **L0**: Highest resolution (native pixels)
- **L1**: First downsample (1/4 the pixels of L0)
- **L2**: Second downsample (1/16 the pixels of L0)
- Be consistent throughout - don't mix with Deep Zoom notation

**Tile Coordinates:**
- Use (x, y) consistently
- Specify level when relevant: "tile (x, y) at level L0"

### 5. Data Presentation Rules

**Tables:**
- Always include units in headers
- Use consistent decimal places
- Sort by logical order (quality or compression)
- Include footnotes for clarification

**Example Table Format:**
```
| Method | Storage (MB) | Ratio to Raw | Ratio to Q90 | PSNR vs Q90 (dB) | SSIM vs Q90 |
|--------|-------------|--------------|--------------|------------------|-------------|
| Raw    | 30,000      | 1:1          | -            | ∞                | 1.000       |
| Q90    | 4,870       | 6.2:1        | 1:1          | Reference        | Reference   |
| ORIGAMI   | 880         | 34:1         | 5.5:1        | 49.8             | 0.98        |
```

**Graphs:**
- Always label axes with units
- Include error bars when applicable
- Use consistent color scheme
- Provide legends

### 6. Claims and Evidence

**Every numerical claim must have:**
1. Source of measurement
2. Test conditions
3. Reproducibility information

**Bad:** "ORIGAMI is 5× better than JPEG"

**Good:** "On our test set of 50 H&E tiles, ORIGAMI achieved 5.5:1 storage ratio compared to JPEG Q90 while maintaining PSNR of 49.8 dB"

### 7. Abstract and Introduction Rules

**Abstract must include:**
- Exact compression ratios with baselines
- Quality metrics with values
- Performance numbers with units
- Clear statement of what problem we solve

**Introduction must:**
- State the problem with quantification
- Show why existing solutions are insufficient
- Preview our solution with key metrics
- Be accessible to non-experts

### 8. Mathematical Notation

**Be precise:**
- Define all variables on first use
- Use consistent notation throughout
- Include units where applicable

**Example:**
```
Let S_raw = uncompressed storage in bytes
Let S_jpeg = JPEG Q90 storage in bytes
Let S_origami = ORIGAMI storage in bytes

Compression ratio (raw to ORIGAMI) = S_raw : S_origami = 34:1
Compression ratio (JPEG to ORIGAMI) = S_jpeg : S_origami = 5.5:1
```

### 9. Common Mistakes to Avoid

1. **Don't say "X× compression"** - use "X:1 compression ratio"
2. **Don't mix percentages and ratios** in the same sentence
3. **Don't omit quality context** when citing compression
4. **Don't use "significant" without statistics**
5. **Don't claim "lossless" unless truly lossless**
6. **Don't compare different quality levels** without noting the difference
7. **Don't use vague terms** like "much better" or "dramatically improved"

### 10. Standard Phrases

Use these standardized phrases for consistency:

- "ORIGAMI achieves a storage ratio of 5.5:1 compared to JPEG Q90"
- "At comparable visual quality (PSNR > 45 dB)"
- "Storage requirements are reduced to 18% of the JPEG Q90 baseline"
- "The compression ratio from raw pixels to ORIGAMI is 34:1"
- "Quality degradation is minimal (ΔPSNR < 2 dB)"

### 11. Benchmark Reporting

When reporting performance:
- **Latency**: Use ms or μs with percentiles (P50, P95, P99)
- **Throughput**: Use tiles/second or MB/second
- **Resource usage**: CPU% (cores), Memory (MB/GB)
- **Concurrency**: Specify number of connections/threads

**Example:**
"Under 64 concurrent connections, ORIGAMI serves 368 tiles/second with P95 latency of 12ms on a 4-core Apple M1 processor, using 850MB RAM."

### 12. Configuration Documentation

Always specify complete configuration when presenting results:

```
Configuration: ORIGAMI-Q32
- Quantization: 32 levels
- Residual JPEG quality: 32
- LZ4 compression: enabled
- Baseline pyramid: JPEG Q90
- Tile size: 256×256 pixels
```

## Review Checklist

Before submitting, verify:

- [ ] All compression ratios use X:1 format with clear endpoints
- [ ] Every compression claim includes quality metrics
- [ ] No ambiguous "× compression" or "× reduction" statements
- [ ] Consistent use of L0/L1/L2 notation
- [ ] All numbers have units and precision specified
- [ ] Tables and figures follow standard format
- [ ] Abstract contains concrete metrics
- [ ] Claims are supported by data
- [ ] Configuration is fully specified
- [ ] Performance includes test conditions