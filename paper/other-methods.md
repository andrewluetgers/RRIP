Below is a **“related work” compression table** for WSIs (and near-WSI digital pathology stacks) that you can drop into the *Background / Related Work* section of an ORIGAMI paper. Two caveats up front:

* **Compression ratio is not apples-to-apples** across papers: authors vary the reference (raw RGB, scanner-native JPEG, TIFF, JP2 pyramid, “tissue-only,” etc.), the content (tissue density/background), and the target (diagnostic acceptability vs PSNR/SSIM).
* Where a paper only reports **relative bitrate savings** (not an absolute ratio), I show that as “% savings” and a derived “× improvement vs baseline” when possible.

---

## WSI compression ratios and quality metrics (selected related work)

| Method / family                                                                               |                  Lossless? |                                                                                                                                                                 Typical reported storage reduction | Quality metric(s) used in paper                                                       | Notes                                                                                                                                                                                                                                                                                                                           |
| --------------------------------------------------------------------------------------------- | -------------------------: | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Conventional “generic” lossless compressors on raw WSI pixels** (PNG/TIFF/LZMA/Zstd, etc.)  |                          ✅ |                                                                                                                                             Often **~1–2×** on WSI raw pixels (patch-level varies) | Compression ratio (primary)                                                           | WISE reports that generic compressors are typically not enough for WSI-scale savings (their Table 1 shows WSI ratios around ~0.87–1.93 depending on method/setting).                                                                                                                                                            |
| **JPEG2000 in virtual microscopy / WSI**                                                      | ❌ (usually) / ✅ (optional) |                                                         “Good quality” reported around **~25:1–30:1** in one virtual microscopy workflow (note: includes background effects) ([PubMed Central][1]) | Visual / workflow considerations (random access, progressive decode)                  | JP2’s biggest practical win in WSI is often **random access + multi-resolution** (precincts/tiles) rather than purely ratio. ([PubMed Central][1])                                                                                                                                                                              |
| **JPEG2000 diagnostic study (WSI)**                                                           |                          ❌ |                                                                                                                                                      Study tested **8:1, 16:1, 32:1, 64:1, 128:1** | Diagnostic accuracy / ROC style evaluation                                            | This is a commonly cited “how far can we push compression” clinical-style evaluation set. ([PubMed Central][2])                                                                                                                                                                                                                 |
| **ROI/background-aware JP2-WSI (lossless tissue + heavy background compression)**             |                      mixed |                                                                Reported **~10–300×** overall compression ratio, with **~10–200× typical** depending on tissue/background mix ([PubMed Central][2]) | Compression ratio; ROI-driven coding                                                  | Important: this “overall ratio” is dominated by background/empty regions being ultra-compressible. ([PubMed Central][2])                                                                                                                                                                                                        |
| **HEVC / H.265 (digital pathology imagery; multi-plane microscopy stack)**                    |                          ❌ |                                                              For comparable SSIM targets, a paper reports JPEG at **~30–43×**, JP2 at **~58–126×**, and HEVC at **~84–262×** ([PubMed Central][3]) | SSIM (target-matched), PSNR reported as well                                          | This dataset is a *multi-plane pathology microscopy* use case (z-stacks). Still highly relevant because it quantifies HEVC-vs-JP2-vs-JPEG under a consistent SSIM target. ([PubMed Central][3])                                                                                                                                 |
| **Scalable HEVC (SHVC) for WSI telepathology streaming**                                      |                          ❌ |                         **~54% bitrate saving vs JPEG**, **~12% vs JPEG2000** at comparable quality → roughly **~2.2×** better than JPEG and **~1.14×** better than JP2 (at that operating point)  | JND / visual discrimination model (artifact detectability); BD-rate style comparisons | Nice “WSI streaming” angle: scalable layers, good for interactive viewers.                                                                                                                                                                                                                                                      |
| **WISE (CVPR 2025): delta/projection + bitmap + dictionary for WSI**                          |                          ✅ |                                                                                                Claims **up to 136×** and **~36× on average** over prior methods (lossless WSI compression claim).  | Compression ratio (primary), plus some PSNR analyses on internal bitmaps              | WISE also shows patch-level compression ratios across datasets, and an ablation that explains which components drive gains.                                                                                                                                                                                                     |
| **Mosaic-based color-transform optimization (for JPEG2000 lossy & lossy-to-lossless on WSI)** |                      mixed | Not a new “ratio line,” but shows **better PSNR / visual metrics at identical compression ratios** (e.g., up to ~1.1 dB PSNR gain vs KLT in some settings) ([UAB Digital Documents Repository][4]) | PSNR, HDR-VDP-2, task metric (nuclei detection F1)                                    | This matters for ORIGAMI’s “color/channel strategy” discussion: *color transforms strongly affect coding efficiency* in pathology. ([UAB Digital Documents Repository][4])                                                                                                                                                         |
| **Graph-based rate control with lossless ROI coding (HEVC-based)**                            |                      mixed |                                                                          Focus is on **meeting target bitrate** while keeping ROI lossless (not a single headline compression ratio) ([PubMed][5]) | Rate-control accuracy; ROI constraints                                                | Useful related-work anchor if ORIGAMI evolves into “ROI-aware residual allocation.” ([PubMed][5])                                                                                                                                                                                                                                  |
| **Laplacian pyramid / pyramid residual coding (general image compression idea)**              |                    depends |                                                                                                         No single WSI headline ratio (it’s a *representation* that can be paired with many coders) | Depends on coder                                                                      | In ORIGAMI terms: your “coarse prior + residual” framing is essentially **pyramidal / Laplacian-style residual modeling**, but your *serving-time* framing (tile server + cached reconstruction) is the WSI-specific systems twist. (For the paper, you’d cite classic pyramid coding and then show your WSI-specific adaptation.) |

---

## What WISE attributes its biggest gains to (useful for your “Compared to WISE” section)

WISE’s reported gains come from stacking multiple ideas:

1. **Hierarchical projection / delta-style coding** across row/column/channel (their “hierarchical linear projection” step).
2. **Bitmap (bit-plane) encoding** to reorganize “effective bits” into patterns that become more compressible.
3. **Dictionary coding (LZW-style)** on top of the bitmaps for repeated local patterns.

Their **ablation table** shows the stepwise improvements from baseline → +hierarchical projection → +bitmap encoding (which is a clean way to explain “where the wins come from”).

---

## How this relates to ORIGAMI’s “chroma-from-coarse / luma residual” idea

You asked whether *existing techniques match* “stretch coarse chroma + store luma residuals.” Two key “this is not coming out of nowhere” anchors for your paper’s related work framing:

* **Chroma subsampling (4:2:0 / 4:2:2)** is the classic version of “lower-res chroma, higher-res luma,” but it’s usually **paired with block transforms (DCT)** and not framed as “viewer-time reconstruction from an interpolated prior.”
* **Pyramidal / Laplacian residual coding** is the classic version of “coarse prior + residual,” but again, not usually packaged as a **tile-serving architecture** with “generate all children tiles under a parent view” caching rules.

So ORIGAMI’s novelty angle (as you’ve been steering it) is less “no one ever thought of priors + residuals,” and more:

* *A WSI-serving-specific factorization* (keep L2+ as normal; derive L1/L0 via interpolated priors),
* *Residuals chosen to be cheap and CPU-decodable at serving time* (JPEG residual channel, etc.),
* *A cache policy aligned with navigation behavior* (serve one tile → precompute its sibling tiles under that L2).

---

## If you want, I can format this into a “Related Work” subsection

If you tell me whether your ORIGAMI paper is targeting **lossless archival** (WISE-adjacent) vs **lossy serving format** (your current direction), I’ll reshape the above into:

* a proper narrative related-work section (1–2 pages),
* a cleaned comparison table with consistent definitions (e.g., ratios vs raw RGB 24 bpp where papers provide bpp/bitrate),
* and a short “positioning statement” that makes ORIGAMI sound non-redundant with JP2/HEVC/WISE.

[1]: https://pmc.ncbi.nlm.nih.gov/articles/PMC3043697/ "
            The Application of JPEG2000 in Virtual Microscopy - PMC
        "
[2]: https://pmc.ncbi.nlm.nih.gov/articles/PMC3352607/ "
            Compressing pathology whole-slide images using a human and model observer evaluation - PMC
        "
[3]: https://pmc.ncbi.nlm.nih.gov/articles/PMC6921690/ "
            Video compression to support the expansion of whole-slide imaging into cytology - PMC
        "
[4]: https://ddd.uab.cat/pub/artpub/2019/200911/ieetramedima_a2019m1v38n1p21.pdf "Mosaic-Based Color-Transform Optimization for the Lossy and Lossy-to-Lossless compression of Pathology Whole-Slide Images"
[5]: https://pubmed.ncbi.nlm.nih.gov/29993629/ "Graph-Based Rate Control in Pathology Imaging With Lossless Region of Interest Coding - PubMed"
