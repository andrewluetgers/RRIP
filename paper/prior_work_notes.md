# Prior Work & Related Literature for ORIGAMI Paper

## Overview

This document summarizes prior art research conducted across ~80 targeted searches
covering 1983-2026 literature. The ORIGAMI technique combines three elements:
(1) multi-hop resolution pyramid, (2) luma-only residual correction, and
(3) chroma derived entirely from bilinear interpolation of a low-resolution base.
No prior work was found that combines all three.

---

## 1. Foundational: Laplacian Pyramid Coding

**Burt, P.J. & Adelson, E.H. (1983). "The Laplacian Pyramid as a Compact Image Code."
IEEE Transactions on Communications, 31(4), 532-540.**
- Paper: https://persci.mit.edu/pub_pdfs/pyramid83.pdf
- Stores a low-resolution base image plus difference (residual) images at each scale
- ORIGAMI's theoretical ancestor — same predict-then-correct pyramid structure
- Key difference: Applied to grayscale; does not exploit luma/chroma separation

## 2. JPEG Hierarchical Mode (ITU-T T.81, Annex K)

**ITU-T T.81 (1992). "Information technology — Digital compression and coding of
continuous-tone still images — Requirements and guidelines." Annex K.**
- Overview: https://www.sciencedirect.com/topics/computer-science/hierarchical-mode
- Tutorial: https://users.ece.utexas.edu/~ryerraballi/MSB/pdfs/M4L1_HJPEG.pdf
- Encodes a resolution pyramid with differential frames at each level
- Nearly identical structure to ORIGAMI, but:
  - Applies the pyramid to ALL channels equally (Y, Cb, Cr each get residual hierarchy)
  - Never gained practical adoption (almost no implementations exist)
  - Not tile-based; no concept of DZI pyramids or HTTP tile serving

## 3. Embedded Zerotree Wavelet (EZW) and SPIHT

**Shapiro, J.M. (1993). "Embedded Image Coding Using Zerotrees of Wavelet Coefficients."
IEEE Transactions on Signal Processing, 41(12), 3445-3462.**
- Paper: https://ieeexplore.ieee.org/document/258085/

**Said, A. & Pearlman, W.A. (1996). "A New, Fast, and Efficient Image Codec Based on
Set Partitioning in Hierarchical Trees." IEEE TCSVT, 6(3), 243-250.**
- Wikipedia: https://en.wikipedia.org/wiki/Embedded_zerotrees_of_wavelet_transforms
- Progressive wavelet coders exploiting cross-scale correlations
- Pyramid-based but wavelet-domain, not spatial-domain predict-and-correct
- Does not separate luma/chroma handling

## 4. Scalable Video Coding (H.264/SVC)

**Schwarz, H., Marpe, D., & Wiegand, T. (2007). "Overview of the Scalable Video Coding
Extension of the H.264/AVC Standard." IEEE TCSVT, 17(9), 1103-1120.**
- Fraunhofer HHI: https://www.hhi.fraunhofer.de/en/departments/vca/research-groups/video-coding-technologies/research-topics/past-research-topics/scalable-video-coding-in-h264-avc/inter-layer-prediction-for-scalable-video-coding.html
- Inter-layer prediction: upsamples base layer to predict enhancement layer, codes residual
- Structurally similar to ORIGAMI, but:
  - Implemented within complex video codec, not independent JPEG tiles
  - Corrects ALL channels (Y, Cb, Cr) in enhancement layers
  - Tiles not independently decodable

## 5. Scalable HEVC (SHVC)

**Boyce, J.M. et al. (2016). "Overview of SHVC: Scalable Extensions of the High
Efficiency Video Coding Standard." IEEE TCSVT, 26(1), 20-34.**
- Paper: https://ieeexplore.ieee.org/document/7254165/
- Fraunhofer: https://hevc.hhi.fraunhofer.de/shvc
- Provides spatial, SNR, bit depth, and color gamut scalability
- Uses inter-layer reference picture processing with texture and motion resampling
- Key difference from ORIGAMI: corrects all channels; not tile-independent

## 6. Scalable HEVC for WSI (Closest Domain-Specific Match)

**Bug, D., Bartsch, F. et al. (2020). "Scalable HEVC for Histological Whole-Slide
Image Compression." Informatik 2019, Springer.**
- Paper: https://link.springer.com/chapter/10.1007/978-3-658-29267-6_71
- **Most closely related paper found** — applies SHVC to whole-slide images
- Uses inter-layer prediction where low-res base is upsampled to predict higher-res layers
- Key differences from ORIGAMI:
  - Uses HEVC codec machinery, not simple JPEG tiles
  - Does NOT separate luma and chroma (no luma-only residuals)
  - Does NOT operate on existing tiled DZI pyramids
  - Tiles not independently decodable for HTTP serving

## 7. LCEVC (Low Complexity Enhancement Video Coding)

**V-Nova (2020). MPEG-5 Part 2 / ISO/IEC 23094-2.**
- Overview: https://www.lcevc.org/how-lcevc-works/
- IEEE: https://ieeexplore.ieee.org/document/9795094
- Encodes low-resolution base with any codec, adds residual enhancement layers
- **Closest structural match** — low-res base upsampled with residual correction
- Key difference: corrects ALL channels in enhancement layers; designed for video

## 8. Chroma from Luma (CfL) in AV1

**Egge, N.E. & Valin, J.-M. (2017). "Predicting Chroma from Luma in AV1."
Data Compression Conference (DCC), 2018.**
- Paper: https://ar5iv.labs.arxiv.org/html/1711.03951
- Also: https://ieeexplore.ieee.org/document/8416610/
- Predicts chroma pixels as a linear function of reconstructed luma pixels
- Closest conceptual match to ORIGAMI's chroma strategy (deriving color from brightness)
- Key differences:
  - Operates at block level (intra-prediction), not at resolution pyramid level
  - Saves ~5% BD-rate — much smaller gain than ORIGAMI's approach
  - Does not use multi-hop resolution pyramid

## 9. Cross-Component Prediction (CCP) in HEVC RExt

**Nguyen, T. & Marpe, D. (2015). "Cross-component prediction in HEVC."
Fraunhofer HHI.**
- Paper: https://www.researchgate.net/publication/283470863_Cross-component_prediction_in_HEVC
- Fraunhofer: https://www.hhi.fraunhofer.de/en/departments/vca/research-groups/video-coding-technologies/research-topics/past-research-topics/cross-component-prediction-in-hevc-range-extensions.html
- Predicts chroma residuals from luma residuals within the coding loop
- Operates in residual domain at block level, not resolution pyramid level
- Saves 2-18% bitrate for natural content

## 10. JPEG XL Squeeze Transform

**Alakuijala, J. et al. (2024). "JPEG XL Image Coding System."
IEEE Transactions on Image Processing.**
- Paper: https://arxiv.org/pdf/2506.05987
- Modular mode: https://cloudinary.com/blog/jpeg-xls-modular-mode-explained
- Squeeze transform splits image into low-frequency (downscaled) + residual channels
- Applied repeatedly to build a resolution pyramid
- Also includes chroma-from-luma mechanism
- Key difference: applies full squeeze/residual to ALL channels; ORIGAMI's advantage
  comes from NOT correcting chroma

## 11. FLIF and FUIF (Predecessors to JPEG XL Modular Mode)

**Sneyers, J. & Wuille, P. (2016). "FLIF: Free Lossless Image Format Based on MANIAC
Compression." IEEE ICIP 2016.**
- Paper: http://flif.info/papers/FLIF_ICIP16.pdf
- FUIF: https://github.com/cloudinary/fuif
- Blog: https://cloudinary.com/blog/fuif_why_do_we_need_a_new_image_file_format
- Progressive interlaced encoding where truncation yields lower-resolution image
- Similar pyramid structure with residuals for each successive resolution
- Key difference: applies to all channels equally

## 12. Luma Mapping with Chroma Scaling (LMCS) in VVC/H.266

**Pham, T.D. et al. (2020). "Luma Mapping with Chroma Scaling."
IEEE TCSVT.**
- Paper: https://ieeexplore.ieee.org/document/9105793/
- Maps luma code values and scales chroma residuals in luma-dependent fashion
- Asymmetric treatment of luma and chroma within a single resolution
- Key difference: not a resolution pyramid; chroma is still corrected

## 13. Efficient Chroma Sub-Sampling and Luma Modification

**Zhu, S., Cui, C. et al. (2019). "Efficient Chroma Sub-Sampling and Luma Modification
for Color Image Compression." IEEE TCSVT, 29(6), 1559-1563.**
- Paper: https://ieeexplore.ieee.org/document/8629276/
- Proposes modifying luma to compensate for chroma subsampling distortion
- Recognizes interplay between luma and chroma quality
- Key difference: within standard JPEG/HEVC framework, not pyramidal

## 14. Flexible Luma-Chroma Bit Allocation in Learned Compression

**Bao, Y. et al. (2023). "Flexible Luma-Chroma Bit Allocation in Learned Image
Compression." IEEE Signal Processing Letters.**
- Paper: https://ieeexplore.ieee.org/document/10017994/
- Allows adaptive bit allocation between luma and chroma at inference time
- Can increase Y PSNR at expense of chroma PSNR
- Key difference: still allocates SOME bits to chroma; single-resolution autoencoder

## 15. Dual-Layer Image Compression via Adaptive Downsampling

**Zhang, Y. & Wu, F. (2023). "Dual-layer Image Compression via Adaptive Downsampling
and Spatially Varying Upconversion."**
- Paper: https://arxiv.org/abs/2302.06096
- Low-res base + neural network upconversion + enhancement residuals
- Conceptually related base+enhancement architecture
- Key differences: neural network required; corrects all channels; single-hop

## 16. Color Learning for Image Compression

**Various authors (2023). "Color Learning for Image Compression."**
- Paper: https://arxiv.org/abs/2306.17460
- Divides compression into separate luminance and chrominance branches
- Uses CIEDE2000 loss for perceptually-optimized color
- Key difference: both branches at same resolution; both get full residual correction

## 17. Colorization-Based Image Compression

**Xiao, Y. et al. (2022). "Interactive Deep Colorization and its Application for Image
Compression." IEEE TVCG.**
- Paper: https://ieeexplore.ieee.org/document/9186041/
- Sends full-resolution grayscale + sparse color hints; reconstructs via deep colorization
- **Closest conceptual family** to ORIGAMI's approach
- Key differences: sends full-res grayscale (not low-res RGB); requires neural network

**Crayon (2025). "Convolutional Deep Colorization for Image Compression: A Color Grid
Based Approach."**
- Paper: https://arxiv.org/abs/2502.05402
- Retains every nth pixel's color; uses U-Net to reconstruct chrominance
- Key differences: sparse color hints at full resolution; requires neural network

## 18. Video Compression for WSI Z-Stacks

**Mankavelil, A. et al. (2019). "Video compression to support the expansion of
whole-slide imaging into cytology." J. Medical Imaging, 6(4).**
- Paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC6921690/
- Uses HEVC to exploit redundancy across focal planes (z-stack) in WSIs
- Validates principle that cross-level redundancy in WSI yields compression gains
- Key difference: z-axis redundancy, not pyramid-level redundancy

## 19. Recent WSI Compression (2024-2026)

**WISE — Mao et al. (2025). "A Framework for Gigapixel Whole-Slide-Image Lossless
Compression." CVPR 2025.**
- Paper: https://arxiv.org/abs/2503.18074
- Lossless WSI compression via hierarchical projection coding; does NOT exploit
  inter-level pyramid redundancy

**AdaSlide — Lee et al. (2025). "Adaptive compression framework for giga-pixel whole
slide images." Nature Communications.**
- Paper: https://www.nature.com/articles/s41467-025-66889-0
- RL-based per-region adaptive quality; no inter-level prediction

**Fischer et al. (2025). "Enhanced Diagnostic Fidelity in Pathology WSI Compression
via Deep Learning."**
- Paper: https://arxiv.org/abs/2503.11350
- Deep feature similarity loss; single-resolution compression

**PathAE (2025). "Pathology Image Compression with Pre-trained Autoencoders."
MICCAI 2025.**
- Paper: https://arxiv.org/abs/2503.11591
- Repurposes Stable Diffusion autoencoders; single-tile compression

**Fischer et al. (2025). "Unlocking the Potential of Digital Pathology: Novel Baselines
for Compression."**
- Paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC11889581/
- Comprehensive evaluation of JPEG, JPEG2000, JPEG-XL, HEVC, VVC for WSI

**Stain Deconvolution — Fischer et al. (2024). "Learned Image Compression for HE-stained
Histopathological Images via Stain Deconvolution." MICCAI 2024.**
- Paper: https://arxiv.org/abs/2406.12623
- Decomposes H&E images into RGB+HED channels; exploits domain-specific color structure

## 20. Chroma Quality and Perception

**Halide (2024). "Same Image, Different Score?"**
- Blog: https://halide.cx/blog/chroma-handling/
- Demonstrates how chroma upsampling filter quality and subsampling mode significantly
  affect perceptual metrics

**Nigel Tao (2024). "JPEG Chroma Upsampling."**
- Blog: https://nigeltao.github.io/blog/2024/jpeg-chroma-upsampling.html
- Detailed analysis of JPEG chroma upsampling filter choices and visual quality impact

## 21. Additional Related Work

**Reciprocal Pyramid Network — (2023). "Exploring Resolution Fields for Scalable Image
Compression with Uncertainty Guidance."**
- Paper: https://arxiv.org/abs/2306.08941
- Cross-resolution context mining between pyramid levels; neural compression

**Adaptive Resolution and Chroma Subsampling — (2026).**
- Paper: https://arxiv.org/html/2602.06100
- ARCS framework jointly optimizing spatial resolution and chroma subsampling

**Bilinear Interpolation Chroma Subsampling — (2022). "An effective bilinear
interpolation-based iterative chroma subsampling method for color images."
Multimedia Tools and Applications.**
- Paper: https://dl.acm.org/doi/10.1007/s11042-022-12743-0

**US Patent 8,374,446 (2013). "Encoding and decoding of digital signals based on
compression of hierarchical pyramid."**
- Patent: https://patents.justia.com/patent/8374446
- General hierarchical pyramid compression; does not describe luma-only residuals

---

## Novelty Assessment Summary

| Element | Found In Prior Art? | Closest Match |
|---------|:-------------------:|---------------|
| Resolution pyramid with predict-then-correct | Yes | Laplacian pyramid, JPEG hierarchical, SVC, JPEG XL squeeze |
| Luma-only residuals (no chroma correction) | **No** | All prior systems correct all channels |
| Chroma from low-res RGB prior via bilinear upsampling | **Partial** | AV1 CfL (block level, not pyramid level) |
| Ultra-aggressive chroma ratio (1/16 pixels) improving color fidelity | **No** | Convention limits to 4:2:0 (1/4 pixels) |
| Multi-hop pyramid (L2→L1→L0) with chroma never corrected | **No** | — |
| Tile-independent decoding for HTTP serving | **No** | SHVC/SVC have complex inter-tile dependencies |
| No neural network required at decode time | **Partial** | LCEVC (but corrects all channels) |
| Retrofits onto existing DZI tile pyramids | **No** | All prior systems require new format/codec |

**Conclusion**: The specific combination of elements in ORIGAMI — a multi-hop resolution
pyramid where enhancement layers carry ONLY luma residuals, with chroma derived entirely
from smooth bilinear interpolation of the low-resolution base, operating on independent
JPEG tiles within an existing DZI pyramid — does not appear in any published work from
1983-2026 found across ~80 targeted literature searches.
