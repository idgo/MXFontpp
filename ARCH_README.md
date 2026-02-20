# MXFont++ Training Deep Dive

A comprehensive guide to understanding the MXFont++ (Multiple Experts Font Generation) training pipeline — from high-level architecture to low-level implementation details.

---

## Table of Contents

1. [High-Level Overview](#1-high-level-overview)
2. [Core Idea: Style-Content Factorization](#2-core-idea-style-content-factorization)
3. [Overall Architecture Diagram](#3-overall-architecture-diagram)
4. [Data Preparation Pipeline](#4-data-preparation-pipeline)
5. [Model Architecture (Detail)](#5-model-architecture-detail)
6. [Training Loop Walkthrough](#6-training-loop-walkthrough)
7. [Loss Functions Explained](#7-loss-functions-explained)
8. [Configuration Reference](#8-configuration-reference)
9. [Inference Flow](#9-inference-flow)
10. [File Structure Map](#10-file-structure-map)

---

## 1. High-Level Overview

MXFont++ is a **few-shot font generation** model. Given a few reference images of a target font style, it can generate **any** character in that style — even characters it has never seen before.

```
┌─────────────────────────────────────────────────────────────────┐
│                     MXFont++ in One Sentence                    │
│                                                                 │
│  "Show me 3-4 characters in your handwriting, and I'll         │
│   generate thousands of characters in the same style."          │
└─────────────────────────────────────────────────────────────────┘
```

**Key Insight**: Chinese/CJK characters share structural components (radicals/strokes). MXFont++ learns to **separate style from content** so it can recombine them freely:

```
  Style (HOW it looks)     +     Content (WHAT character)     =    Generated Image
  ┌──────────────┐              ┌──────────────┐                  ┌──────────────┐
  │  手写风格     │              │     "龍"      │                  │  手写风格的龍 │
  │  (Handwritten │       +      │  (Dragon      │          =      │  (Handwritten │
  │   style)      │              │   structure)  │                  │   Dragon)     │
  └──────────────┘              └──────────────┘                  └──────────────┘
  From: Reference images        From: Source font                  Output: New!
```

---

## 2. Core Idea: Style-Content Factorization

The model's fundamental approach is **factorization** — decomposing font features into two independent factors:

```
                    ┌─────────────────────────────────────┐
                    │      Feature Factorization           │
                    │                                      │
                    │   Encoded    ┌──► Style Factor       │
                    │   Features ──┤   (Font appearance)   │
                    │              └──► Content Factor      │
                    │                  (Char structure)     │
                    └─────────────────────────────────────┘

  During Generation:
  ┌──────────────────────────────────────────────────────────────┐
  │                                                              │
  │  Reference Images ──encode──► factorize ──► Style Factors    │
  │  (same font,                                    │            │
  │   different chars)                              ├──► Combine ──► Decode ──► Output
  │                                                  │            │
  │  Source Images   ──encode──► factorize ──► Content Factors   │
  │  (same char,                                                 │
  │   different fonts)                                           │
  │                                                              │
  └──────────────────────────────────────────────────────────────┘
```

### The Chinese Character Decomposition System

Characters are decomposed into **primitive components (primals)**:

```
  Character "想" is decomposed into: ["木", "目", "心"]
  Character "明" is decomposed into: ["日", "月"]
  Character "龍" is decomposed into: ["立", "月", "三", "己", "彡"]
```

- **`chn_primals.json`**: ~373 primitive components (radicals, strokes, basic shapes)
- **`chn_decomposition.json`**: Maps ~20,000+ characters to their primal components

This decomposition is used by the **Auxiliary Classifier** to ensure the model learns meaningful component-aware features.

---

## 3. Overall Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        MXFont++ COMPLETE ARCHITECTURE                       │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐                                                             │
│  │ Style Images │ ─(B, n_s, 1, 128, 128)─┐                                  │
│  │ (Same font,  │                         │  flatten                         │
│  │  diff chars) │                         ▼                                  │
│  └─────────────┘                   ┌──────────────┐    ┌──────────────────┐  │
│                                    │    Style      │    │    8 Parallel    │  │
│                                    │   Encoder     │───►│    Experts       │  │
│                                    │ (CNN+GC+CBAM) │    │ (ResBlk+HAA)    │  │
│  ┌─────────────┐                   └──────────────┘    └───────┬──────────┘  │
│  │  Char Images │ ─(B, n_c, 1, 128, 128)─┐                    │             │
│  │ (Same char,  │                         │  flatten           │             │
│  │  diff fonts) │                         ▼                    │             │
│  └─────────────┘                   ┌──────────────┐    ┌───────┴──────────┐  │
│                                    │    Style      │    │    8 Parallel    │  │
│                                    │   Encoder     │───►│    Experts       │  │
│                                    │   (shared)    │    │    (shared)      │  │
│                                    └──────────────┘    └───────┬──────────┘  │
│                                                                │             │
│                     ┌──────────────────────────────────────────┘             │
│                     │                                                        │
│                     ▼                                                        │
│          ┌─────────────────────┐                                             │
│          │   FACTORIZE (1x1    │                                             │
│          │   Conv per expert)  │                                             │
│          ├─────────┬───────────┤                                             │
│          │ emb_dim │ emb_dim   │                                             │
│          │   = 0   │   = 1     │                                             │
│          │ (Style) │ (Content) │                                             │
│          └────┬────┴─────┬─────┘                                             │
│               │          │                                                   │
│    Style imgs─┘          └─Char imgs                                         │
│    get Style              get Content                                        │
│    factor                  factor                                            │
│       │                      │                                               │
│       ▼                      ▼                                               │
│  ┌─────────┐           ┌──────────┐                                          │
│  │  Mean   │           │   Mean   │      (Average over n_s / n_c samples)    │
│  │ Pooling │           │  Pooling │                                          │
│  └────┬────┘           └─────┬────┘                                          │
│       │                      │                                               │
│       └──────────┬───────────┘                                               │
│                  ▼                                                            │
│       ┌─────────────────────┐                                                │
│       │   DEFACTORIZE       │  Concatenate + Reconstruct (1x1 Conv)          │
│       │   (Combine factors) │                                                │
│       └──────────┬──────────┘                                                │
│                  │                                                            │
│                  ▼                                                            │
│       ┌─────────────────────┐                                                │
│       │      DECODER        │  Progressive Upsampling:                       │
│       │  (ResBlk + CBAM +   │  16x16 → 32x32 → 64x64 → 128x128             │
│       │   Upsample + Skip)  │  + Skip connection from Experts               │
│       └──────────┬──────────┘                                                │
│                  │                                                            │
│                  ▼                                                            │
│       ┌─────────────────────┐                                                │
│       │  Generated Image    │  (B, 1, 128, 128)                              │
│       │  (Target font+char) │                                                │
│       └─────────────────────┘                                                │
│                                                                              │
│  ═══════════════════════════ LOSSES ═════════════════════════════════════     │
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Discriminator │  │  Pixel (L1)  │  │  Feature     │  │ Aux Classifier   │  │
│  │ (Hinge GAN)   │  │  Loss        │  │  Matching    │  │ (Style + Comp)   │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────────┘  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────────────┐    │
│  │ HSIC Expert  │  │  HSIC Factor │  │  Contrastive Loss               │    │
│  │ Independence │  │ Independence │  │  (Style ≠ Content separation)   │    │
│  └──────────────┘  └──────────────┘  └──────────────────────────────────┘    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Data Preparation Pipeline

### 4.1 Two Dataset Modes

```
┌────────────────────────────────────────────────────────────────────┐
│                    DATASET MODES                                   │
├──────────────────────┬─────────────────────────────────────────────┤
│   TTF Mode           │   Image Folder Mode                        │
│   (dataset_type:ttf) │   (dataset_type:image)                     │
├──────────────────────┼─────────────────────────────────────────────┤
│                      │                                             │
│  data/ttfs/train/    │  data/images/train/                        │
│    ├── FontA.ttf     │    ├── FontA/                              │
│    ├── FontA.txt     │    │   ├── 一.png                          │
│    ├── FontB.ttf     │    │   ├── 二.png                          │
│    └── FontB.txt     │    │   └── ...                             │
│                      │    ├── FontB/                              │
│  .ttf = font file    │    │   ├── 一.png                          │
│  .txt = char list    │    │   └── ...                             │
│                      │    OR (flat structure):                    │
│  Renders on-the-fly  │    ├── 001_一.png                          │
│  from TTF fonts      │    ├── 002_二.png                          │
│                      │    └── ...                                 │
│                      │                                             │
│  Best for: Large-    │  Best for: Fine-tuning with custom         │
│  scale training      │  handwriting images                        │
│  from many fonts     │                                             │
└──────────────────────┴─────────────────────────────────────────────┘
```

### 4.2 Training Batch Construction

Each training sample creates a **triplet** of data. This is the key to learning style-content separation:

```
┌──────────────────────────────────────────────────────────────────────┐
│            HOW ONE TRAINING SAMPLE IS CONSTRUCTED                    │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Target: Font_A + "想"                                               │
│                                                                      │
│  ┌─── Style Images (n_in_s=3) ───┐    Same FONT, different CHARS    │
│  │  Font_A + "明"                 │                                  │
│  │  Font_A + "天"                 │    → Captures STYLE              │
│  │  Font_A + "水"                 │                                  │
│  └────────────────────────────────┘                                  │
│                                                                      │
│  ┌─── Char Images (n_in_c=3) ────┐    Same CHAR, different FONTS    │
│  │  Font_B + "想"                 │                                  │
│  │  Font_C + "想"                 │    → Captures CONTENT            │
│  │  Font_D + "想"                 │                                  │
│  └────────────────────────────────┘                                  │
│                                                                      │
│  ┌─── Target Image ──────────────┐                                   │
│  │  Font_A + "想"                 │    → Ground truth to compare      │
│  └────────────────────────────────┘                                  │
│                                                                      │
│  Model's goal: Combine Style(Font_A) + Content("想") = Target        │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 4.3 Data Transforms

```
Training Transform:
  Raw Image (128x128) → [Optional: RandomAffine] → Resize(128x128) → ToTensor → Normalize([0.5],[0.5])
                          (degrees=10,                                            (scales to [-1,1])
                           translate=3%,
                           scale=0.9-1.1,
                           shear=10)

Validation Transform:
  Raw Image (128x128) → Resize(128x128) → ToTensor → Normalize([0.5],[0.5])
```

### 4.4 Character Filtering

- Characters must exist in `chn_decomposition.json` (have known component breakdown)
- Characters must appear in **at least 2 fonts** (for cross-font training)
- Optional: `--fixed_char_txt` to restrict to a specific character set

---

## 5. Model Architecture (Detail)

### 5.1 Style Encoder

Extracts initial visual features from a character image.

```
Input: (B, 1, 128, 128)  ← Grayscale character image
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│  ConvBlock(1 → C, 3x3)           no norm, no activation │  C=32 → 32 channels
│                       (128x128)                          │
│  ConvBlock(C → C*2, 3x3)  ↓ downsample                  │  64 channels, 64x64
│                       (64x64)                            │
│  GCBlock(C*2)              Global Context Attention      │  Captures global style
│                       (64x64)                            │
│  ConvBlock(C*2 → C*4, 3x3)  ↓ downsample                │  128 channels, 32x32
│                       (32x32)                            │
│  CBAM(C*4)                 Channel + Spatial Attention   │  Focuses on important regions
└──────────────────────────────────────────────────────────┘
       │
       ▼
Output: (B, 128, 32, 32)
```

**Global Context Block (GCBlock)**: Learns a single global attention vector across the entire spatial feature map, then transforms and adds it back. This helps capture the overall "style feel" of the font.

**CBAM (Convolutional Block Attention Module)**: Applies channel attention (WHAT is important) then spatial attention (WHERE is important).

### 5.2 Experts Module

**8 parallel expert networks** process the same features independently, each specializing in different aspects.

```
Input: (B, 128, 32, 32)  from Style Encoder
       │
       ├─────── Expert 0 ──────── Expert 1 ──────── ... ──────── Expert 7
       │            │                  │                              │
       ▼            ▼                  ▼                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Each Expert:                                                               │
│                                                                             │
│  ResBlock(C*4 → C*4, 3x3)       (32x32)                                    │
│  CBAM(C*4)                       Attention                                  │
│  ResBlock(C*4 → C*4, 3x3)       (32x32) ─────── SKIP connection ──────►    │
│  ResBlock(C*4 → C*8, 3x3) ↓     downsample to (16x16)                      │
│  ResBlock(C*8 → C*8)            (16x16)                                     │
│  TransformerBlock(C*8)           HAA Self-Attention                          │
│  TransformerBlock(C*8)           HAA Self-Attention                          │
│                                                                             │
│  Output: "last" = (C*8, 16, 16) = (256, 16, 16)                            │
│          "skip" = (C*4, 32, 32) = (128, 32, 32)                            │
└─────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
Output: "last": (B, 8, 256, 16, 16)    ← Stacked from 8 experts
        "skip": (B, 8, 128, 32, 32)    ← Skip connections for decoder
```

### 5.3 HAA (Hybrid Attention Architecture)

Each expert uses **Transformer blocks** with a novel hybrid attention mechanism:

```
┌──────────────────────────────────────────────────────────────────────┐
│                   TransformerBlock                                    │
│                                                                      │
│  Input x ─────────────────────────┐                                  │
│      │                            │ (residual)                       │
│      ▼                            │                                  │
│  LayerNorm                        │                                  │
│      │                            │                                  │
│      ▼                            │                                  │
│  ┌─ Split channels in half ─┐     │                                  │
│  │                          │     │                                  │
│  ▼                          ▼     │                                  │
│  ┌────────────┐  ┌─────────────┐  │                                  │
│  │ Restormer  │  │  Custom     │  │                                  │
│  │ Attention  │  │  Attention  │  │                                  │
│  │ (Full res  │  │ (Downsampled│  │                                  │
│  │  Q, K, V)  │  │  K, V to   │  │                                  │
│  │            │  │  16x16)     │  │                                  │
│  └─────┬──────┘  └──────┬──────┘  │                                  │
│        │                │         │                                  │
│        └───── Concat ───┘         │                                  │
│              │                    │                                  │
│              └──────── Add ◄──────┘                                  │
│                    │                                                 │
│                    ├──────────────────────┐                           │
│                    │                      │ (residual)               │
│                    ▼                      │                           │
│               LayerNorm                   │                           │
│                    │                      │                           │
│                    ▼                      │                           │
│             Gated-DConv FFN              │                           │
│            (GELU gating +                │                           │
│             depthwise conv)              │                           │
│                    │                      │                           │
│                    └──────── Add ◄────────┘                          │
│                          │                                           │
│                          ▼                                           │
│                       Output                                         │
└──────────────────────────────────────────────────────────────────────┘
```

**Why two attention paths?**
- **Restormer path**: Full-resolution self-attention captures fine details
- **Custom path**: Downsampled K/V (to 16x16) captures global relationships efficiently

### 5.4 Factorization & Defactorization Blocks

This is the **core innovation** — separating style and content:

```
┌──────────────────────────────────────────────────────────────────────┐
│                    FACTORIZE (per expert)                             │
│                                                                      │
│  Input: Expert feature (C, H, W)                                     │
│      │                                                               │
│      ▼                                                               │
│  1x1 Conv → BN → LeakyReLU   (C → emb_num * C)                      │
│      │                         emb_num = 2                           │
│      ▼                                                               │
│  Reshape to (emb_num, C, H, W)                                       │
│      │                                                               │
│      ├── emb_dim=0 → Style Factor  (C, H, W)                        │
│      └── emb_dim=1 → Content Factor (C, H, W)                       │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│                    DEFACTORIZE (per expert)                           │
│                                                                      │
│  Input: [Style Factor, Content Factor]                               │
│      │                                                               │
│      ▼                                                               │
│  Concatenate along channel dim → (2*C, H, W)                        │
│      │                                                               │
│      ▼                                                               │
│  1x1 Conv → BN → LeakyReLU   (emb_num * C → C)                      │
│      │                                                               │
│      ▼                                                               │
│  Reconstructed feature (C, H, W)                                     │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 5.5 Decoder

Reconstructs the 128x128 image from expert features:

```
Input: "last" (B, 8, 256, 16, 16),  "skip" (B, 8, 128, 32, 32)
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│  Flatten experts: (B, 8*256, 16, 16) = (B, 2048, 16, 16)       │
│                                                                  │
│  ConvBlock(2048 → 256, 1x1)           Compress                  │  16x16
│  ResBlock(256 → 256, 3x3)                                       │  16x16
│  ResBlock(256 → 256, 3x3)                                       │  16x16
│  CBAM(256)                             Attention                 │  16x16
│  ResBlock(256 → 256, 3x3)                                       │  16x16
│  ConvBlock(256 → 128, 3x3) ↑          Upsample                  │  32x32
│                                                                  │
│  ──── SKIP CONNECTION ──── (concat with skip features)           │
│  Integrator: (8*128 → 128) + cat → (256, 32, 32)                │
│                                                                  │
│  ConvBlock(256 → 64, 3x3) ↑           Upsample                  │  64x64
│  ConvBlock(64 → 32, 3x3) ↑            Upsample                  │  128x128
│  CBAM(32)                              Attention                 │  128x128
│  ConvBlock(32 → 1, 3x3)               Final                     │  128x128
│  Sigmoid / Tanh                        Output activation         │
└──────────────────────────────────────────────────────────────────┘
       │
       ▼
Output: (B, 1, 128, 128)  ← Generated character image
```

### 5.6 Discriminator

Multi-task discriminator with projection:

```
Input: (B, 1, 128, 128)
       │
       ▼
┌──────────────────────────────────────────────────────┐
│  ConvBlock(1 → 32, stride=2)              64x64     │
│  ResBlock(32 → 64,  downsample)           32x32     │
│  ResBlock(64 → 128, downsample)           16x16     │  ← Feature
│  ResBlock(128 → 256, downsample)          8x8       │  ← Matching
│  ResBlock(256 → 512)                      8x8       │  ← (all layers
│  ResBlock(512 → 512)                      8x8       │     used for FM)
│  ReLU + AdaptiveAvgPool2d(1)              1x1       │
│                                                      │
│  ProjectionDiscriminator:                            │
│    Font embedding: n_fonts × 512                     │
│    Char embedding: n_chars × 512                     │
│    font_out = einsum(features, font_emb)             │
│    char_out = einsum(features, char_emb)             │
└──────────────────────────────────────────────────────┘
       │
       ▼
Output: [font_logits, char_logits]

Uses SPECTRAL NORMALIZATION on all layers for training stability.
```

### 5.7 Auxiliary Classifier

Classifies style and component from factorized features:

```
Input: Expert "last" features (256, 16, 16)
       │
       ▼
┌──────────────────────────────────────────────────┐
│  ResBlock(256 → 512, downsample)       8x8      │
│  ResBlock(512 → 512)                   8x8      │
│  AdaptiveAvgPool2d(1)                  1x1      │
│  Flatten                               512      │
│  Dropout(0.2)                                    │
│                                                  │
│  ┌─── Style Head ───┐  ┌─── Component Head ──┐  │
│  │ Linear(512 →      │  │ Linear(512 →        │  │
│  │   n_fonts)        │  │   n_primals)        │  │
│  └───────────────────┘  └─────────────────────┘  │
└──────────────────────────────────────────────────┘
       │
       ▼
Output: [style_logits, component_logits]
```

---

## 6. Training Loop Walkthrough

Here is the complete training loop, step by step:

```
┌──────────────────────────────────────────────────────────────────────┐
│                    TRAINING LOOP (per iteration)                     │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  STEP 1: FORWARD PASS                                                │
│  ━━━━━━━━━━━━━━━━━━━                                                │
│                                                                      │
│  1a. Encode all images through Style Encoder + Experts               │
│      style_feats = gen.encode(style_imgs)  → (B*3, 8, 256, 16, 16)  │
│      char_feats  = gen.encode(char_imgs)   → (B*3, 8, 256, 16, 16)  │
│                                                                      │
│  1b. Compute Expert Independence Loss (HSIC)                        │
│      → Encourages each expert to learn different things              │
│                                                                      │
│  1c. Factorize all features into style/content                       │
│      style_facts_s = gen.factorize(style_feats, dim=0)  ← style     │
│      style_facts_c = gen.factorize(style_feats, dim=1)  ← content   │
│      char_facts_s  = gen.factorize(char_feats, dim=0)   ← style     │
│      char_facts_c  = gen.factorize(char_feats, dim=1)   ← content   │
│                                                                      │
│  1d. Compute Contrastive Loss                                        │
│      → Style factors similar, Content factors similar (within group) │
│      → Style ≠ Content (between groups)                              │
│                                                                      │
│  1e. Compute Factor Independence Loss (HSIC)                        │
│      → Style factor ⊥ Content factor                                 │
│                                                                      │
│  1f. Generate image                                                  │
│      mean_style = average style_facts_s over n_s samples             │
│      mean_char  = average char_facts_c over n_c samples              │
│      gen_feats = gen.defactorize([mean_style, mean_char])            │
│      gen_imgs  = gen.decode(gen_feats)                               │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  STEP 2: UPDATE DISCRIMINATOR                                        │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━                                        │
│                                                                      │
│  2a. Forward real images through discriminator                       │
│  2b. Forward fake images (detached) through discriminator            │
│  2c. Compute hinge loss: max(0, 1-real) + max(0, 1+fake)            │
│  2d. Backprop and update discriminator                               │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  STEP 3: UPDATE GENERATOR                                            │
│  ━━━━━━━━━━━━━━━━━━━━━━━━                                           │
│                                                                      │
│  3a. Forward fake images through discriminator (again)               │
│  3b. Compute GAN generator loss: -mean(fake_logits)                  │
│  3c. Compute Feature Matching loss (L1 between disc features)        │
│  3d. Compute Pixel loss (L1 between gen_imgs and target)             │
│  3e. Compute Auxiliary Classifier losses:                            │
│      - Style factors → classify font (CE) + uniform component (KL)  │
│      - Content factors → classify component (CE) + uniform font (KL)│
│      - Generated image → same classification (frozen AC)             │
│  3f. Backprop AC, then backprop generator                            │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  STEP 4: EMA UPDATE                                                  │
│  ━━━━━━━━━━━━━━━━━                                                  │
│                                                                      │
│  gen_ema = 0.999 * gen_ema + 0.001 * gen                             │
│  (Smooth version of generator used for inference & validation)       │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  STEP 5: LOGGING & CHECKPOINTING                                    │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                                     │
│                                                                      │
│  Every print_freq:  Log losses and discriminator stats               │
│  Every val_freq:    Run validation, save comparison images           │
│  Every save_freq:   Save checkpoint (.pth)                           │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 7. Loss Functions Explained

### Summary Table

| Loss | Weight Config | What It Does | Applied To |
|------|:---:|---|---|
| **Pixel (L1)** | `pixel_w: 0.1` | Direct pixel comparison with ground truth | Generator |
| **GAN (Hinge)** | `gan_w: 1.0` | Adversarial realism | Generator + Discriminator |
| **Feature Matching** | `fm_w: 1.0` | Match discriminator intermediate features | Generator |
| **Aux Classifier** | `ac_w: 1.0` | Style/component classification of factors | Aux Classifier |
| **Aux Classifier (Gen)** | `ac_gen_w: 1.0` | Classification of generated images | Generator |
| **Aux Cross-Classifier** | `ac_cross_w: 1.0` | Uniform distribution (independence) | Generator |
| **Expert Independence** | `indp_exp_w: 1.0` | HSIC between expert pairs | Generator |
| **Factor Independence** | `indp_fact_w: 1.0` | HSIC between style/content factors | Generator |
| **Contrastive** | `contrast_loss_w: 1.0` | Cosine similarity separation | Generator |

### 7.1 Pixel Loss (L1)

```
L_pixel = |Generated_Image - Target_Image|
```

Simple but effective — directly penalizes pixel-level differences.

### 7.2 GAN Losses (Hinge)

```
Discriminator:  L_D = max(0, 1 - D(real)) + max(0, 1 + D(fake))
Generator:      L_G = -mean(D(fake))
```

The discriminator produces **two logits**: font classification and character classification (both trained adversarially).

### 7.3 Feature Matching Loss

```
L_FM = mean(|D_features(real) - D_features(fake)|)
```

Matches intermediate feature maps of the discriminator, stabilizing training.

### 7.4 Auxiliary Classifier Losses

The auxiliary classifier ensures factorized features are meaningful:

```
For STYLE factors:
  ✓ Should correctly classify FONT          → CrossEntropy(style_head, font_label)
  ✗ Should NOT encode component info        → KL(comp_head || Uniform)

For CONTENT factors:
  ✓ Should correctly classify COMPONENTS    → CrossEntropy(comp_head, primal_labels)
  ✗ Should NOT encode font info             → KL(style_head || Uniform)
```

### 7.5 Independence Losses (HSIC)

**HSIC (Hilbert-Schmidt Independence Criterion)** measures statistical independence:

```
Expert Independence:    HSIC(Expert_i, Expert_j) → 0   for all pairs i≠j
Factor Independence:    HSIC(Style_factor, Content_factor) → 0
```

Uses **RBF kernel** for computing HSIC. Encourages:
- Each expert to specialize in different features
- Style and content factors to be truly independent

### 7.6 Contrastive Loss

```
Minimize: cosine_similarity(Style_factors, Content_factors)
```

Further enforces that style and content live in different representation spaces.

---

## 8. Configuration Reference

### 8.1 `cfgs/defaults.yaml` — Full Parameter Reference

#### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_ddp` | `False` | Enable multi-GPU Distributed Data Parallel training |
| `port` | — | Port for DDP communication |
| `batch_size` | `8` | Training batch size |
| `max_iter` | `800000` | Total training iterations |
| `seed` | `2` | Random seed |
| `g_lr` | `2e-4` | Generator learning rate |
| `d_lr` | `1e-3` | Discriminator learning rate (higher = faster disc training) |
| `ac_lr` | `2e-4` | Auxiliary classifier learning rate |
| `adam_betas` | `[0.0, 0.9]` | Adam optimizer betas (β1=0 common in GANs) |
| `init` | `kaiming` | Weight initialization method |
| `n_workers` | `32` | DataLoader worker count |

#### Data Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dataset_type` | `ttf` | `"ttf"` for font files, `"image"` for image folders |
| `language` | `chn` | Language (Chinese) |
| `decomposition` | — | Path to `chn_decomposition.json` |
| `primals` | — | Path to `chn_primals.json` |
| `dset.train.data_dir` | — | Training data directory |
| `dset.train.n_in_s` | `3` | Number of style reference images per sample |
| `dset.train.n_in_c` | `3` | Number of content reference images per sample |
| `dset.val.data_dir` | — | Validation data directory |
| `dset.val.n_ref` | `4` | Number of reference images for validation |
| `dset.val.source_font` | — | Source font for validation rendering |
| `dset_aug.normalize` | `True` | Normalize to [-1, 1] (uses Tanh output) |
| `dset_aug.random_affine` | `False` | Random affine augmentation |

#### Model Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `C` | `32` | Base channel count (model width) |
| `g_args.style_enc.norm` | `in` | Style encoder normalization: Instance Norm |
| `g_args.experts.n_experts` | `8` | Number of parallel expert networks |
| `g_args.emb_num` | `2` | Embedding dimensions (style=0, content=1) |
| `g_args.dec.out` | `sigmoid` | Decoder output activation (`tanh` if normalized) |
| `d_args.w_norm` | `spectral` | Discriminator weight normalization |
| `ac_args.conv_dropout` | `0.3` | Aux classifier conv dropout |
| `ac_args.clf_dropout` | `0.2` | Aux classifier head dropout |

#### Loss Weights

| Parameter | Default | Effect When Increased |
|-----------|:-------:|---|
| `pixel_w` | `0.1` | Sharper but potentially blurrier outputs |
| `gan_w` | `1.0` | More realistic but potentially unstable |
| `fm_w` | `1.0` | More structurally consistent |
| `ac_w` | `1.0` | Better style/component separation |
| `ac_gen_w` | `1.0` | Generated images better classified |
| `ac_cross_w` | `1.0` | Stronger factor independence |
| `indp_exp_w` | `1.0` | More diverse experts |
| `indp_fact_w` | `1.0` | Stronger style-content separation |
| `contrast_loss_w` | `1.0` | Cleaner factor separation |

#### Logging & Saving

| Parameter | Default | Description |
|-----------|---------|-------------|
| `work_dir` | `./result` | Output directory |
| `print_freq` | `1000` | Print losses every N steps |
| `val_freq` | `10000` | Run validation every N steps |
| `save_freq` | `50000` | Save checkpoint every N steps |
| `save` | `all-last` | Save mode: `all` (step), `last`, or `all-last` |
| `resume` | — | Path to checkpoint to resume from |

### 8.2 Config Override

Configs are layered: `defaults.yaml` → `train.yaml` → command line:

```bash
# Basic training
python train.py cfgs/train.yaml

# Override via command line
python train.py cfgs/train.yaml --batch_size 16 --max_iter 400000

# Use fixed character set
python train.py cfgs/train.yaml --fixed_char_txt common_chars.txt
```

---

## 9. Inference Flow

```
┌──────────────────────────────────────────────────────────────────────┐
│                        INFERENCE PIPELINE                            │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. Load trained model (gen_ema weights from checkpoint)             │
│                                                                      │
│  2. Prepare reference images (3-4 chars in target style)             │
│     ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐                             │
│     │ "永" │ │ "東" │ │ "心" │ │ "明" │  ← Your handwriting         │
│     └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘                             │
│        │        │        │        │                                  │
│        └────────┴────────┴────────┘                                  │
│                    │                                                 │
│                    ▼                                                 │
│  3. Encode + Factorize → Extract Style Factors                       │
│     style_factors = mean(factorize(encode(refs), dim=0))             │
│                    │                                                 │
│                    ▼                                                 │
│  4. For each target character:                                       │
│     a. Render source char image (from standard font)                 │
│     b. Encode + Factorize → Extract Content Factor                   │
│        content_factor = factorize(encode(source), dim=1)             │
│     c. Combine: defactorize([style_factor, content_factor])          │
│     d. Decode → Generated character image                            │
│                    │                                                 │
│                    ▼                                                 │
│  5. Output: All characters in target style                           │
│     ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ...               │
│     │ "你" │ │ "好" │ │ "世" │ │ "界" │ │ "龍" │                    │
│     └──────┘ └──────┘ └──────┘ └──────┘ └──────┘                    │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 10. File Structure Map

```
MXFontpp_old/
├── train.py                    # Main training entry point
├── cfgs/
│   ├── defaults.yaml           # Default configuration (all parameters)
│   └── train.yaml              # Training-specific overrides
│
├── models/                     # Neural network architectures
│   ├── __init__.py             # Exports Generator, disc_builder, aux_clf_builder
│   ├── generator.py            # Generator: encode → factorize → defactorize → decode
│   ├── style_encoder.py        # CNN + GCBlock + CBAM feature extractor
│   ├── experts.py              # 8 parallel expert networks (ResBlk + HAA)
│   ├── haa.py                  # Hybrid Attention Architecture (Restormer-based)
│   ├── decoder.py              # Progressive upsampling decoder with skip connections
│   ├── discriminator.py        # Projection discriminator (font + char classification)
│   ├── aux_classifier.py       # Auxiliary classifier (style + component heads)
│   └── modules/
│       ├── __init__.py         # Exports all building blocks
│       ├── blocks.py           # ConvBlock, ResBlock, LinearBlock
│       ├── cbam.py             # Channel + Spatial Attention Module
│       ├── frn.py              # Filter Response Normalization (alternative to BN)
│       ├── globalcontext.py    # Global Context Block (for style encoder)
│       └── modules.py          # Weight initialization utilities
│
├── trainer/                    # Training logic
│   ├── __init__.py             # Exports FactTrainer, Evaluator, load_checkpoint
│   ├── base_trainer.py         # Base: GAN losses, pixel loss, FM loss, checkpointing
│   ├── fact_trainer.py         # Main trainer: factorization losses, AC losses, training loop
│   ├── criterions.py           # Loss functions: g_crit, d_crit, fm_crit
│   ├── evaluator.py            # Validation image generation
│   ├── trainer_utils.py        # Utilities: cyclize, load_checkpoint, expert_assign
│   └── hsic.py                 # HSIC independence criterion (RBF kernel)
│
├── datasets/                   # Data loading
│   ├── __init__.py             # Factory functions for train/val loaders
│   ├── ttf_dataset.py          # TTF font dataset (renders on-the-fly)
│   ├── imagefolder_dataset.py  # Image folder dataset (loads pre-rendered)
│   └── ttf_utils.py            # Font reading, character rendering, filtering
│
├── utils/                      # Utilities
│   ├── __init__.py             # Exports all utilities
│   ├── utils.py                # AverageMeter, freeze/unfreeze, tensor ops
│   ├── visualize.py            # Image grid creation, normalization
│   ├── writer.py               # DiskWriter, TBWriter for saving outputs
│   └── logger.py               # Colorized logging
│
├── data/
│   ├── chn_decomposition.json  # Character → primal components mapping
│   └── chn_primals.json        # List of ~373 primitive components
│
├── inference.ipynb             # Jupyter notebook for interactive inference
├── inference_utils.py          # FontGenerator class for inference
└── api.py                      # Flask API wrapper
```

---

## Quick Start

```bash
# 1. Train from TTF fonts
python train.py cfgs/train.yaml \
    --dset.train.data_dir data/ttfs/train \
    --dset.val.data_dir data/ttfs/val \
    --decomposition data/chn_decomposition.json \
    --primals data/chn_primals.json \
    --work_dir ./result

# 2. Train from image folders
python train.py cfgs/train.yaml \
    --dataset_type image \
    --dset.train.data_dir data/images/train \
    --dset.val.data_dir data/images/val \
    --decomposition data/chn_decomposition.json \
    --primals data/chn_primals.json \
    --work_dir ./result

# 3. Fine-tune from checkpoint
python train.py cfgs/train.yaml \
    --resume result/checkpoints/last.pth \
    --max_iter 900000

# 4. Train with specific characters only
python train.py cfgs/train.yaml --fixed_char_txt common_chars.txt
```

---

## Key Design Decisions Summary

| Decision | Why |
|----------|-----|
| **8 Experts** | Different experts learn different structural aspects (strokes, radicals, etc.) |
| **Factorization into 2 dims** | Clean separation: dim 0 = style (how), dim 1 = content (what) |
| **HSIC Independence** | Statistical guarantee that experts/factors don't duplicate info |
| **Hybrid Attention (HAA)** | Full-res for detail + downsampled for global context |
| **Projection Discriminator** | Efficient multi-task (font + char) discrimination |
| **EMA Generator** | Smoother, more stable outputs for inference |
| **Hinge GAN** | More stable than vanilla GAN, no log computation |
| **Spectral Norm** | Lipschitz constraint on discriminator for training stability |

---

*Based on MX-Font (NAVER Corp., MIT License) with modifications for the MXFont++ variant.*
