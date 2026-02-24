# Fine-Tuning MXFont++

This guide describes how to fine-tune a pretrained MXFont++ model on your own character images using `cfgs/finetune.yaml`.

## Quick Start

```bash
python train.py cfgs/finetune.yaml
```

Before running, you must configure the config file (see below).

## Prerequisites

1. **Pretrained checkpoint** — Download from the [release](https://drive.google.com/drive/folders/1x1DahG0ilAnbL-8o6mq_C2fMas_udpYq?usp=drive_link) or train your own.
2. **Character images** — A folder of PNG images, one character per file.
3. **Source TTF font** — A standard font (e.g. SimSun, NotoSansTC) for content structure references.

## Configuration

Edit `cfgs/finetune.yaml` and set these required values:

| Option | Description |
|--------|-------------|
| `resume` | Path to pretrained checkpoint (e.g. `340000.pth`) |
| `work_dir` | Output directory for checkpoints and logs |
| `dset.train.data_dir` | Folder containing your character images |
| `dset.train.source_font` | Path to TTF font for content references |

### Image Folder Structure

Choose one of these layouts:

**Option A — Single character per file (recommended)**
```
my_images/
  一.png
  二.png
  三.png
  ...
```

**Option B — Underscore prefix**
```
my_images/
  001_一.png
  002_二.png
  ...
```

**Option C — Multi-font subfolders**
```
my_images/
  font_name/
    一.png
    二.png
    ...
```

Image filenames or subfolder names define the character. The model uses `chn_decomposition.json` to filter characters; only characters present in the decomposition file are used.

## Fine-Tuning Options

The `fine_tune` section controls which parts of the generator are updated:

| Option | Default | Effect |
|--------|---------|--------|
| `freeze_style_enc` | `false` | If `true`, keeps the pretrained style encoder frozen |
| `freeze_experts` | `false` | If `true`, keeps pretrained expert networks frozen |
| `freeze_fact_blocks` | `false` | If `true`, keeps factorization and reconstruction blocks frozen |

- Unfrozen modules continue training; frozen ones use pretrained weights.
- For small datasets, try freezing more (e.g. `freeze_style_enc: true`) to avoid overfitting.
- For larger, varied data, keep defaults (all unfrozen).

## Learning Rate & Schedule

Fine-tuning uses the same learning rates as full training by default. You can lower them if training is unstable:

- `g_lr`: Generator (default `2e-4`)
- `d_lr`: Discriminator (default `1e-3`)
- `ac_lr`: Auxiliary classifier (default `2e-4`)

With `max_iter: 10000`, fine-tuning typically needs far fewer iterations than full training.

## Fixed Character Set (Optional)

To restrict training to specific characters, pass a text file:

```bash
python train.py cfgs/finetune.yaml
```

`chars.txt` should list one character per line (or a single concatenated string). Only characters present in the decomposition file and in this list are used.

## Example Configuration

```yaml
resume: 340000.pth
work_dir: ./finetune_result
dataset_type: image # ttf is not ready yet. Please use image.

dset:
  train:
    data_dir: my_character_images # a list of images with `{number}_{char}.png` or `{char}.png`
    source_font: dataset/SimSun-01.ttf
    extension: png
    n_in_s: 3
    n_in_c: 3
  val:
    data_dir: my_character_images   # can reuse train folder
    extension: png

fine_tune:
  enabled: true # always true
  freeze_style_enc: false
  freeze_experts: false
  freeze_fact_blocks: false

batch_size: 6 # 6 => 23 GB GPU memory
max_iter: 10000 
```

## Outputs

- **Checkpoints**: saved in `work_dir/checkpoints/` (every `save_freq` iterations)
- **Validation images**: in `work_dir/images/`
- **Log**: `work_dir/log.log`

---

## Model Comparison (`model_comparsion.py`)

Use `model_comparsion.py` to visualize generated fonts and compare base vs fine-tuned (or multiple) models. It produces a grid: **[Source] [Reference] [Base] [Test0] [Test1] …**

### Requirements

- `--base-model`: Path to base/pretrained weights (required)
- `--ref-path`: Directory with reference images (style source; required)
- `--source-font`: TTF font for character structure (required)
- `--chars` **or** `--char-file`: Characters to generate (one required)

### Reference Image Layout

Same structure as eval: per-font subfolders, one character per PNG:

```
ref_images/
  font_name/
    一.png
    二.png
    ...
```

### Examples

```bash
# Base model only (3 columns: Source, Reference, Base)
python model_comparsion.py --base-model 340000.pth --ref-path ref_images/ --source-font dataset/SimSun-01.ttf --chars "大家庭日前"

# Compare base vs fine-tuned model
python model_comparsion.py --base-model 340000.pth --test-model finetune_result/checkpoints/10000.pth --ref-path ref_images/ --source-font dataset/SimSun-01.ttf --char-file common_chars.txt

# Multiple test models
python model_comparsion.py --base-model base.pth --test-model t1.pth,t2.pth,t3.pth --ref-path ref_images/ --source-font font.ttf --chars "測試文字"

# Save per-character images and custom output
python model_comparsion.py --base-model merged.pth --ref-path ref_images/ --source-font font.ttf --char-file chars.txt --output-dir ./out --output-size 256 --save-images
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--base-model` | (required) | Base model weights path |
| `--test-model` | — | Comma-separated test model paths (e.g. fine-tuned) |
| `--ref-path` | — | Reference images directory |
| `--source-font` | — | TTF font for character structure |
| `--chars` | — | Inline string of characters |
| `--char-file` | — | Text file with characters (one line or one per line) |
| `--output-dir` | `./out` | Output directory |
| `--output-size` | 128 | Grid tile and saved image size |
| `--save-images` | off | Save per-character generated images |
| `--batch-size` | 40 | Batch size for reference processing |
| `--gen-batch-size` | 300 | Batch size for generation |

Output: `{output_dir}/all_results_grid.png` (and per-image files if `--save-images`).
