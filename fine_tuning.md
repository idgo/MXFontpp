# Fine-Tuning MXFont++

This guide describes how to fine-tune a pretrained MXFont++ model on your own character images using `cfgs/finetune.yaml`.



Before running, you must configure the config file (see below).

## Environment setup (conda)

```bash
conda env create -f environment.yml
conda activate mxfontpp
```

For GPU training, ensure PyTorch is installed with CUDA support. See [README Prerequisites](README.md#prerequisites) for details.



## Prerequisites

1. **Pretrained checkpoint** — Download from the [release](https://drive.google.com/drive/folders/1x1DahG0ilAnbL-8o6mq_C2fMas_udpYq?usp=drive_link) or train your own.
2. **Character images** — A folder of PNG images, one character per file. See [Generating Images from TTF](#generating-images-from-ttf) below if you have a font and need to create these.
3. **Source TTF font** — A standard font (e.g. SimSun, NotoSansTC) for content structure references.

## Generating Images from TTF

Use `scripts/generate_char_images.py` to render characters from a font file (TTF, OTF, WOFF, WOFF2) as PNGs. This produces the character image folder required for fine-tuning.

**Basic usage** (path to font is required):

```bash
# From specific characters
python scripts/generate_char_images.py source-font.ttf --chars "一二三四五" -o my_character_images

# From a character list file (one per line or one concatenated line)
python scripts/generate_char_images.py source-font.ttf --char-file chars.txt -o my_character_images

# Randomly sample CJK characters (up to 500 by default)
python scripts/generate_char_images.py source-font.ttf -o my_character_images

# Render all characters in the font
python scripts/generate_char_images.py source-font.ttf --all -o my_character_images
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--output-dir` / `-o` | `./char_images` | Output directory for PNGs |
| `--chars` | — | String of characters (e.g. `"你好"`) |
| `--char-file` | — | Text file with characters (one per line or concatenated) |
| `--all` | off | Render all characters in the font cmap |
| `--size` | 128 | Image width and height (128 matches model input) |
| `--prefix` | `""` | Filename prefix (e.g. `001_` → `001_一.png`) |
| `--max-count` | 500 | Maximum images to output (random mode) |
| `--delete` | off | Delete output directory before generating |

Output: one PNG per character, named `{char}.png` or `{prefix}{char}.png`. Only characters present in `chn_decomposition.json` will be used during fine-tuning.

**Example workflow** (generate images from a TTF, then fine-tune):

```bash
# 1. Generate character images from your font
python scripts/generate_char_images.py myfont.ttf --char-file data/common_chars.txt -o my_character_images

# 2. Set dset.train.data_dir to my_character_images in cfgs/finetune.yaml, then run
python finetuning.py cfgs/finetune.yaml --dataset_path my_character_images
```

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

## CLI Overrides

You can override config values from the command line:

| Option | Description |
|--------|-------------|
| `--epochs N` | Train for N epochs (overrides `max_iter`) |
| `--max_iter N` | Maximum training iterations |
| `--output_path PATH` | Override `work_dir` |
| `--dataset_path PATH` | Override `dset.train.data_dir` and `dset.val.data_dir` |

Precedence: `--epochs` overrides `--max_iter`, which overrides `max_iter` in the config.

Example: important command for finetuning

```bash
python finetuning.py cfgs/finetune.yaml --max_iter 5000 --output_path ./my_finetune --dataset_path my_character_images
```

## Example Configuration

```yaml
resume: 340000.pth # pre-trained model
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

Use `model_comparsion.py` to visualize generated fonts and compare base vs fine-tuned (or multiple) models. It produces a grid with columns **[Source] [Reference] [Base] [Test0] [Test1] …** (S, R, B, T0, T1... in row labels).

### Required Arguments

- `--base-model`: Path to the model to inteference
- `--ref-path`: Directory with reference images (style source)
- `--source-font`: Path to font file for character structure
- `--chars` **or** `--char-file`: Characters to generate (one required)

### Reference Image Layout

Place reference images in the ref-path directory. The script recursively finds PNG/JPG/JPEG files. Per-character lookup: `{char}.png` or any filename containing the character.

```
ref_images/
  一.png
  二.png
  三.png
  ...
```

Or use subfolders; the script uses `rglob` to find images.

### Examples

```bash
# Save per-character images and custom output
python model_comparsion.py --base-model merged.pth --ref-path ref_images/ --source-font font.ttf --char-file chars.txt --output-dir ./out --output-size 256 --save-images

# Clear output directory before writing
python model_comparsion.py --base-model merged.pth --ref-path ref_images/ --source-font font.ttf --chars "你好" --output-dir ./out --delete
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--base-model` | (required) | Base model weights path |
| `--test-model` | — | Comma-separated test model paths (e.g. fine-tuned) |
| `--ref-path` | — | Reference images directory (required) |
| `--source-font` | — | Font for character structure (required) |
| `--chars` | — | Inline string of characters |
| `--char-file` | — | Text file with characters (one line or concatenated) |
| `--output-dir` | `./out` | Output directory |
| `--output-size` | 128 | Grid tile and saved image size |
| `--save-images` | off | Save per-character generated images |
| `--batch-size` | 40 | Batch size for reference image processing |
| `--gen-batch-size` | 200 | Batch size for generation |
| `--delete` | off | Delete existing files in output dir before writing |

### Output

- **Grid**: `{output_dir}/all_results_grid.png` (always saved)
- **Per-character images** (with `--save-images`): `{idx:04d}_{char}.png` or `{idx:04d}_{char}_test{k}.png` for each test model
- **Character mapping**: `char_mapping.txt` (filename → character) when using `--save-images`
