
## Prerequisites

* **Python 3.8–3.11**
* **PyTorch >= 1.5** — [Install](https://pytorch.org/get-started/locally/) (choose CUDA if you have a GPU)

### Environment setup (conda, recommended)

```bash
# Create and activate the environment
conda env create -f environment.yml
conda activate mxfontpp
```

To use CUDA, install PyTorch with CUDA support after creating the env:

```bash
conda activate mxfontpp
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia  # adjust cuda version as needed
```

### Alternative: pip

```bash
pip install -r requirements.txt
```

Install PyTorch separately from [pytorch.org](https://pytorch.org/get-started/locally/) (match your CUDA version if using GPU).

## Web UI (Finetune Experiments, debug only)

A small web app to list fine-tuning experiment results, view logs/checkpoints/images, and start new runs. For local/debug use only; Flask is not in the main requirements.

1. **Install** (only when using the Web UI): `pip install flask`
2. **Run the server** from the project root:
   ```bash
   python -m webui.app
   ```
   Options: `--port 5000` (default), `--host 127.0.0.1`, `--experiments-root PATH`, `--project-root PATH`
3. **Open** http://127.0.0.1:5000 in a browser.
4. **List**: View all experiment folders under `finetune_result/` (or the configured experiments root).
5. **Detail**: Click an experiment to see its log, checkpoints (download), and validation images.
6. **New experiment**: Use "New experiment", set experiment name and dataset path (path to your character image folder; must be under the project directory), optionally epochs and fixed character list file, then "Start training". The run executes `finetuning.py` with the given options.

# Usage

Note that, we only provide the example font files; not the font files used for the training the provided weight *(generator.pth)*.
The example font files are downloaded from here. The ckpt can be found in [here](https://drive.google.com/drive/folders/1x1DahG0ilAnbL-8o6mq_C2fMas_udpYq?usp=drive_link).

## Preparing Data
* The examples of datasets are in *(./data)*

### Font files (.ttf)
* Prepare the TrueType font files(.ttf) to use for the training and the validation.
* Put the training font files and validation font files into separate directories.

### The text files containing the available characters of .ttf files (.txt)
* If you have the available character list of a .ttf file, save its available characters list to a text file (.txt) with the same name in the same directory with the ttf file.
    * (example) **TTF file**: data/ttfs/train/MaShanZheng-Regular.ttf, **its available characters**: data/ttfs/train/MaShanZheng-Regular.txt
* You can also generate the available characters files automatically using the `get_chars_from_ttf.py`
```
# Generating the available characters file

python get_chars_from_ttf.py --root_dir path/to/ttf/dir
```
* --root_dir: The root directory to find the .ttf files. All the .ttf files under this directory and its subdirectories will be processed.

### The json files with decomposition information (.json)
* The files for the decomposition information are needed.
    * The files for the Chinese characters are provided. (data/chn_decomposition.json, data/primals.json)
    * If you want to train the model with a language other than Chinese, the files for the decomposition rule (see below) are also needed.
        * **Decomposition rule**
            * structure: dict *(in json format)*
            * format: {char: [list of components]}
            * example: {'㐬': ['亠', '厶', '川'], '㐭': ['亠', '囗', '口']}
        * **Primals**
            * structure: list *(in json format)*
            * format: [**All** the components in the decomposition rule file]
            * example: ['亠', '厶', '川', '囗', '口']


## Training

### Modify the configuration file (cfgs/train.yaml)

```
- use_ddp:  whether to use DataDistributedParallel, for multi-GPUs.
- port:  the port for the DataDistributedParallel training.

- work_dir:  the directory to save checkpoints, validation images, and the log.
- decomposition:  path to the "decomposition rule" file.
- primals:  path to the "primals" file.

- dset:  (leave blank)
  - train:  (leave blank)
    - data_dir : path to .ttf files for the training
  - val: (leave blank)
    - data_dir : path to .ttf files for the validation
    - source_font : path to .ttf file used as the source font during the validation

```

### Run training
```
python train.py cfgs/train.yaml
```
* **arguments**
    * path/to/config (first argument): path to configration file.
    * \-\-resume (optional) : path to checkpoint to resume.

### Training log metrics

While training, the logger prints metrics every `print_freq` steps. The log uses three objects: **L** (losses), **D** (discriminator stats), and **S** (auxiliary classifier stats). Here is what each value means and how to read it.

#### Line 1: GAN losses and discriminator accuracies

| Label   | Meaning |
|--------|--------|
| **\|D** | **Discriminator loss.** Hinge loss: real logits should be > 1, fake < -1. Lower = discriminator is fitting. |
| **\|G** | **Generator adversarial loss.** Negative mean of fake logits (generator wants D to score fakes high). Lower = generator pushing fakes to look real. |
| **\|FM** | **Feature-matching loss.** L1 between real and fake *intermediate* discriminator features. Lower = generator features match real. |
| **\|R_font** | **Real font accuracy.** Fraction of *real* images where the font-head logit is **> 0**. Good D → near 1.0. |
| **\|F_font** | **Fake font accuracy.** Fraction of *fake* images where the font-head logit is **< 0**. Good D → near 1.0. |
| **\|R_uni** | Same as R_font but for the **character (unicode)** head on real images. |
| **\|F_uni** | Same as F_font but for the character head on fake images. |

The discriminator uses hinge loss: “correct” for real = logit > 0, for fake = logit < 0. For a healthy GAN, **R_font**, **F_font**, **R_uni**, **F_uni** should stay high (e.g. > 0.9). If they drift toward 0.5, the discriminator is not distinguishing real vs fake well.

#### Line 2: Auxiliary classifier on encoder (style/content)

These come from the **auxiliary classifier** on the **encoded** style/content factors (from real reference + character images).

| Label        | Meaning |
|-------------|--------|
| **\|AC_s**  | **Style (font) AC loss.** Cross-entropy for predicting *font id* from *style factors*. Lower = style factors encode font identity. |
| **\|AC_c**  | **Component (character) AC loss.** Cross-entropy for predicting *character/component* from *content factors*. Lower = content factors encode character identity. |
| **\|cr_AC_s** | **Cross (regularization) AC for style.** KL of style-head output toward *uniform* over fonts. |
| **\|cr_AC_c** | Same for the component head (uniform over components). |
| **\|AC_acc_s** | **Style classification accuracy** (encoder factors → font id). Higher = style factors identify font well. |
| **\|AC_acc_c** | **Component classification accuracy** (encoder factors → character). Higher = content factors identify character well. |

You want **AC_s**, **AC_c** to decrease and **AC_acc_s**, **AC_acc_c** to increase. The cross terms are regularizers.

#### Line 3: Auxiliary classifier on generated images

Same auxiliary classifier, but applied to **generated** images (via `gen_ema` encode → factorize). Measures whether **generated** glyphs preserve font and character identity.

| Label          | Meaning |
|----------------|--------|
| **\|AC_g_s**   | AC loss for *font* from *generated* image style factors. |
| **\|AC_g_c**   | AC loss for *character* from *generated* image content factors. |
| **\|cr_AC_g_s** | Uniform regularization for style on generated. |
| **\|cr_AC_g_c** | Uniform regularization for component on generated. |
| **\|AC_g_acc_s** | **Font accuracy on generated images.** High = generated glyphs look like the target font. |
| **\|AC_g_acc_c** | **Character accuracy on generated images.** High = generated glyphs preserve character identity. |

If **AC_g_acc_s** is low, the model is not preserving target font style; if **AC_g_acc_c** is low, it is not preserving character identity.

#### Line 4: Reconstruction and factorization

| Label        | Meaning |
|-------------|--------|
| **\|L1**    | **Pixel L1 loss** between *generated* and *target* images. Lower = better pixel-level match. |
| **\|INDP_EXP**  | **Expert independence (HSIC).** Dependence between *expert* feature maps in the encoder. Lower = experts more independent. |
| **\|INDP_FACT** | **Factor independence (HSIC).** Dependence between *style vs content* factors. Lower = style and content more disentangled. |

**L1** is the main reconstruction signal. **INDP_EXP** and **INDP_FACT** are regularization for disentanglement.

#### Summary: what to watch

1. **GAN balance:** D and G stable; R_font, F_font, R_uni, F_uni high (e.g. > 0.9).
2. **Factored representation:** AC_s, AC_c decreasing; AC_acc_s, AC_acc_c increasing.
3. **Generated glyph quality:** AC_g_acc_s and AC_g_acc_c increasing; L1 (pixel) decreasing.
4. **Disentanglement:** INDP_EXP and INDP_FACT low or decreasing.


### Test

### Preparing the reference images
* Prepare the reference images and the .ttf file to use as the source font.
* The reference images are should be placed in this format:

```
    * data_dir
    |-- font1
        |-- char1.png
        |-- char2.png
        |-- char3.png
    |-- font2
        |-- char1.png
        |-- char2.png
            .
            .
            .
```

* The names of the directory or the image files are not important, however, **the images with the same reference style are should be grouped with the same directory.**
* If you want to generate only specific characters, prepare the file containing the list of the characters to generate.
    * The example file is provided. (data/chn_gen.json)
    
### Modify the configuration file (cfgs/eval.yaml)

```
- dset:  (leave blank)
  - test:  (leave blank)
    - data_dir: path to reference images
    - source_font: path to .ttf file used as the source font during the generation
    - gen_chars_file: path to file of the characters to generate. Leave blank if you want to generate all the available characters in the source font.

```
    
### Run test
```
python eval.py \
    cfgs/eval.yaml \
    --weight generator.pth \
    --result_dir path/to/save/images
```
* **arguments**
  * path/to/config (first argument): path to configration file.
  * \-\-weight : path to saved weight to test.
  * \-\-result_dir: path to save generated images.
  
## Code license

This project is distributed under [MIT license](LICENSE), except [modules.py](models/modules/modules.py) which is adopted from https://github.com/NVlabs/FUNIT.

```
MX-Font++
Copyright (c) 2021-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

## Acknowledgement

This project is based on [clovaai/dmfont](https://github.com/clovaai/dmfont), [clovaai/mxfont](https://github.com/clovaai/mxfont) and [clovaai/lffont](https://github.com/clovaai/lffont).

