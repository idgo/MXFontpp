
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

