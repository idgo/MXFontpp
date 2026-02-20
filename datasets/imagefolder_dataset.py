"""
MX-Font
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

from pathlib import Path
from itertools import chain
import json
import random
from PIL import Image

import torch
from torch.utils.data import Dataset

from .ttf_utils import get_filtered_chars, read_font, render


def sample(population, k):
    if len(population) < k:
        sampler = random.choices
    else:
        sampler = random.sample
    sampled = sampler(population, k=k)
    return sampled


class ImageTrainDataset(Dataset):
    """Training dataset that loads character images from folders.
    
    Supports TWO folder structures:
    
    1. Multi-font structure (recommended, requires at least 2 fonts):
        data_dir/
            font_name1/
                char1.png (filename is the character, e.g., "一.png")
                char2.png
                ...
            font_name2/
                char1.png
                char2.png
                ...
    
    2. Flat structure with underscore naming (single font fine-tuning):
        data_dir/
            1_一.png  (format: {anything}_{char}.png, the char is extracted after last underscore)
            2_二.png
            ...
        
        For flat structure, set data_dir to parent folder and the folder name becomes the font name.
    
    Fine-tuning with source_font:
        When source_font is provided (path to a .ttf file), it is used to render
        char_imgs (content references) from a standard font. This enables effective
        single-font fine-tuning by providing cross-font content signals:
        - style_imgs: from your images (captures your handwriting style)
        - char_imgs:  from source_font (captures character structure)
        - target:     from your images (what the model learns to generate)
    """
    def __init__(self, data_dir, primals, decomposition, transform=None,
                 n_in_s=3, n_in_c=3, char_filter=None, extension="png",
                 source_font=None):
        
        self.data_dir = Path(data_dir)
        self.primals = primals
        self.decomposition = decomposition
        self.extension = extension
        
        # Load source font for content references (critical for single-font fine-tuning)
        self.source_font = None
        self._source_font_key = "__source__"
        if source_font:
            self.source_font = read_font(source_font)
            print(f"Loaded source font for content references: {source_font}")
        
        # Detect folder structure and load fonts/characters
        self.key_char_dict, self.char_to_path = self.load_data_list(self.data_dir, char_filter)
        
        if not self.key_char_dict:
            raise ValueError(
                f"No valid characters found in {data_dir}. "
                f"Expected structure:\n"
                f"  1. Multi-font: data_dir/font_name/char.png (e.g., data_dir/FontA/一.png)\n"
                f"  2. Flat: data_dir/prefix_char.png (e.g., data_dir/1_一.png)\n"
                f"Make sure characters exist in decomposition.json"
            )
        
        print("Character counts per font:")
        for key, chars in self.key_char_dict.items():
            print(f"  - {key}: {len(chars)} characters")
        
        # Build char -> font mapping
        self.char_key_dict = {}
        for key, charlist in self.key_char_dict.items():
            for char in charlist:
                self.char_key_dict.setdefault(char, []).append(key)
        
        # Check number of fonts and handle source_font
        n_image_fonts = len(self.key_char_dict)
        if n_image_fonts < 2:
            if self.source_font is not None:
                print(f"Fine-tuning mode: {n_image_fonts} image font(s) + source font for content references.")
                # No filter_chars needed — source font provides cross-font signal
            else:
                print(f"WARNING: Only {n_image_fonts} font(s) found and no source_font specified.")
                print("For effective fine-tuning, set source_font in config (e.g., source_font: path/to/NotoSans.ttf)")
                print("Falling back to same-font char_imgs (less effective).")
        else:
            # Multi-font: filter chars that appear in more than one font
            self.key_char_dict, self.char_key_dict = self.filter_chars()
        
        # Build data list (only from image fonts, not the source font)
        self.data_list = [(key, char) for key, chars in self.key_char_dict.items() for char in chars]
        self.keys = sorted(self.key_char_dict.keys())
        
        # If source_font is used, add it as a virtual font for font index tracking
        if self.source_font is not None and self._source_font_key not in self.keys:
            self.keys.append(self._source_font_key)
        
        self.chars = sorted(set.union(set(), *map(set, self.key_char_dict.values())))
        
        print(f"Total unique characters: {len(self.chars)}")
        print(f"Total unique fonts: {len(self.keys)}")
        print(f"Total unique data: {len(self.data_list)}")
        
        if len(self.data_list) == 0:
            raise ValueError(
                "No training data found after filtering. "
                "For multi-font training, each character must appear in at least 2 fonts. "
                "For single-font fine-tuning, set source_font in config."
            )
        
        self.transform = transform
        self.n_in_s = n_in_s
        self.n_in_c = n_in_c
        self.n_chars = len(self.chars)
        self.n_fonts = len(self.keys)
    
    def load_data_list(self, data_dir, char_filter=None):
        """Load available characters for each font folder.
        
        Returns:
            key_char_dict: {font_name: [char1, char2, ...]}
            char_to_path: {(font_name, char): image_path}
        """
        key_char_dict = {}
        char_to_path = {}
        
        # First, check for font subdirectories
        font_dirs = [x for x in data_dir.iterdir() if x.is_dir()]
        
        if font_dirs:
            # Multi-font structure: data_dir/font_name/char.png
            for font_dir in font_dirs:
                font_name = font_dir.name
                chars = []
                for img_path in font_dir.glob(f"*.{self.extension}"):
                    char = self._extract_char_from_filename(img_path.stem)
                    if char and len(char) == 1:
                        if char_filter is None or char in char_filter:
                            if char in self.decomposition:
                                chars.append(char)
                                char_to_path[(font_name, char)] = img_path
                if chars:
                    key_char_dict[font_name] = chars
        else:
            # Flat structure: data_dir contains images directly
            # Use parent folder name or "default" as font name
            font_name = data_dir.name if data_dir.name else "default"
            chars = []
            for img_path in data_dir.glob(f"*.{self.extension}"):
                char = self._extract_char_from_filename(img_path.stem)
                if char and len(char) == 1:
                    if char_filter is None or char in char_filter:
                        if char in self.decomposition:
                            chars.append(char)
                            char_to_path[(font_name, char)] = img_path
            if chars:
                key_char_dict[font_name] = chars
        
        return key_char_dict, char_to_path
    
    def _extract_char_from_filename(self, filename):
        """Extract character from filename.
        
        Supports:
            - Single char: "一" -> "一"
            - Underscore format: "123_一" -> "一"
        """
        if '_' in filename:
            # Format: prefix_char -> extract char after last underscore
            char = filename.rsplit('_', 1)[-1]
        else:
            char = filename
        return char if len(char) == 1 else None
    
    def filter_chars(self):
        """Filter characters that appear in more than one font."""
        char_key_dict = {}
        for char, keys in self.char_key_dict.items():
            if len(keys) > 1:
                char_key_dict[char] = keys
        
        filtered_chars = list(char_key_dict)
        key_char_dict = {}
        for key, chars in self.key_char_dict.items():
            filtered = list(set(chars).intersection(filtered_chars))
            if filtered:
                key_char_dict[key] = filtered
        
        return key_char_dict, char_key_dict
    
    def load_image(self, font_name, char):
        """Load and transform an image."""
        img_path = self.char_to_path[(font_name, char)]
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        return img
    
    def __getitem__(self, index):
        key, char = self.data_list[index]
        fidx = self.keys.index(key)
        cidx = self.chars.index(char)
        
        # Target image (user's font)
        trg_img = self.transform(self.load_image(key, char))
        trg_dec = [self.primals.index(x) for x in self.decomposition[char]]
        
        # Style images (same font, different chars) — captures style
        style_chars = sample([c for c in self.key_char_dict[key] if c != char], self.n_in_s)
        style_imgs = torch.stack([self.transform(self.load_image(key, c)) for c in style_chars])
        style_decs = [[self.primals.index(x) for x in self.decomposition[c]] for c in style_chars]
        
        # Char images (same char, different fonts) — captures content structure
        if self.source_font is not None:
            # Fine-tuning mode: render content references from source font
            char_imgs = torch.stack([self.transform(render(self.source_font, char))
                                     for _ in range(self.n_in_c)])
            source_fidx = self.keys.index(self._source_font_key)
            char_fids = [source_fidx] * self.n_in_c
        else:
            # Multi-font mode: use same char from different image fonts
            available_keys = [k for k in self.char_key_dict.get(char, [key]) if k != key]
            if not available_keys:
                # Fallback: use same font (less effective)
                available_keys = [key]
            char_keys = sample(available_keys, self.n_in_c)
            char_imgs = torch.stack([self.transform(self.load_image(k, char)) for k in char_keys])
            char_fids = [self.keys.index(k) for k in char_keys]
        
        char_decs = [trg_dec] * self.n_in_c
        
        ret = {
            "trg_imgs": trg_img,
            "trg_decs": trg_dec,
            "trg_fids": torch.LongTensor([fidx]),
            "trg_cids": torch.LongTensor([cidx]),
            "style_imgs": style_imgs,
            "style_decs": style_decs,
            "style_fids": torch.LongTensor([fidx] * self.n_in_s),
            "char_imgs": char_imgs,
            "char_decs": char_decs,
            "char_fids": torch.LongTensor(char_fids)
        }
        
        return ret
    
    def __len__(self):
        return len(self.data_list)
    
    @staticmethod
    def collate_fn(batch):
        _ret = {}
        for dp in batch:
            for key, value in dp.items():
                saved = _ret.get(key, [])
                _ret.update({key: saved + [value]})
        
        ret = {
            "trg_imgs": torch.stack(_ret["trg_imgs"]),
            "trg_decs": _ret["trg_decs"],
            "trg_fids": torch.cat(_ret["trg_fids"]),
            "trg_cids": torch.cat(_ret["trg_cids"]),
            "style_imgs": torch.stack(_ret["style_imgs"]),
            "style_decs": [*chain(*_ret["style_decs"])],
            "style_fids": torch.stack(_ret["style_fids"]),
            "char_imgs": torch.stack(_ret["char_imgs"]),
            "char_decs": [*chain(*_ret["char_decs"])],
            "char_fids": torch.stack(_ret["char_fids"])
        }
        
        return ret


class ImageValDataset(Dataset):
    """Validation dataset that loads character images from folders.
    
    Supports the same folder structures as ImageTrainDataset.
    """
    def __init__(self, data_dir, source_font=None, char_filter=None, n_ref=4, n_gen=20, 
                 transform=None, extension="png"):
        
        self.data_dir = Path(data_dir)
        self.source_font = read_font(source_font) if source_font is not None else None
        self.n_ref = n_ref
        self.n_gen = n_gen
        self.extension = extension
        
        # Load fonts and available characters
        self.key_char_dict, self.char_to_path = self.load_data_list(self.data_dir, char_filter)
        
        if not self.key_char_dict:
            raise ValueError(f"No valid characters found in {data_dir}")
        
        if self.source_font is None:
            self.char_key_dict = {}
            for key, charlist in self.key_char_dict.items():
                for char in charlist:
                    self.char_key_dict.setdefault(char, []).append(key)
            
            # Only filter if we have multiple fonts
            if len(self.key_char_dict) > 1:
                self.key_char_dict, self.char_key_dict = self.filter_chars()
        
        self.ref_chars, self.gen_chars = self.sample_ref_gen_chars(self.key_char_dict)
        self.gen_char_dict = {k: self.gen_chars for k in self.key_char_dict}
        self.data_list = [(key, char) for key, chars in self.gen_char_dict.items() for char in chars]
        self.transform = transform
    
    def load_data_list(self, data_dir, char_filter=None):
        """Load available characters for each font folder."""
        key_char_dict = {}
        char_to_path = {}
        
        # First, check for font subdirectories
        font_dirs = [x for x in data_dir.iterdir() if x.is_dir()]
        
        if font_dirs:
            # Multi-font structure
            for font_dir in font_dirs:
                font_name = font_dir.name
                chars = []
                for img_path in font_dir.glob(f"*.{self.extension}"):
                    char = self._extract_char_from_filename(img_path.stem)
                    if char and len(char) == 1:
                        if char_filter is None or char in char_filter:
                            chars.append(char)
                            char_to_path[(font_name, char)] = img_path
                if chars:
                    key_char_dict[font_name] = chars
        else:
            # Flat structure
            font_name = data_dir.name if data_dir.name else "default"
            chars = []
            for img_path in data_dir.glob(f"*.{self.extension}"):
                char = self._extract_char_from_filename(img_path.stem)
                if char and len(char) == 1:
                    if char_filter is None or char in char_filter:
                        chars.append(char)
                        char_to_path[(font_name, char)] = img_path
            if chars:
                key_char_dict[font_name] = chars
        
        return key_char_dict, char_to_path
    
    def _extract_char_from_filename(self, filename):
        """Extract character from filename."""
        if '_' in filename:
            char = filename.rsplit('_', 1)[-1]
        else:
            char = filename
        return char if len(char) == 1 else None
    
    def filter_chars(self):
        """Filter characters that appear in more than one font."""
        char_key_dict = {}
        for char, keys in self.char_key_dict.items():
            if len(keys) > 1:
                char_key_dict[char] = keys
        
        filtered_chars = list(char_key_dict)
        key_char_dict = {}
        for key, chars in self.key_char_dict.items():
            filtered = list(set(chars).intersection(filtered_chars))
            if filtered:
                key_char_dict[key] = filtered
        
        return key_char_dict, char_key_dict
    
    def sample_ref_gen_chars(self, key_char_dict):
        if not key_char_dict:
            return [], []
        common_chars = sorted(set.intersection(*map(set, key_char_dict.values())))
        if not common_chars:
            print("WARNING: No common characters in validation dataset. Using union of characters instead.")
            common_chars = sorted(set.union(*map(set, key_char_dict.values())))
        
        sampled_chars = sample(common_chars, self.n_ref + self.n_gen)
        ref_chars = sampled_chars[:self.n_ref]
        gen_chars = sampled_chars[self.n_ref:]
        
        return ref_chars, gen_chars
    
    def load_image(self, font_name, char):
        """Load and transform an image."""
        img_path = self.char_to_path[(font_name, char)]
        img = Image.open(img_path).convert('L')
        return img
    
    def __getitem__(self, index):
        key, char = self.data_list[index]
        
        ref_imgs = torch.stack([self.transform(self.load_image(key, c)) for c in self.ref_chars])
        
        if self.source_font is not None:
            source_img = self.transform(render(self.source_font, char))
        else:
            available_keys = self.char_key_dict.get(char, [key])
            source_key = random.choice(available_keys)
            source_img = self.transform(self.load_image(source_key, char))
        
        trg_img = self.transform(self.load_image(key, char))
        
        ret = {
            "style_imgs": ref_imgs,
            "source_imgs": source_img,
            "fonts": key,
            "chars": char,
            "trg_imgs": trg_img
        }
        
        return ret
    
    def __len__(self):
        return len(self.data_list)
    
    @staticmethod
    def collate_fn(batch):
        _ret = {}
        for dp in batch:
            for key, value in dp.items():
                saved = _ret.get(key, [])
                _ret.update({key: saved + [value]})
        
        ret = {
            "style_imgs": torch.stack(_ret["style_imgs"]),
            "source_imgs": torch.stack(_ret["source_imgs"]),
            "fonts": _ret["fonts"],
            "chars": _ret["chars"],
            "trg_imgs": torch.stack(_ret["trg_imgs"])
        }
        
        return ret


class ImageTestDataset(Dataset):
    def __init__(self, data_dir, source_font, gen_chars_file=None, transform=None, extension="png"):

        self.data_dir = Path(data_dir)
        self.source_font = read_font(source_font)
        self.gen_chars = get_filtered_chars(source_font)
        if gen_chars_file is not None:
            gen_chars = json.load(open(gen_chars_file))
            self.gen_chars = list(set(self.gen_chars).intersection(set(gen_chars)))

        #dict: {font_name: [ref img1 fp, ref img2 fp, ...], ...}
        self.font_ref_chars = self.load_data_list(self.data_dir, extension)

        self.gen_char_dict = {k: self.gen_chars for k in self.font_ref_chars}
        """
        list: [(font_name1, gen char1), (font_name1, gen char2), ...,
               (font_name2, gen char1), (font_name2, gen char2), ...,
               ......]
        """
        self.data_list = [(key, char) for key, chars in self.gen_char_dict.items() for char in chars]
        self.transform = transform

    def load_data_list(self, data_dir, extension):
        fonts = [x.name for x in data_dir.iterdir() if x.is_dir()]

        font_chars = {}
        for font in fonts:
            chars = [x.name for x in (self.data_dir / font).glob(f"*.{extension}")]
            font_chars[font] = chars
        return font_chars

    def __getitem__(self, index):
        font, char = self.data_list[index]
        ref_imgs = torch.stack([self.transform(Image.open(str(self.data_dir / font / f"{rc}")))
                                for rc in self.font_ref_chars[font]])
        source_img = self.transform(render(self.source_font, char))

        ret = {
            "style_imgs": ref_imgs,
            "source_imgs": source_img,
            "fonts": font,
            "chars": char,
        }

        return ret

    def __len__(self):
        return len(self.data_list)

    @staticmethod
    def collate_fn(batch):
        _ret = {}
        for dp in batch:
            for key, value in dp.items():
                saved = _ret.get(key, [])
                _ret.update({key: saved + [value]})

        ret = {
            "style_imgs": torch.stack(_ret["style_imgs"]),
            "source_imgs": torch.stack(_ret["source_imgs"]),
            "fonts": _ret["fonts"],
            "chars": _ret["chars"],
        }

        return ret
