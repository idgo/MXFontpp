#!/usr/bin/env python3
"""
Generate font images from a character list using reference images.
- Base model is required.
- Optionally add one or more test models (--test-model can be repeated); output columns [Source] [Reference] [Base] [Test0] [Test1] ...
- If no test model is given, outputs only [Source] [Reference] [Base].

Usage:
  # Base model only (3-panel: Source, Reference, Base)
  python model_comparsion.py --base-model path/to/merged.pth

  # Base vs one or more test models (Source, Reference, Base, Test0, Test1, ...)
  python model_comparsion.py --base-model path/to/base.pth --test-model path/to/test1.pth
  python model_comparsion.py --base-model path/to/base.pth --test-model t1.pth,t2.pth,t3.pth

  # Custom ref, output, and character list
  python model_comparsion.py --base-model 340000.pth --ref-path ref_images/ --output-dir ./out --char-file common_chars.txt

  # Inline characters instead of file
  python model_comparsion.py --base-model merged.pth --chars "大家庭日前"

  # Also save per-character generated images (default: grid only)
  python model_comparsion.py --base-model merged.pth --save-images

  # Output image size (saved images and grid tiles; default 128)
  python model_comparsion.py --base-model merged.pth --output-size 256

  # All options
  python model_comparsion.py --base-model BASE.pth [--test-model T1.pth,T2.pth,...] [--ref-path DIR] [--chars STRING | --char-file FILE] [--source-font FONT] [--output-dir DIR] [--output-size N] [--save-images] [--batch-size N] [--gen-batch-size N]

  python model_comparsion.py --help   # list options and defaults
"""

from pathlib import Path
import argparse
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import math

from datasets import read_font
from inference_utils import FontGenerator, find_reference_image

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate font images; compare base model with optional test model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-model", required=True, help="Path to base model weights (required)")
    parser.add_argument("--test-model", default=None, metavar="PATH1,PATH2,...", help="Comma-separated test model paths; output columns [Base] [Test0] [Test1] ...")
    parser.add_argument("--ref-path", default=None, help="Path to reference images directory")
    char_group = parser.add_mutually_exclusive_group(required=True)
    char_group.add_argument("--chars", default=None, help="String of characters to generate")
    char_group.add_argument("--char-file", default=None, help="Path to text file with characters")
    parser.add_argument("--source-font", default=None, help="Source font for character structure")
    parser.add_argument("--output-dir", default="./out", help="Output directory")
    parser.add_argument("--output-size", type=int, default=128, help="Output image size in pixels (width and height); used for saved images and grid tiles")
    parser.add_argument("--save-images", action="store_true", help="Save each character's generated image only (default: only save grid)")
    parser.add_argument("--batch-size", type=int, default=40, help="Batch size for processing reference images")
    parser.add_argument("--gen-batch-size", type=int, default=300, help="Batch size for generating character images")
    return parser.parse_args()


def parse_test_model_paths(test_model_arg):
    """Parse comma-separated test model paths into a list (empty if None/empty)."""
    return [p.strip() for p in (test_model_arg or "").split(",") if p.strip()]


def load_chars_from_file(file_path):
    """Load characters from a text file (one line, or one char per line)."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return "".join(content.split())


def load_char_list(args):
    """Load and validate character list from --char-file or --chars. Returns (chars,). Raises on error."""
    if args.char_file is not None:
        if not Path(args.char_file).exists():
            raise FileNotFoundError(f"Char file not found: {args.char_file}")
        chars = load_chars_from_file(args.char_file)
        print(f"Loaded {len(chars)} characters from {args.char_file}")
    else:
        chars = args.chars or ""
        print(f"Using {len(chars)} characters from --chars")
    if not chars:
        raise ValueError("No characters to generate (--chars is empty or --char-file has no content)")
    print(f"Characters to generate ({len(chars)}): {chars[:50]}{'...' if len(chars) > 50 else ''}")
    return chars


def safe_char_for_filename(char):
    """Return a filesystem-safe string for a character (for use in filenames)."""
    return char if char not in r'\/:*?"<>|' else f"U{ord(char):04X}"


def tensor_to_pil(tensor):
    """Convert a tensor to PIL Image."""
    # Tensor is in range [-1, 1], convert to [0, 255]
    img = tensor.squeeze().cpu().numpy()
    img = ((img + 1) / 2 * 255).clip(0, 255).astype('uint8')
    return Image.fromarray(img, mode='L')


def create_composite_image(source_img, ref_img_path, base_generated_img, test_generated_imgs=None, img_size=128, padding=5):
    """
    Create a composite image: [Source] [Reference] [Base] [Test0] [Test1] ...
    test_generated_imgs: list of tensors (0 or more). No top labels.
    """
    test_generated_imgs = test_generated_imgs or []
    source_pil = tensor_to_pil(source_img)
    base_generated_pil = tensor_to_pil(base_generated_img)
    source_pil = source_pil.resize((img_size, img_size), Image.Resampling.LANCZOS)
    base_generated_pil = base_generated_pil.resize((img_size, img_size), Image.Resampling.LANCZOS)

    if ref_img_path and Path(ref_img_path).exists():
        ref_pil = Image.open(ref_img_path).convert('L').resize((img_size, img_size), Image.Resampling.LANCZOS)
    else:
        ref_pil = Image.new('L', (img_size, img_size), 255)

    num_cols = 3 + len(test_generated_imgs)
    total_width = img_size * num_cols + padding * (num_cols - 1)
    total_height = img_size
    composite = Image.new('L', (total_width, total_height), 255)

    composite.paste(source_pil, (0, 0))
    composite.paste(ref_pil, (img_size + padding, 0))
    composite.paste(base_generated_pil, (img_size * 2 + padding * 2, 0))
    for k, test_img in enumerate(test_generated_imgs):
        test_pil = tensor_to_pil(test_img)
        test_pil = test_pil.resize((img_size, img_size), Image.Resampling.LANCZOS)
        x = img_size * (3 + k) + padding * (3 + k)
        composite.paste(test_pil, (x, 0))
    return composite

def create_image_grid(source_imgs_tensor, base_generated_imgs_tensor, ref_paths_list, chars,
                     test_generated_imgs_tensors=None, ref_path="", img_size=64, padding=5, chars_per_row=10):
    """
    Create a grid image: each cell = one character with Source, Reference, Base, Test0, Test1, ... stacked.
    test_generated_imgs_tensors: list of tensors (0 or more), each shape [N, ...].
    """
    test_generated_imgs_tensors = test_generated_imgs_tensors or []
    num_chars = len(chars)
    num_rows_per_cell = 3 + len(test_generated_imgs_tensors)
    num_rows = math.ceil(num_chars / chars_per_row)
    cell_width = img_size + padding * 2
    cell_height = img_size * num_rows_per_cell + padding * (num_rows_per_cell + 1)
    row_label_width = 30
    total_width = row_label_width + chars_per_row * cell_width + padding * (chars_per_row + 1)
    total_height = num_rows * cell_height + padding
    composite = Image.new('RGB', (total_width, total_height), (255, 255, 255))
    draw = ImageDraw.Draw(composite)
    try:
        header_font = ImageFont.truetype("dejavu-sans/DejaVuSans.ttf", 14)
    except Exception:
        header_font = ImageFont.load_default() if hasattr(ImageFont, 'load_default') else None
    header_x = row_label_width - padding
    row_labels = ["S", "R", "B"] + [f"T{k}" for k in range(len(test_generated_imgs_tensors))]
    for r in range(num_rows_per_cell):
        y = padding + r * (img_size + padding) + img_size // 2
        label = row_labels[r]
        if header_font:
            draw.text((header_x, y), label, fill=(0, 0, 0), font=header_font, anchor="rm")
        else:
            draw.text((header_x, y), label, fill=(0, 0, 0), anchor="rm")
    for i, char in enumerate(chars):
        row = i // chars_per_row
        col = i % chars_per_row
        source_pil = tensor_to_pil(source_imgs_tensor[i]).resize((img_size, img_size), Image.Resampling.LANCZOS).convert('RGB')
        ref_img_path = find_reference_image(ref_path, char, ref_paths_list, i, len(chars))
        try:
            ref_pil = Image.open(ref_img_path).convert('RGB').resize((img_size, img_size), Image.Resampling.LANCZOS)
        except Exception:
            ref_pil = Image.new('RGB', (img_size, img_size), (255, 255, 255))
        base_generated_pil = tensor_to_pil(base_generated_imgs_tensor[i]).resize((img_size, img_size), Image.Resampling.LANCZOS).convert('RGB')
        cell_x = row_label_width + padding + col * cell_width + padding
        cell_y = padding + row * cell_height
        composite.paste(source_pil, (cell_x, cell_y))
        composite.paste(ref_pil, (cell_x, cell_y + img_size + padding))
        composite.paste(base_generated_pil, (cell_x, cell_y + 2 * (img_size + padding)))
        for k, test_tensor in enumerate(test_generated_imgs_tensors):
            test_pil = tensor_to_pil(test_tensor[i]).resize((img_size, img_size), Image.Resampling.LANCZOS).convert('RGB')
            composite.paste(test_pil, (cell_x, cell_y + (3 + k) * (img_size + padding)))
    return composite


def get_image_transform():
    """Return the transform used for model input (128x128, normalize)."""
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])


def init_generators(base_weight_path, test_model_paths, ref_path, batch_size):
    """
    Create base and test FontGenerators; extract style factors from ref_path.
    Returns (base_font_generator, test_font_generators, style_facts, ref_paths_list).
    """
    print("Initializing Base FontGenerator...")
    base_font_generator = FontGenerator(base_weight_path)
    print("Base FontGenerator initialized.")

    test_font_generators = []
    if test_model_paths:
        for p in test_model_paths:
            print(f"Initializing Test FontGenerator: {p}...")
            test_font_generators.append(FontGenerator(p))
        print(f"Initialized {len(test_font_generators)} test generator(s).")
        style_facts, ref_paths_list = test_font_generators[0].get_style_factors(ref_path, batch_size)
    else:
        style_facts, ref_paths_list = base_font_generator.get_style_factors(ref_path, batch_size)

    if style_facts is None:
        raise ValueError("Could not extract style factors from reference images.")
    return base_font_generator, test_font_generators, style_facts, ref_paths_list


def run_generation(base_font_generator, test_font_generators, style_facts, source_font_obj,
                  chars, gen_batch_size, transform):
    """
    Run batched generation for base and all test models.
    Returns (source_imgs_tensor, base_generated_imgs_tensor, test_generated_imgs_tensors).
    test_generated_imgs_tensors is a list of tensors (one per test model), or [].
    """
    char_batches = [chars[i:i + gen_batch_size] for i in range(0, len(chars), gen_batch_size)]
    all_source_imgs = []
    all_base_generated_imgs = []
    all_test_generated_imgs = [[] for _ in test_font_generators]

    for batch_idx, batch in enumerate(char_batches):
        print(f"  Processing batch {batch_idx + 1}/{len(char_batches)} ({len(batch)} chars)")
        if test_font_generators:
            _, base_generated_imgs = base_font_generator.generate_batched_char_images(
                style_facts, source_font_obj, batch, transform
            )
            source_imgs_batch, first_test_imgs = test_font_generators[0].generate_batched_char_images(
                style_facts, source_font_obj, batch, transform
            )
            all_source_imgs.append(source_imgs_batch)
            all_test_generated_imgs[0].append(first_test_imgs)
            for g in range(1, len(test_font_generators)):
                _, test_imgs = test_font_generators[g].generate_batched_char_images(
                    style_facts, source_font_obj, batch, transform
                )
                all_test_generated_imgs[g].append(test_imgs)
        else:
            source_imgs_batch, base_generated_imgs = base_font_generator.generate_batched_char_images(
                style_facts, source_font_obj, batch, transform
            )
            all_source_imgs.append(source_imgs_batch)
        all_base_generated_imgs.append(base_generated_imgs)
        torch.cuda.empty_cache()

    source_imgs_tensor = torch.cat(all_source_imgs)
    base_generated_imgs_tensor = torch.cat(all_base_generated_imgs)
    test_generated_imgs_tensors = (
        [torch.cat(imgs) for imgs in all_test_generated_imgs] if test_font_generators else []
    )
    return source_imgs_tensor, base_generated_imgs_tensor, test_generated_imgs_tensors


def _resize_if_needed(pil_img, output_size, model_size=128):
    """Resize PIL image to output_size if different from model_size."""
    if output_size != model_size:
        return pil_img.resize((output_size, output_size), Image.Resampling.LANCZOS)
    return pil_img


def save_generated_images(output_dir, chars, base_generated_imgs_tensor, test_generated_imgs_tensors,
                          output_size):
    """Save per-character generated images and char_mapping.txt when --save-images."""
    if test_generated_imgs_tensors:
        for i, char in enumerate(chars):
            safe_char = safe_char_for_filename(char)
            for k, gen_tensor in enumerate(test_generated_imgs_tensors):
                img = tensor_to_pil(gen_tensor[i])
                img = _resize_if_needed(img, output_size)
                img.save(output_dir / f"{i:04d}_{safe_char}_test{k}.png")
        total = len(test_generated_imgs_tensors) * len(chars)
        print(f"Done! Saved {total} generated images ({len(test_generated_imgs_tensors)} models x {len(chars)} chars) in {output_dir}")
    else:
        print(f"Saving {len(chars)} generated images ({output_size}x{output_size}) to {output_dir}...")
        for i, char in enumerate(chars):
            safe_char = safe_char_for_filename(char)
            img = tensor_to_pil(base_generated_imgs_tensor[i])
            img = _resize_if_needed(img, output_size)
            img.save(output_dir / f"{i:04d}_{safe_char}.png")
        print(f"Done! Saved {len(chars)} generated images in {output_dir}")

    mapping_path = output_dir / "char_mapping.txt"
    with open(mapping_path, "w", encoding="utf-8") as f:
        for i, char in enumerate(chars):
            safe_char = safe_char_for_filename(char)
            if test_generated_imgs_tensors:
                for k in range(len(test_generated_imgs_tensors)):
                    f.write(f"{i:04d}_{safe_char}_test{k}.png\t{char}\n")
            else:
                f.write(f"{i:04d}_{safe_char}.png\t{char}\n")
    print(f"Character mapping saved to {mapping_path}")


def save_grid(output_dir, source_imgs_tensor, base_generated_imgs_tensor, test_generated_imgs_tensors,
              ref_paths_list, chars, ref_path, output_size, chars_per_row=40):
    """Create and save the grid image to output_dir/all_results_grid.png."""
    grid_image = create_image_grid(
        source_imgs_tensor,
        base_generated_imgs_tensor,
        ref_paths_list,
        chars,
        test_generated_imgs_tensors=test_generated_imgs_tensors,
        ref_path=ref_path,
        img_size=output_size,
        padding=5,
        chars_per_row=chars_per_row,
    )
    grid_path = output_dir / "all_results_grid.png"
    grid_image.save(grid_path)
    print(f"Grid image saved: {grid_path.absolute()}")


def main():
    args = parse_args()
    base_weight_path = args.base_model
    test_model_paths = parse_test_model_paths(args.test_model)
    ref_path = args.ref_path
    source_font = args.source_font
    output_dir = Path(args.output_dir)
    output_size = args.output_size
    save_images = args.save_images
    batch_size = args.batch_size
    gen_batch_size = args.gen_batch_size

    chars = load_char_list(args)
    if source_font is None:
        raise ValueError("--source-font is required (path to a font file for character structure)")
    if not Path(source_font).exists():
        raise FileNotFoundError(f"Source font not found: {source_font}")

    base_font_generator, test_font_generators, style_facts, ref_paths_list = init_generators(
        base_weight_path, test_model_paths, ref_path, batch_size
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    transform = get_image_transform()
    source_font_obj = read_font(source_font)

    source_imgs_tensor, base_generated_imgs_tensor, test_generated_imgs_tensors = run_generation(
        base_font_generator,
        test_font_generators,
        style_facts,
        source_font_obj,
        chars,
        gen_batch_size,
        transform,
    )

    if save_images:
        save_generated_images(
            output_dir,
            chars,
            base_generated_imgs_tensor,
            test_generated_imgs_tensors,
            output_size,
        )
    else:
        print(f"Generated {len(chars)} characters (per-image save disabled; use --save-images to write each generated image).")

    save_grid(
        output_dir,
        source_imgs_tensor,
        base_generated_imgs_tensor,
        test_generated_imgs_tensors,
        ref_paths_list,
        chars,
        ref_path,
        output_size,
    )


if __name__ == "__main__":
    main()
