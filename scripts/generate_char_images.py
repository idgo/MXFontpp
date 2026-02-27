#!/usr/bin/env python3
"""
Generate PNG images of characters from a font file (TTF, OTF, WOFF, WOFF2).
Default image size: 128x128 (or 1000x1000 when using --random).
Output: one PNG per character, named as {char}.png (with optional prefix).
"""

import argparse
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont


# Extensions that PIL can load directly
PIL_LOADABLE = {".ttf", ".otf"}
# All supported input extensions (fontTools can read these)
FONT_EXTENSIONS = {".ttf", ".otf", ".woff", ".woff2"}


def font_to_pil_loadable(font_path: str) -> Tuple[str, Optional[str]]:
    """
    Return (path_for_pil, temp_path_or_None).
    Converts WOFF/WOFF2 to a temporary TTF using fontTools so PIL can load it.
    If a temp file is created, caller should delete it when done.
    """
    path = Path(font_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Font file not found: {path}")
    suffix = path.suffix.lower()
    if suffix not in FONT_EXTENSIONS:
        raise ValueError(
            f"Unsupported font format: {suffix}. Use one of: {', '.join(FONT_EXTENSIONS)}"
        )
    if suffix in PIL_LOADABLE:
        return str(path), None

    # WOFF/WOFF2: load with fontTools and save as temp TTF
    font = TTFont(path)
    fd, tmp = tempfile.mkstemp(suffix=".ttf")
    os.close(fd)
    font.save(tmp)
    font.close()
    return tmp, tmp


def get_supported_chars(font_path: str) -> List[str]:
    """Return list of characters supported by the font (from cmap)."""
    font = TTFont(font_path)
    cmap = font.getBestCmap()
    font.close()
    if not cmap:
        return []
    return [chr(c) for c in sorted(cmap.keys())]


def is_cjk_unified(c: str) -> bool:
    """True if the character is in CJK Unified Ideographs (Chinese) blocks."""
    if len(c) != 1:
        return False
    cp = ord(c)
    # CJK Unified Ideographs (main block, common Chinese)
    if 0x4E00 <= cp <= 0x9FFF:
        return True
    # CJK Unified Ideographs Extension A
    if 0x3400 <= cp <= 0x4DBF:
        return True
    # CJK Compatibility Ideographs (common in fonts)
    if 0xF900 <= cp <= 0xFAFF:
        return True
    return False


def render_char(
    font: ImageFont.FreeTypeFont,
    char: str,
    size: Tuple[int, int] = (128, 128),
    pad: int = 5,
) -> Image.Image:
    """
    Render a single character, centered, to a grayscale image of the given size.

    The character is centered within the image with `pad` pixels margin on each side
    (font should be chosen so the glyph fits in the inner region).
    """
    width, height = size
    img = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(img)

    # Use textbbox when available (Pillow >= 8.0) for accurate bounds.
    if hasattr(draw, "textbbox"):
        left, top, right, bottom = draw.textbbox((0, 0), char, font=font)
    else:
        if hasattr(font, "getbbox"):
            left, top, right, bottom = font.getbbox(char)
        else:
            w, h = font.getsize(char)
            left, top, right, bottom = 0, 0, w, h

    text_w = right - left
    text_h = bottom - top

    # Center in the inner rectangle (pad from each edge).
    inner_w = width - 2 * pad
    inner_h = height - 2 * pad
    cx = pad + inner_w // 2
    cy = pad + inner_h // 2
    x = cx - (text_w // 2) - left
    y = cy - (text_h // 2) - top

    draw.text((x, y), char, font=font, fill=0)
    return img


def chars_from_args(
    chars_str: Optional[str],
    char_file: Optional[str],
    font_path: Optional[str],
    all_chars: bool,
    random_count: Optional[int] = None,
) -> List[str]:
    """Build the list of characters to generate from CLI args."""
    if random_count is not None:
        if not font_path:
            raise ValueError(
                "--random requires a font path to read the font's character set."
            )
        supported = get_supported_chars(font_path)
        # Restrict to Chinese (CJK Unified Ideographs) only; no English, symbols, etc.
        chinese_only = [c for c in supported if is_cjk_unified(c)]
        if not chinese_only:
            return []
        n = min(random_count, len(chinese_only))
        return random.sample(chinese_only, n)

    if all_chars:
        if not font_path:
            raise ValueError(
                "--all requires a font path to read the font's character set."
            )
        return get_supported_chars(font_path)

    if char_file:
        path = Path(char_file)
        if not path.exists():
            raise FileNotFoundError(f"Char file not found: {path}")
        text = path.read_text(encoding="utf-8")
        # One char per line, or one line of consecutive chars
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) == 1 and len(lines[0]) > 1:
            return list(lines[0])
        return list("".join(lines))

    if chars_str:
        return list(chars_str)

    return []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate PNG images of characters from a font (TTF, OTF, WOFF, WOFF2)."
    )
    parser.add_argument(
        "font",
        type=str,
        help="Path to font file (.ttf, .otf, .woff, .woff2)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="./char_images",
        help="Output directory for PNGs (default: ./char_images)",
    )
    parser.add_argument(
        "--chars",
        type=str,
        default=None,
        help="String of characters to render (e.g. 'ABC012' or '你好')",
    )
    parser.add_argument(
        "--char-file",
        type=str,
        default=None,
        help="Text file listing characters (one per line or one line of chars)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Render all characters supported by the font (from cmap)",
    )
    parser.add_argument(
        "--random",
        nargs="?",
        type=int,
        default=None,
        const=100,
        metavar="N",
        help="Randomly sample N Chinese (CJK) characters from the font (default: 100). English/symbols excluded.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=128,
        metavar="N",
        help="Image width and height in pixels (default: 128).",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Filename prefix for each PNG (e.g. 'U+' to get U+0041.png)",
    )
    parser.add_argument(
        "--max-count",
        type=int,
        default=None,
        metavar="N",
        help="Maximum number of images to output (cap on characters to render).",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete the output directory before generating images.",
    )
    args = parser.parse_args()

    font_path = args.font
    if not Path(font_path).exists():
        print(f"Error: font file not found: {font_path}", file=sys.stderr)
        sys.exit(1)

    chars = chars_from_args(
        args.chars,
        args.char_file,
        font_path,
        args.all,
        args.random,
    )
    if args.max_count is not None and args.max_count >= 0:
        chars = chars[: args.max_count]
    if not chars:
        print(
            "Error: no characters to generate. Use --chars, --char-file, --all, or --random.",
            file=sys.stderr,
        )
        sys.exit(1)

    out_dir = Path(args.output_dir)
    if args.delete and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    size = (args.size, args.size)
    prefix = args.prefix

    # Font point size for drawing; reduce when padding so glyph fits inside (size - 2*pad)
    RENDER_PAD = 10
    font_pt = max(72, int(args.size * 1.2))
    inner = max(1, args.size - 2 * RENDER_PAD)
    font_pt = max(72, int(font_pt * inner / args.size))
    temp_path: Optional[str] = None

    try:
        pil_path, temp_path = font_to_pil_loadable(font_path)
        font = ImageFont.truetype(pil_path, size=font_pt)
    except Exception as e:
        print(f"Error loading font: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        for char in chars:
            try:
                img = render_char(font, char, size=size, pad=RENDER_PAD)

                # File naming: {char}.png, with optional prefix; sanitize path separator.
                safe_char = char.replace(os.sep, "_")
                name = f"{prefix}{safe_char}.png"

                out_path = out_dir / name
                img.save(out_path, "PNG")
                print(out_path)
            except Exception as e:
                print(f"Skip {char!r}: {e}", file=sys.stderr)

        print(f"Done. Wrote {len(chars)} image(s) to {out_dir}")
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass


if __name__ == "__main__":
    main()
