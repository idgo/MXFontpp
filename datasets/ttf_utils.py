"""
MX-Font
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

from fontTools.ttLib import TTFont
from PIL import Image, ImageFont, ImageDraw
import numpy as np


def get_supported_chars(fontfile):
    """Extract the set of characters supported by the font."""
    try:
        font = TTFont(fontfile)
        cmap = font.getBestCmap()
        if cmap:
            return [chr(y) for y in set(cmap.keys())]
    except Exception as e:
        print(f"Error processing {fontfile}: {e}")
    return []

def get_defined_chars(fontfile):
    ttf = TTFont(fontfile)
    chars = [chr(y) for y in ttf["cmap"].tables[0].cmap.keys()]
    return chars


def get_filtered_chars(fontpath):
    ttf = read_font(fontpath)
    defined_chars = get_supported_chars(fontpath)
    avail_chars = []

    for char in defined_chars:
        img = np.array(render(ttf, char))
        if img.mean() == 255.:
            pass
        else:
            avail_chars.append(char.encode('utf-16', 'surrogatepass').decode('utf-16'))

    return avail_chars


def read_font(fontfile, size=150):
    font = ImageFont.truetype(str(fontfile), size=size)
    return font


def render(font, char, size=(128, 128), pad=20):
    """
    Render a single character to a square grayscale image, keeping the glyph centered
    and avoiding truncation for fonts with non-zero origin/baseline offsets (e.g. simsun.ttc).
    """
    # Pillow 10+ removed font.getsize(); prefer getbbox when available for accurate bounds.
    if hasattr(font, "getbbox"):
        left, top, right, bottom = font.getbbox(char)
        width, height = right - left, bottom - top
    else:
        width, height = font.getsize(char)
        left, top = 0, 0

    max_side = max(width, height)

    # Create a square canvas with padding, then center the glyph's bounding box.
    img_size = max_side + pad * 2
    img = Image.new("L", (img_size, img_size), 255)
    draw = ImageDraw.Draw(img)

    # Center the glyph bbox inside the square and compensate for left/top offsets
    x = pad + (max_side - width) // 2 - left
    y = pad + (max_side - height) // 2 - top

    draw.text((x, y), char, font=font)
    img = img.resize(size, 2)
    return img


# def render(font, char, size=(128,128),pad=20):
#     image_resolution = size[0]
    
#     image = Image.new('L', (image_resolution, image_resolution), color='white')
#     draw = ImageDraw.Draw(image)
    
#     # Calculate the position to center the character in the image
#     text_width, text_height = draw.textsize(char, font)
#     x = (image_resolution - text_width) // 2
#     y = (image_resolution - text_height) // 1.2
    
#     draw.text((x, y), char, font=font, fill='black')
#     return image