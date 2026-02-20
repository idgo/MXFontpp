#!/usr/bin/env python3
"""
Script to generate all CJK extension text files with all characters in each Unicode range.
Generates characters from Unicode ranges directly (no input file needed).
Based on Unicode 17.0 / 中日韓統一表意文字 definitions.
"""

from pathlib import Path
from typing import Dict, Tuple


# CJK Extension ranges - simplified names for output files
# Maps output filename to (start, end) Unicode range
CJK_RANGES: Dict[str, Tuple[int, int]] = {
    'basic': (0x4e00, 0x9fff),      # CJK Unified Ideographs (Basic)
    'extA': (0x3400, 0x4dbf),       # CJK Extension A
    'extB': (0x20000, 0x2a6df),     # CJK Extension B
    'extC': (0x2a700, 0x2b73f),     # CJK Extension C
    'extD': (0x2b740, 0x2b81f),     # CJK Extension D
    'extE': (0x2b820, 0x2ceaf),     # CJK Extension E
    'extF': (0x2ceb0, 0x2ebef),     # CJK Extension F
    'extG': (0x30000, 0x3134f),     # CJK Extension G
    'extH': (0x31350, 0x323af),     # CJK Extension H
    'extI': (0x2ebf0, 0x2ee5f),     # CJK Extension I
    'extJ': (0x323b0, 0x3347f),     # CJK Extension J
}


def generate_chars_in_range(start: int, end: int) -> str:
    """
    Generate all valid characters in a Unicode range.
    Only includes characters that can be successfully encoded/decoded.
    """
    chars = []
    for code_point in range(start, end + 1):
        try:
            char = chr(code_point)
            # Verify the character can be encoded/decoded properly
            char.encode('utf-8').decode('utf-8')
            chars.append(char)
        except (ValueError, UnicodeEncodeError, UnicodeDecodeError):
            # Skip invalid code points
            continue
    return ''.join(chars)


def generate_cjk_extensions(output_dir: str = "cjk_ranges"):
    """
    Generate all CJK extension text files with all characters in each Unicode range.
    No input file needed - generates characters directly from Unicode ranges.
    
    Args:
        output_dir: Directory to save the output files
    """
    print("Generating CJK extension files from Unicode ranges...")
    print("This may take a while for large ranges...\n")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_path.absolute()}\n")
    
    # Generate files for each CJK range
    results = {}
    for range_name, (start, end) in CJK_RANGES.items():
        print(f"Generating {range_name}... [U+{start:05X}-U+{end:05X}]", end=" ", flush=True)
        
        chars = generate_chars_in_range(start, end)
        
        if chars:
            output_file = output_path / f"{range_name}.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(chars)
            
            results[range_name] = {
                'file': str(output_file),
                'count': len(chars),
                'start': start,
                'end': end,
            }
            print(f"✓ {len(chars):,} characters -> {output_file.name}")
        else:
            results[range_name] = {
                'file': None,
                'count': 0,
                'start': start,
                'end': end,
            }
            print("✗ No valid characters found")
    
    # Summary
    print(f"\n{'='*80}")
    print("Summary:")
    print(f"{'='*80}")
    total_chars = sum(r['count'] for r in results.values())
    files_generated = sum(1 for r in results.values() if r.get('file') is not None)
    
    print(f"Total characters generated: {total_chars:,}")
    print(f"Files generated: {files_generated}/{len(CJK_RANGES)}")
    print(f"\nFiles generated in: {output_path.absolute()}")
    
    # Show file sizes
    print(f"\nFile sizes:")
    for range_name, result in results.items():
        if result['file']:
            file_size = Path(result['file']).stat().st_size
            size_mb = file_size / (1024 * 1024)
            if size_mb >= 1:
                print(f"  {range_name}.txt: {size_mb:.2f} MB ({result['count']:,} chars)")
            else:
                size_kb = file_size / 1024
                print(f"  {range_name}.txt: {size_kb:.2f} KB ({result['count']:,} chars)")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate all CJK extension text files with all characters in each Unicode range"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="cjk_ranges",
        help="Output directory for generated files (default: cjk_ranges)"
    )
    
    args = parser.parse_args()
    
    generate_cjk_extensions(args.output_dir)
