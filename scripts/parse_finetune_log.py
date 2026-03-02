#!/usr/bin/env python3
"""Parse finetune log files to extract training metrics in structured format."""

import re
from pathlib import Path
from typing import Iterator


def strip_ansi_codes(text: str) -> str:
    """Remove ANSI escape codes from log lines."""
    ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
    return ansi_escape.sub('', text)


def parse_metrics_line(line: str) -> dict[str, str | float]:
    """Parse a metrics line containing |KEY    VALUE pairs.
    
    Example: |D             3.951 |G             0.156 |FM            0.156
    """
    pairs = re.findall(r'\|([A-Za-z0-9_]+)\s+([-?\d.]+%?)', line)
    result = {}
    for key, value in pairs:
        # Keep percentages as string (e.g., "100.0%") or convert numeric values to float
        if value.endswith('%'):
            result[key] = value
        else:
            try:
                result[key] = float(value)
            except ValueError:
                result[key] = value
    return result


def parse_step_line(line: str) -> dict[str, str | int] | None:
    """Parse Step info line: INFO::02/28 17:27:16 | Step    3700"""
    match = re.search(r'(\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2}).*?Step\s+(\d+)', line)
    if match:
        return {'timestamp': match.group(1), 'step': int(match.group(2))}
    return None


def parse_finetune_log(log_path: str | Path) -> Iterator[dict]:
    """Parse finetune log and yield extracted metric blocks.
    
    Each block contains:
    - step, timestamp: from the Step line
    - All keys from metric lines (D, G, FM, R_font, F_font, etc.)
    
    Yields:
        dict: Combined metrics for each step block
    """
    log_path = Path(log_path)
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")
    
    current_block: dict = {}
    
    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            clean_line = strip_ansi_codes(line.strip())
            
            # Check for Step line
            step_info = parse_step_line(clean_line)
            if step_info:
                if 'step' in current_block:
                    yield current_block
                current_block = dict(step_info)
                continue
            
            # Check for metric lines (start with |)
            if clean_line.startswith('|') and '|' in clean_line[1:]:
                metrics = parse_metrics_line(clean_line)
                current_block.update(metrics)
        
        if current_block:
            yield current_block


def display_metrics(metrics: dict) -> None:
    """Display key-value pairs in a readable format."""
    for key, value in metrics.items():
        print(f"  {key:12} = {value}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Parse finetune log and display metrics')
    parser.add_argument('log_path', nargs='?', default='finetune_result/u_font_2/log.log',
                        help='Path to log file')
    parser.add_argument('--step', type=int, help='Show only this step number')
    parser.add_argument('--last', action='store_true', help='Show only last block')
    args = parser.parse_args()
    
    log_path = Path(args.log_path)
    if not log_path.is_absolute():
        log_path = Path(__file__).parent.parent / log_path
    
    blocks = list(parse_finetune_log(log_path))
    
    if args.last and blocks:
        blocks = [blocks[-1]]
    elif args.step is not None:
        blocks = [b for b in blocks if b.get('step') == args.step]
    
    for i, block in enumerate(blocks):
        print(f"\n--- Step {block.get('step', '?')} @ {block.get('timestamp', '?')} ---")
        display_metrics(block)
    
    if not blocks:
        print("No matching blocks found.")


if __name__ == '__main__':
    main()
