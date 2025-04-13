#!/usr/bin/env python3
import sys
from pathlib import Path

# ——— Configuration —————————————————————————————————————————————
INPUT_FILE  = Path("./data/bible.txt")
OUTPUT_FILE = Path("./data/bible_filtered.txt")
# ———————————————————————————————————————————————————————————————

def filter_bible(input_path: Path, output_path: Path):
    """
    Reads `input_path`, writes to `output_path` a new line with spaces instead of tabs.
    """
    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            space_line = line.replace('\t', ' ')
            fout.write(space_line)
            

if __name__ == "__main__":
    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} not found.", file=sys.stderr)
        sys.exit(1)

    filter_bible(INPUT_FILE, OUTPUT_FILE)
    print(f"Filtered verses written to {OUTPUT_FILE}")
