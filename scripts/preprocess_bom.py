#!/usr/bin/env python3
import re
import sys
from pathlib import Path

# ——— Configuration —————————————————————————————————————————————
INPUT_FILE  = Path("./data/bom.txt")
OUTPUT_FILE = Path("./data/bom_filtered.txt")
# ———————————————————————————————————————————————————————————————

def filter_bom(input_path: Path, output_path: Path):
    """
    Reads `input_path`, writes to `output_path` only those lines
    that start with digits, a colon, digits, then a space (e.g. "1:1 Text…").
    """
    verse_start_pattern = re.compile(r"^\d+:\d+ ")

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        in_verse = False
        for line in fin:
            if in_verse:
                if line.strip() == "":
                    in_verse = False
                    fout.write("\n")
                else:
                    fout.write(line[:-1] + " ")
            if verse_start_pattern.match(line):
                in_verse = True
                fout.write(line[:-1] + " ")

if __name__ == "__main__":
    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} not found.", file=sys.stderr)
        sys.exit(1)

    filter_bom(INPUT_FILE, OUTPUT_FILE)
    print(f"Filtered verses written to {OUTPUT_FILE}")
