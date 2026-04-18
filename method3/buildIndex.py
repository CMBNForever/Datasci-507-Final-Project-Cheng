#!/usr/bin/env python3
"""
This file is used to build the embeded index from the scraped text files

Input:
data/knowledge/scraped/text/*.txt

Outputs:
data/knowledge/index/chunks.jsonl
data/knowledge/index/embeddings.npy
data/knowledge/index/meta.json

Run using the command:
python method3/buildIndex.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from retriever import build_index

def main() -> int:
    """
    Main function to build the index
    """
    try:
        build_index(ROOT)
    except Exception as e:
        print(e, file=sys.stderr)
        return 1
    return 0

if __name__ == "__main__":
    main()
