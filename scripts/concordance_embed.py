#!/usr/bin/env python3
"""
concordance_embed.py

This script:
 1. Reads verse files (Bible and Book of Mormon) from ./data/ in the format:
    Book Chapter:Verse<SPACE>Text (one verse per line).
 2. Reads ./data/issues.csv with columns: issue, viewpoint1, viewpoint2[,viewpoint3].
 3. Uses the OpenAI Embeddings API (model text-embedding-3-large) to embed all verses, issues, and viewpoints in batches.
 4. Computes cosine similarity between each verse and each issue/viewpoint.
 5. Writes results to ./results/embed_concordance.csv with columns:
      is_bible, book, chapter, verse, text,
      <issue_slug>_embed_sim, <issue_slug>_viewpoint_1_embed_sim, ...

Prerequisites:
  - Python 3.8+
  - Install dependencies: pip install openai pandas numpy
  - Set environment variable OPENAI_API_KEY

Usage:
  python embed_concordance.py
"""
import os
import re
import json
import hashlib
import pandas as pd
import numpy as np
from openai import OpenAI
from tqdm import tqdm

# ─── Configuration ─────────────────────────────────────────────────────────────
API_KEY       = os.getenv("OPENAI_API_KEY")
EMBED_MODEL   = "text-embedding-3-large"
DATA_DIR      = "./data"
RESULT_DIR    = "./results"
CACHE_DIR     = "./cache"
BATCH_SIZE    = 512
# ────────────────────────────────────────────────────────────────────────────────

# Initialize OpenAI client
client = OpenAI(api_key=API_KEY)

# Ensure cache directory
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_filename(text: str) -> str:
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return os.path.join(CACHE_DIR, f"{h}.json")

def load_embedding_from_cache(text: str):
    fn = get_cache_filename(text)
    if os.path.exists(fn):
        with open(fn, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def save_embedding_to_cache(text: str, emb: list[float]):
    fn = get_cache_filename(text)
    with open(fn, "w", encoding="utf-8") as f:
        json.dump(emb, f)

def parse_verses(fn: str, is_bible: bool) -> list[dict]:
    verses = []
    with open(fn, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            loc, v_txt = line.split(":", 1)
            book, chapter = loc.rsplit(" ", 1)
            verse, txt = v_txt.split(" ", 1)
            verses.append({
                "is_bible": is_bible,
                "book": book,
                "chapter": chapter,
                "verse": verse,
                "text": txt
            })
    return verses

def batch_embed(texts: list[str]) -> np.ndarray:
    embs = []
    uncached = []
    uncached_idx = []

    # Load cache
    for i, text in enumerate(texts):
        cached = load_embedding_from_cache(text)
        if cached is not None:
            embs.append(cached)
        else:
            embs.append(None)  # placeholder
            uncached.append(text)
            uncached_idx.append(i)

    print(f"{len(uncached)} uncached / {len(texts)} total")

    # Batch embed uncached
    for i in tqdm(range(0, len(uncached), BATCH_SIZE), desc="Embedding new texts"):
        batch = uncached[i : i + BATCH_SIZE]
        resp = client.embeddings.create(
            model=EMBED_MODEL,
            input=batch
        )
        batch_embs = [d.embedding for d in resp.data]
        for j, emb in enumerate(batch_embs):
            idx = uncached_idx[i + j]
            embs[idx] = emb
            save_embedding_to_cache(texts[idx], emb)

    raise ValueError("let's see what's happened")
    return np.array(embs, dtype=np.float32)

def normalize(embs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / norms

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    issues_df = pd.read_csv(os.path.join(DATA_DIR, "issues.csv"))

    bible = parse_verses(os.path.join(DATA_DIR, "bible.txt"), True)
    bom    = parse_verses(os.path.join(DATA_DIR, "bom.txt"),   False)
    all_verses = bible + bom
    verse_texts = [v["text"] for v in all_verses]

    print("Embedding verses...")
    verse_embs = batch_embed(verse_texts)
    verse_embs = normalize(verse_embs)

    embed_items = []
    for _, row in issues_df.iterrows():
        issue_short = row["issue_short"]
        embed_items.append({"issue_short": issue_short, "text": row["issue"], "type": "issue"})
        for i in (1, 2, 3):
            vp = row.get(f"viewpoint{i}")
            if isinstance(vp, str) and vp.strip():
                embed_items.append({
                    "issue_short": f"{issue_short}_viewpoint_{i}",
                    "text": vp,
                    "type": "viewpoint"
                })

    item_texts = [item["text"] for item in embed_items]
    print("Embedding issues and viewpoints...")
    item_embs = batch_embed(item_texts)
    item_embs = normalize(item_embs)

    for item, emb in zip(embed_items, item_embs):
        item["emb"] = emb

    issue_emb_map = {it["issue_short"]: it["emb"] for it in embed_items if it["type"] == "issue"}
    vp_emb_map    = {it["issue_short"]: it["emb"] for it in embed_items if it["type"] == "viewpoint"}

    print("Computing similarities...")
    results = []
    for idx, verse in enumerate(all_verses):
        row = dict(verse)
        ve = verse_embs[idx]
        for issue_short, ie in issue_emb_map.items():
            row[f"{issue_short}_embed_sim"] = float(np.dot(ve, ie))
        for issue_short, vpe in vp_emb_map.items():
            row[f"{issue_short}_embed_sim"] = float(np.dot(ve, vpe))
        results.append(row)

    df = pd.DataFrame(results)
    exp_num = 0
    out_file = os.path.join(RESULT_DIR, f"concordance_embedded_{exp_num}.csv")
    while os.path.exists(out_file):
        exp_num += 1
        out_file = os.path.join(RESULT_DIR, f"concordance_{exp_num}.csv")
    df.to_csv(out_file, index=False, encoding="utf-8")
    print(f"Embedding concordance saved to {out_file}")

if __name__ == "__main__":
    main()
