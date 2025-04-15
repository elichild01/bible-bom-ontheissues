#!/usr/bin/env python3
import os
import re
import json
import time
from tqdm import tqdm
import pandas as pd
from openai import OpenAI, OpenAIError

# ─── Configuration ─────────────────────────────────────────────────────────────
MODEL_NAME = "gpt-4o-mini"
DATA_DIR   = "./data"
RESULT_DIR = "./results"
# ────────────────────────────────────────────────────────────────────────────────

client = OpenAI()


def extract_json(text: str) -> str:
    """Try to pull the first JSON object out of a string."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    raise ValueError("No JSON object found in LLM response.") # FIXME: should be lowered back down
    if match:
        return match.group(0)

def get_llm_ratings(verse: str, issue: str, viewpoints: list[str]) -> dict:
    """
    Query the LLM for relevance and support scores.
    Returns a dict like {"relevance":0.8, "support_1":0.2, "support_2":0.9, ...}
    """
    # Build the prompt
    vp_list = "\n".join(f"{i+1}. {vp}" for i, vp in enumerate(viewpoints))
    prompt = f"""
You are a theological analyst. Please rate the following verse from the Bible or Book of Mormon for its relevance to the issue and agreement with each of the following viewpoints.
Be skeptical in your relevance ratings. Most verses in scripture are not relevant to most theological issues. It's appropriate to give zeros.
Verse: "{verse}"  
Issue: "{issue}"  
Viewpoints:
{vp_list}

Return **only** a JSON object with:
- "relevance": number between 0 and 1 (how relevant this verse is to the issue)
- For each viewpoint i, "support_i": number between 0 and 1 (how much the verse supports viewpoint i)

Example output:
{{
  "relevance": 0.75,
  "support_1": 0.10,
  "support_2": 0.85
}}
"""
    
    # Make the API call using the updated method
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )

    # Extract the content from the response
    content = response.choices[0].message.content.strip()
    
    try:
        # Attempt to parse the content directly as JSON
        return json.loads(content)
    except json.JSONDecodeError:
        # If direct parsing fails, try extracting the JSON part from the content
        js = extract_json(content)
        return json.loads(js)
    
def parse_verses(fn: str, is_bible: bool) -> list[dict]:
    """
    Read a verse file where each line is:
      Book Chapter:Verse<Space>Text
    Returns a list of dicts with keys: is_bible, book, chapter, verse, text
    """
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

def main():
    # Prepare output directory
    os.makedirs(RESULT_DIR, exist_ok=True)

    # 1) Load issues
    issues_df = pd.read_csv(os.path.join(DATA_DIR, "issues.csv"))
    # ensure columns: issue, viewpoint1, viewpoint2, maybe viewpoint3

    # 2) Load all verses
    all_verses = []
    all_verses += parse_verses(os.path.join(DATA_DIR, "bible_filtered.txt"), True)
    all_verses += parse_verses(os.path.join(DATA_DIR, "bom_filtered.txt"), False)

    # 3) For each verse × issue, call LLM and collect
    results = []

    try:
        for vnum, v in tqdm(enumerate(all_verses), total=len(all_verses), desc="Processing verses"):
            # Skip to desired section of scripture
            # if (v['book'] != 'Luke' or v['chapter'] != '24') and \
            #     (v['book'] != 'John' or v['chapter'] != '1'):
            #     continue
            if vnum < 35594:
                continue
            base = dict(v)  # copy the is_bible/book/chapter/verse/text
            for _, issue_row in issues_df.iterrows():
                # This is where we break if the issue is one we've already covered
                issue = issue_row["issue"]
                # gather viewpoints (skip NaN)
                vps = [issue_row.get(f"viewpoint{i}") for i in (1, 2, 3)]
                viewpoints = [vp for vp in vps if isinstance(vp, str) and vp.strip()]

                # call LLM
                try:
                    ratings = get_llm_ratings(v["text"], issue, viewpoints)
                except Exception as e:
                    print(f"[Warning] LLM call failed for {v['book']} {v['chapter']}:{v['verse']} on '{issue}': {e}")
                    ratings = {"relevance": None, **{f"support_{i+1}": None for i in range(len(viewpoints))}}
                    raise e

                # slug for column names
                issue_name_short = issue_row['issue_short']
                base[f"{issue_name_short}_relevance"] = ratings.get("relevance")
                for i in range(len(viewpoints)):
                    base[f"{issue_name_short}_support_{i+1}"] = ratings.get(f"support_{i+1}")
                # time.sleep(0.125) # avoid rate limiting

            results.append(base)
            
    except OpenAIError as oaie:
        print(f"An OpenAI error occurred: {oaie}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    # 4) Dump to CSV
    df = pd.DataFrame(results)
    exp_num = 0
    out_path = os.path.join(RESULT_DIR, f"concordance_{exp_num}.csv")
    # If the file already exists, create a new file numbered one up from the previous high
    while os.path.exists(out_path):
        exp_num += 1
        out_path = os.path.join(RESULT_DIR, f"concordance_{exp_num}.csv")
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Done! Wrote {len(df)} rows to {out_path}")

if __name__ == "__main__":
    main()