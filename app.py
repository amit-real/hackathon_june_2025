#!/usr/bin/env python3
"""
paper_summarizer.py
-------------------

CLI tool that produces an enriched, structured summary of a research paper
using *only* the OpenAI API.

Features
========
• Accepts either an arXiv URL/ID   (via --arxiv)
  or a local PDF path              (via --pdf)
• Extracts full text with pdfplumber
• Splits text into token-bounded chunks (tiktoken)
• Calls Chat Completions on each chunk, requesting a JSON object with:
    - bullets        (concise key points)
    - use_cases      (potential applications)
    - strengths      (methodological advantages)
    - limitations    (caveats, weak spots)
    - open_questions (future work)
• Merges & deduplicates those objects
• Runs a final "condenser" call to trim / rank the lists
• Pretty-prints for human reading (or pipe the JSON elsewhere)

Usage
-----
    python paper_summarizer.py --arxiv 2405.12345
    python paper_summarizer.py --pdf   path/to/file.pdf
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import textwrap
from pathlib import Path
from typing import Dict, List

import pdfplumber
import requests
import tiktoken
from tqdm import tqdm

from openai import OpenAI

# --------------------------------------------------------------------------- #
# Configuration constants – tweak to taste                                    #
# --------------------------------------------------------------------------- #
MODEL_NAME = "gemini-2.5-flash"
# MODEL_NAME = "gpt-4o-mini"           # cheap & powerful; use gpt-3.5-turbo-0125 if needed
TOKENS_PER_CHUNK = 3000              # leave head-room under model context window
FINAL_BULLETS = 8                    # how many bullets / items to keep
TIMEOUT = 30                         # seconds for HTTP download

# Target JSON schema all prompts must return
JSON_KEYS = [
    "bullets",
    "use_cases",
    "strengths",
    "limitations",
    "open_questions",
]

# --------------------------------------------------------------------------- #
# OpenAI client (relies on OPENAI_API_KEY env var)                            #
# --------------------------------------------------------------------------- #
# client = OpenAI(api_key="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")  # sets key & org from env
client = OpenAI(api_key="AIzaSyDMkzWN6h6vbNpZ-VgLxoRM0sgDGoP6Ong",
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
# --------------------------------------------------------------------------- #
# Utility functions                                                           #
# --------------------------------------------------------------------------- #
def download_arxiv_pdf(arxiv_id: str, out_dir: Path) -> Path:
    """Download the PDF from arXiv and return the local path."""
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    out_path = out_dir / f"{arxiv_id}.pdf"

    print(f"⇩ Downloading {url}")
    with requests.get(url, stream=True, timeout=TIMEOUT) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
    return out_path


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Return concatenated text from all pages."""
    text_parts: List[str] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            text_parts.append(text)
    return "\n".join(text_parts)


def chunk_text(text: str, max_tokens: int) -> List[str]:
    """Split text into chunks with <= max_tokens each (uses tiktoken)."""
    # Use a standard encoding instead of one tied to a specific OpenAI model name.
    # This is the correct approach when using non-OpenAI models like Gemini.
    enc = tiktoken.get_encoding("cl100k_base")
    words = text.split()

    chunks: List[str] = []
    current_words: List[str] = []
    current_tokens = 0

    for w in words:
        # A small correction here: encoding the space separately can be more accurate
        tok_len = len(enc.encode(" " + w))
        if current_tokens + tok_len > max_tokens:
            chunks.append(" ".join(current_words))
            current_words, current_tokens = [], 0
        current_words.append(w)
        current_tokens += tok_len

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


# --------------------------------------------------------------------------- #
# OpenAI calls                                                                #
# --------------------------------------------------------------------------- #
# def summarize_chunk(chunk: str) -> Dict[str, List[str]]:
#     """Call GPT once on a chunk, asking for structured JSON."""
#     system = (
#         "You are an expert research analyst. "
#         "Read the following fragment of an academic paper and return a **JSON object only** "
#         f"with the keys {JSON_KEYS}. "
#         "Each key must map to a list of concise (<=25-word) strings. "
#         "Do NOT wrap the JSON in markdown or add any extra keys."
#     )

#     resp = client.chat.completions.create(
#         model=MODEL_NAME,
#         messages=[
#             {"role": "system", "content": system},
#             {"role": "user", "content": chunk},  # safety guard
#         ],
#         temperature=0.2,
#     )
#     print(resp)
#     exit()
#     return json.loads(resp.choices[0].message.content)


# def condense_json(raw_json: Dict[str, List[str]]) -> Dict[str, List[str]]:
#     """Ask GPT to rank/trim each list down to FINAL_BULLETS items."""
#     prompt = textwrap.dedent(
#         f"""
#         Merge / deduplicate / polish the lists in the JSON below.
#         For each key keep the **{FINAL_BULLETS} most important** items (or fewer
#         if content is lacking). Rephrase for clarity, keep technical precision.
#         Return the same JSON structure, no markdown.
#         """
#     )

#     resp = client.chat.completions.create(
#         model=MODEL_NAME,
#         messages=[
#             {"role": "system", "content": "JSON condenser"},
#             {"role": "user", "content": prompt + "\n\n" + json.dumps(raw_json)},
#         ],
#         temperature=0.2,
#     )

#     return json.loads(resp.choices[0].message.content)


def summarize_chunk(chunk: str) -> Dict[str, List[str]]:
    """Call the API on a chunk, asking for structured JSON."""
    system = (
        "You are an expert research analyst. "
        "Your task is to extract specific information from a fragment of an academic paper. "
        "Read the text and return a single, valid JSON object and nothing else. "
        "Do NOT add explanations, apologies, or markdown formatting like ```json. "
        f"The JSON object must have these exact keys: {JSON_KEYS}. "
        "Each key's value must be a list of short, concise strings (max 25 words)."
    )
    
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": chunk},
            ],
            temperature=0.2,
            # extra_body={"safety_settings": safety_settings},
        )

        raw_content = resp.choices[0].message.content or ""

        # Aggressively find a JSON object within the response string
        match = re.search(r"\{.*\}", raw_content, re.DOTALL)
        if match:
            json_str = match.group(0)
            return json.loads(json_str)
        else:
            print(f"⚠️  WARNING: API did not return a JSON object for a chunk. Response was:\n---{raw_content}---\n")
            return {k: [] for k in JSON_KEYS}

    except json.JSONDecodeError:
        print(f"❌ ERROR: Failed to decode extracted JSON. The string was:\n---{match.group(0) if 'match' in locals() and match else raw_content}---\n")
        return {k: [] for k in JSON_KEYS}
    except Exception as e:
        print(f"❌ An unexpected error occurred during API call: {e}")
        return {k: [] for k in JSON_KEYS}


def condense_json(raw_json: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Ask the API to rank/trim each list down to FINAL_BULLETS items."""
    prompt = textwrap.dedent(
        f"""
        Merge / deduplicate / polish the lists in the JSON below.
        For each key keep the **{FINAL_BULLETS} most important** items (or fewer
        if content is lacking). Rephrase for clarity, keep technical precision.
        Return a single, valid JSON object and nothing else. Do not add explanations or markdown.
        """
    )
    
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
    
    input_json_str = json.dumps(raw_json, indent=2)

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful JSON processing assistant."},
                {"role": "user", "content": prompt + "\n\n" + input_json_str},
            ],
            temperature=0.2,
        )
        
        raw_content = resp.choices[0].message.content or ""
        
        match = re.search(r"\{.*\}", raw_content, re.DOTALL)
        if match:
            json_str = match.group(0)
            return json.loads(json_str)
        else:
            print(f"⚠️  WARNING: Condenser API did not return a JSON object. Returning original merged JSON.")
            return raw_json # Fallback to un-condensed JSON

    except Exception as e:
        print(f"❌ An error occurred during condensation step: {e}. Returning original merged JSON.")
        return raw_json # Fallback to un-condensed JSON

# --------------------------------------------------------------------------- #
# Merge helpers                                                               #
# --------------------------------------------------------------------------- #
def merge_dicts(dicts: List[Dict[str, List[str]]]) -> Dict[str, List[str]]:
    """Combine lists under each JSON key, deduplicating (case-insensitive)."""
    merged = {k: [] for k in JSON_KEYS}
    seen_per_key = {k: set() for k in JSON_KEYS}

    for d in dicts:
        for k in JSON_KEYS:
            for item in d.get(k, []):
                norm = item.lower().strip()
                if norm not in seen_per_key[k]:
                    merged[k].append(item.strip())
                    seen_per_key[k].add(norm)
    return merged


# --------------------------------------------------------------------------- #
# Pretty printing                                                             #
# --------------------------------------------------------------------------- #
def pretty_print(result: Dict[str, List[str]]) -> None:
    print("\n🔹  \033[1mKey Takeaways\033[0m")
    for bullet in result["bullets"]:
        print(f"• {bullet}")

    def section(title: str, key: str) -> None:
        items = result.get(key, [])
        if items:
            print(f"\n🔸  \033[1m{title}\033[0m")
            for i, item in enumerate(items, 1):
                print(f"{i}. {item}")

    section("Potential Use-cases", "use_cases")
    section("Strengths", "strengths")
    section("Limitations", "limitations")
    section("Open Questions / Next Steps", "open_questions")
    print()  # final newline


# --------------------------------------------------------------------------- #
# Main runner                                                                 #
# --------------------------------------------------------------------------- #
def run(arxiv: str | None, pdf: str | None) -> None:
    # 1. Resolve / download PDF
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        if arxiv:
            arxiv_id = re.sub(r"(https?://arxiv\.org/(abs|pdf)/)?", "", arxiv)
            pdf_path = download_arxiv_pdf(arxiv_id, tmp_dir)
        else:
            pdf_path = Path(pdf).expanduser()
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF not found: {pdf_path}")

        print(f"📄 Extracting text from {pdf_path.name}")
        full_text = extract_text_from_pdf(pdf_path)

    # 2. Chunk & summarise
    chunks = chunk_text(full_text, TOKENS_PER_CHUNK)
    print(f"✂️  Chunked into {len(chunks)} piece(s); summarising…")
    partial_dicts = [summarize_chunk(c) for c in tqdm(chunks, unit="chunk")]

    # 3. Merge & condense
    merged = merge_dicts(partial_dicts)
    final = condense_json(merged)

    # 4. Display
    pretty_print(final)


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate an enriched summary of a research paper using OpenAI."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--arxiv", help="arXiv ID or full URL (e.g. 2405.12345)")
    group.add_argument("--pdf", help="Path to a local PDF file")

    args = parser.parse_args()
    run(args.arxiv, args.pdf)
