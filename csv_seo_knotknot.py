#!/usr/bin/env python3
"""
CSV SEO — Knot Knot (API-connected)
-----------------------------------
Builds three CSVs from a Shopify export:
  1) knotknot_review.csv
     Columns: Handle, Proposed Title, Product Type, SEO Title, SEO Description, Primary Image URL
  2) knotknot_proposed_full.csv
     Adds: Body (HTML), and consolidated Alt Texts (per-image) for review
  3) knotknot_safe_update.csv
     Minimal Shopify import that only updates:
       - Title
       - Body (HTML)
       - SEO Title
       - SEO Description
       - Image Alt Text  (via additional image rows with Handle + Image Src + Image Alt Text)

KEY GUARANTEES (aligned with your master prompt):
- No invention of specs: the model is instructed to avoid adding dimensions/materials/claims not present in input.
- Rename off/awkward names to community-appropriate terms (e.g., “St. Andrew’s Cross”).
- “Pet Play Coffee Table” must stay exactly as-is when applicable.
- SEO Title ≤ 60 chars; SEO Description ≤ 160 chars.
- Image Alt Text concise (≤ ~125 chars), descriptive, brand-aware (Knot Knot).
- Safety note included when relevant (impact/anal toys/etc.).
- Categorization proposed by the model; we also apply a conservative keyword fallback.
- Starts with a subset using --limit or --handles-file and can scale to entire catalog.

USAGE:
  python csv_seo_knotknot.py \
      --input products_export.csv \
      --output-dir ./out \
      --limit 10 \
      --model gpt-4o-mini \
      --temperature 0.2

REQUIREMENTS:
  pip install pandas requests
  And set environment variable (example):
    export OPENAI_API_KEY=your_api_key_here

NOTES:
- This script calls OpenAI’s Chat Completions or Responses API (compatible via /v1/chat/completions).
- Internet is required for API calls. It does NOT fetch images; it passes image URLs to the model as context.
- If the model returns invalid JSON, we fall back to conservative rule-based transforms.

"""

import os
import sys
import argparse
import json
import math
import time
from typing import List, Dict, Any, Tuple, Optional
import re
from urllib.parse import urlparse
import pandas as pd
import requests

# -----------------------------
# Configuration & Constants
# -----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com")  # override if using a proxy
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.2
BRAND_NAME = "Knot Knot"
MAX_ALT_TEXT_LEN = 125
MAX_SEO_TITLE = 60
MAX_SEO_DESC = 160
MAX_IMAGES_TO_SEND = 3  # primary + a couple more to ground the model visually

# Conservative keyword → product type mapping (fallback if model omits)
KEYWORD_TO_TYPE = [
    ("st. andrew", "Bondage Cross"),
    ("andrew’s cross", "Bondage Cross"),
    ("st andrews cross", "Bondage Cross"),
    ("x-frame", "Bondage Cross"),
    ("spread bar", "Restraints"),
    ("spreadbar", "Restraints"),
    ("hogtie", "Restraints"),
    ("cuff", "Restraints"),
    ("shackle", "Restraints"),
    ("gag", "Gags"),
    ("bit gag", "Gags"),
    ("ball gag", "Gags"),
    ("paddle", "Impact Toys"),
    ("flogger", "Impact Toys"),
    ("whip", "Impact Toys"),
    ("crop", "Impact Toys"),
    ("collar", "Collars & Leashes"),
    ("leash", "Collars & Leashes"),
    ("harness", "Harnesses"),
    ("hood", "Hoods & Masks"),
    ("blindfold", "Sensory Play"),
    ("mask", "Hoods & Masks"),
    ("nipple", "Nipple Toys"),
    ("clamp", "Nipple Toys"),
    ("plug", "Anal Toys"),
    ("anal", "Anal Toys"),
    ("dildo", "Dildos"),
    ("chastity", "Chastity"),
    ("cage", "Chastity"),
    ("furniture", "Bondage Furniture"),
    ("stocks", "Bondage Furniture"),
    ("stockade", "Bondage Furniture"),
    ("pet play coffee table", "Bondage Furniture"),
    ("pet play", "Pet Play"),
]

def flexible_bdsmlanguage(title: str) -> str:
    """
    Light-touch normalization that keeps space for creativity.
    Applies gentle, BDSM-centric synonyms without forcing a rigid term.
    """
    if not title:
        return title
    t = title.strip()
    # Preserve invariant
    if "pet play coffee table" in t.lower():
        return "Pet Play Coffee Table"

    low = t.lower()
    # Minimal synonym nudges; avoid injecting the literal word 'BDSM'
    synonyms = [
        ("torture rack", "Cross"),
        ("x-torture rack", "Cross"),
        ("x torture rack", "Cross"),
        ("restraint", "Restraint"),
        ("chair", "Bondage Chair"),
        ("gag", "Gag"),
        ("play toy", "Toy"),
        ("furniture", "Bondage Furniture"),
    ]
    for pat, repl in synonyms:
        if pat in low:
            low = low.replace(pat, repl.lower())
    return low.title()

# -----------------------------
# Helpers
# -----------------------------

def clamp(text: str, max_len: int) -> str:
    if not text:
        return text
    text = text.strip()
    if len(text) <= max_len:
        return text
    return text[:max_len-1].rstrip() + "…"

def sanitize_alt_text(text: str) -> str:
    text = text or ""
    text = " ".join(text.split())
    return clamp(text, MAX_ALT_TEXT_LEN)

# No strict rename normalization; allow model creativity.

def guess_product_type(text: str) -> Optional[str]:
    s = (text or "").lower()
    for kw, ptype in KEYWORD_TO_TYPE:
        if kw in s:
            return ptype
    return None

def primary_image_for_product(group_df: pd.DataFrame) -> Optional[str]:
    # Choose the image with smallest Image Position; fallback to first non-null Image Src
    images = group_df[["Image Src", "Image Position"]].dropna(subset=["Image Src"])
    if images.empty:
        return None
    images = images.copy()
    # Numeric sorting: positions might be float or str
    def pos_val(v):
        try:
            return float(v)
        except Exception:
            return math.inf
    images["__pos"] = images["Image Position"].apply(pos_val)
    images = images.sort_values(by="__pos")
    return images.iloc[0]["Image Src"]

def _infer_title_from_image_and_type(product_payload: Dict[str, Any], product_type: Optional[str]) -> str:
    """
    Heuristic fallback name when the model proposes an unhelpful title like 'Trojan'.
    Uses image URL tokens and product_type to pick a safe, descriptive name
    without inventing specs.
    """
    # Tokens from image URLs
    words: List[str] = []
    for u in (product_payload.get("images") or []):
        try:
            path = urlparse(u).path.lower()
            for part in re.split(r"[^a-z0-9]+", path):
                if part:
                    words.append(part)
        except Exception:
            continue

    def has(*ks: str) -> bool:
        return any(k in words for k in ks)

    # Simple keyword mapping
    if has("gag", "bit", "muzzle"):
        return "BDSM Gag"
    if has("hood"):
        return "Bondage Hood"
    if has("mask", "blindfold"):
        return "Bondage Mask"
    if has("cross", "xframe", "x-frame", "st-andrew"):
        return "St. Andrew’s Cross"
    if has("paddle"):
        return "Paddle"
    if has("flogger"):
        return "Flogger"
    if has("whip"):
        return "Whip"
    if has("plug"):
        return "Anal Plug"
    if has("collar") and has("leash"):
        return "Collar and Leash"
    if has("collar"):
        return "Collar"
    if has("harness"):
        return "Body Harness"
    if has("stockade", "stocks"):
        return "Stockade"
    if has("bench"):
        return "Bondage Bench"
    if has("chair"):
        return "Bondage Chair"
    if has("table"):
        return "Bondage Table"
    if has("spread") and has("bar"):
        return "Spread Bar"
    if has("cage") and has("chastity"):
        return "Chastity Cage"

    # Fall back to product_type or a generic safe name
    if product_type and isinstance(product_type, str) and product_type.strip():
        return product_type.strip()
    return "BDSM Accessory"


def _prompt_approve_rename(handle: str, original: str, proposed: str, reason: Optional[str] = None) -> Tuple[str, bool]:
    """
    Interactive prompt asking user to approve a title change.
    Returns (final_title, accept_all_for_rest).
    """
    print("\n— Rename proposal —")
    print(f"Handle: {handle}")
    print(f"Original: {original}")
    print(f"Proposed: {proposed}")
    if reason:
        print(f"Reason:   {reason}")
    print("Options: [y] accept, [n] keep original, [e] edit, [a] accept all")
    while True:
        try:
            choice = input("Approve rename? [y/n/e/a]: ").strip().lower()
        except EOFError:
            # Non-interactive; default to keeping proposed
            return proposed, False
        if choice in ("y", "yes", ""):
            return proposed, False
        if choice in ("n", "no"):
            return original, False
        if choice == "a":
            return proposed, True
        if choice == "e":
            custom = input("Enter custom title: ").strip()
            if custom:
                return custom, False
        print("Please choose y, n, e, or a.")


def _strip_bdsm_tokens(title: str) -> str:
    """
    Remove standalone 'BDSM' tokens from titles to avoid overuse.
    If removal empties the title, keep the original.
    """
    if not isinstance(title, str):
        return title
    original = title
    # Remove 'BDSM' token optionally followed by punctuation like ':', '-', '—'
    cleaned = re.sub(r"(?i)\bBDSM\b\s*[:\-–—]?\s*", "", original)
    # Collapse whitespace and trim punctuation leftovers
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip().strip("-–—: ")
    return cleaned if cleaned else original

def _select_image_urls(product_payload: Dict[str, Any], max_images: int = MAX_IMAGES_TO_SEND) -> List[str]:
    """Return up to N image URLs, prioritizing the primary image first."""
    primary = product_payload.get("primary_image")
    imgs = [u for u in (product_payload.get("images") or []) if isinstance(u, str) and u.strip()]
    ordered: List[str] = []
    if primary and isinstance(primary, str) and primary.strip():
        ordered.append(primary)
    for u in imgs:
        if u not in ordered:
            ordered.append(u)
    return ordered[:max_images]


def build_messages(product_payload: Dict[str, Any], visual_hint: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Construct the system/user messages for the OpenAI API.
    We ask the model to produce STRICT JSON with all required fields.
    """
    sys_msg = {
        "role": "system",
        "content": (
            "You are an expert Shopify SEO copywriter for a BDSM brand named 'Knot Knot'. "
            "Base naming and descriptions primarily on the initial product title and visual cues from the provided images. "
            "Propose creative, evocative, and SEO-friendly product titles that use BDSM community terminology and synonyms (e.g., restraints, gags, bondage furniture, impact toys). "
            "Do not prepend or overuse the literal word 'BDSM' in titles; prefer natural community terms like 'bondage', 'gag', 'cross', etc., when relevant. "
            "Avoid repetitive phrasing across items and keep names natural to the scene. "
            "Use terminology BDSM buyers search for. "
            "Do NOT invent specs (no new dimensions, materials, certifications). "
            "If the source content lacks details, keep copy high-level and honest. "
            "Include a safety note when relevant (impact toys, anal toys, etc.). "
            "Respect constraints: SEO Title ≤ 60 chars, SEO Description ≤ 160 chars, "
            "alt texts concise (≤125 chars) and brand-aware. "
            "If the product is or resembles 'Pet Play Coffee Table', KEEP the title exactly 'Pet Play Coffee Table'. "
            "Return only JSON. No markdown. No commentary."
        )
    }

    user_instr = {
        "brand": BRAND_NAME,
        "constraints": {
            "seo_title_max": MAX_SEO_TITLE,
            "seo_desc_max": MAX_SEO_DESC,
            "alt_text_max": MAX_ALT_TEXT_LEN
        },
        "input": product_payload,
        "visual_hint": visual_hint or {},
        "required_output_schema": {
            "proposed_title": "string",
            "product_type": "string",
            "seo_title": "string (≤60 chars)",
            "seo_description": "string (≤160 chars)",
            "body_html": "string (HTML allowed: p, ul, li, strong)",
            "alt_texts": [
                {"image_src": "string (from provided images)", "alt_text": "string (≤125 chars)"}
            ]
        },
        "style_rules": [
            "Short intro sentence; then 3–5 bullet features.",
            "Favor community-preferred BDSM terms and synonyms; avoid generic words like 'adult toy'.",
            "Allow creative, varied titles that remain accurate and brand-appropriate.",
            "Tone: assured, discreet, and buyer-friendly.",
            "No medical/therapeutic claims. No illegal content. No explicit sexual health promises.",
            "Avoid explicit language; keep it tasteful and compliant."
        ],
        "category_rules": [
            "Prefer a best-fit BDSM category; if unsure, choose a conservative fit (e.g., 'Restraints', 'Impact Toys', 'Bondage Furniture')."
        ],
        "naming_goals": [
            "Encourage varied naming; avoid repeating the same root words across items.",
            "Keep titles specific enough to communicate use without inventing specs.",
            "Align with search behavior and community terminology."
        ]
    }

    # Compose a multimodal user message: JSON instructions + image URLs
    content_parts: List[Dict[str, Any]] = [
        {"type": "text", "text": json.dumps(user_instr, ensure_ascii=False)}
    ]
    for img_url in _select_image_urls(product_payload):
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": img_url}
        })

    user_msg = {"role": "user", "content": content_parts}
    
    return [sys_msg, user_msg]


def build_messages_visual(product_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Compose messages to ask the model to analyze images first and return a concise
    visual hint object we can cross-reference with the original title.
    """
    sys_msg = {
        "role": "system",
        "content": (
            "You are assisting with product catalog curation for a BDSM brand. "
            "Analyze the provided images to identify WHAT the product visually appears to be. "
            "Return only JSON with: visual_guess_title (short), visual_category, visual_terms (list), confidence (0-1), notes (short). "
            "Avoid inventing specs, materials, or dimensions. No brand claims."
        )
    }

    instr = {
        "original_title": product_payload.get("title"),
        "goal": "Identify the product visually; avoid generic marketplace mislabels.",
        "required_output_schema": {
            "visual_guess_title": "string",
            "visual_category": "string",
            "visual_terms": ["string"],
            "confidence": "float 0..1",
            "notes": "string"
        }
    }

    parts: List[Dict[str, Any]] = [{"type": "text", "text": json.dumps(instr, ensure_ascii=False)}]
    for img_url in _select_image_urls(product_payload):
        parts.append({"type": "image_url", "image_url": {"url": img_url}})

    user_msg = {"role": "user", "content": parts}
    return [sys_msg, user_msg]

def call_openai_chat(messages: List[Dict[str, Any]], model: str, temperature: float = 0.2, retries: int = 3, timeout: int = 60) -> Dict[str, Any]:
    """
    Calls OpenAI Chat Completions API. Returns parsed JSON dict (model output).
    Retries on transient errors and JSON parsing issues.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY env var is not set.")

    url = f"{OPENAI_API_BASE.rstrip('/')}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "temperature": temperature,
        "messages": messages,
        "response_format": {"type": "json_object"},
    }

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 429:
                # Rate limit: exponential backoff
                wait_s = 2 ** attempt
                time.sleep(wait_s)
                continue
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return json.loads(content)
        except Exception as e:
            last_err = e
            # brief backoff
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"OpenAI API call failed after {retries} attempts: {last_err}")


def analyze_visuals(product_payload: Dict[str, Any], model: str, temperature: float = 0.0) -> Optional[Dict[str, Any]]:
    """
    Run a lightweight visual-first analysis to get a structured hint from images.
    Returns a dict or None if the call fails.
    """
    try:
        messages = build_messages_visual(product_payload)
        return call_openai_chat(messages, model=model, temperature=temperature)
    except Exception:
        return None

def conservative_rule_based(product_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fallback if the model response is unavailable/invalid. Uses only input fields.
    """
    handle = product_payload.get("handle") or ""
    title_raw = product_payload.get("title") or handle.replace("-", " ").title()
    title = flexible_bdsmlanguage(title_raw)

    # SEO Title: clamp to 60; prefer title
    seo_title = clamp(title, MAX_SEO_TITLE)

    # SEO Description: generic, safe, no invented specs
    base_desc = f"{title} by {BRAND_NAME}. Shop discreet, quality gear for confident play."
    seo_desc = clamp(base_desc, MAX_SEO_DESC)

    # Body: simple structure with optional safety note
    body_intro = f"<p>{title} — designed for reliable performance and confident control.</p>"
    features = [
        "<li>Thoughtful design for secure fit</li>",
        "<li>Durable construction</li>",
        "<li>Easy to maintain</li>",
    ]
    # Safety hint if risky keywords
    source_text = " ".join([
        title_raw or "", product_payload.get("body_html") or "", product_payload.get("type") or "", handle
    ]).lower()
    safety = ""
    if any(x in source_text for x in ["paddle", "flogger", "whip", "crop", "impact"]):
        safety = "<p><strong>Safety:</strong> Start light, build gradually, and monitor skin response.</p>"
    if any(x in source_text for x in ["plug", "anal"]):
        safety = "<p><strong>Safety:</strong> Use plenty of water-based lubricant and choose appropriate sizes.</p>"

    body_html = f"{body_intro}<ul>{''.join(features)}</ul>{safety}"

    # Product type guess
    combined = " ".join([
        title_raw or "", product_payload.get("type") or "", product_payload.get("category") or "", handle
    ])
    ptype = guess_product_type(combined) or "Accessories"

    # Alt texts
    alts = []
    for img in product_payload.get("images", []):
        # concise default
        base_alt = f"{title} — {BRAND_NAME}"
        alts.append({"image_src": img, "alt_text": sanitize_alt_text(base_alt)})

    return {
        "proposed_title": title,
        "product_type": ptype,
        "seo_title": seo_title,
        "seo_description": seo_desc,
        "body_html": body_html,
        "alt_texts": alts
    }

def build_product_payload(handle: str, group_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Prepare inputs for the model: title, body, tags, type/category, images.
    """
    # Take first non-null occurrence for fields commonly at product level
    def first_non_null(col: str) -> Optional[str]:
        if col not in group_df.columns:
            return None
        vals = group_df[col].dropna()
        if vals.empty:
            return None
        return str(vals.iloc[0]).strip()

    title = first_non_null("Title")
    body_html = first_non_null("Body (HTML)")
    vendor = first_non_null("Vendor")
    product_category = first_non_null("Product Category")
    ptype = first_non_null("Type") or first_non_null("Product Type")
    tags = first_non_null("Tags")

    # images (all)
    images = group_df["Image Src"].dropna().astype(str).unique().tolist()

    # primary
    primary_img = primary_image_for_product(group_df)

    payload = {
        "handle": handle,
        "title": title,
        "body_html": body_html,
        "vendor": vendor,
        "category": product_category,
        "type": ptype,
        "tags": tags,
        "primary_image": primary_img,
        "images": images
    }
    return payload

def process_catalog(
    df: pd.DataFrame,
    limit: Optional[int],
    handles_subset: Optional[List[str]],
    model: str,
    temperature: float,
    confirm_renames: bool,
    visual_first: bool,
    verbose: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Main processing loop. Returns (review_df, proposed_full_df, safe_update_df).
    """
    # Group by Handle (unique products)
    if "Handle" not in df.columns:
        raise ValueError("Input CSV must have a 'Handle' column.")
    grouped = list(df.groupby("Handle", sort=False))
    if handles_subset:
        handles_set = set(h.strip() for h in handles_subset if h.strip())
        grouped = [(h, g) for (h, g) in grouped if h in handles_set]
    if limit:
        grouped = grouped[:int(limit)]

    review_rows = []
    full_rows = []

    # For safe update, we will create:
    # - One product row per handle with updated Title, Body (HTML), SEO Title, SEO Description.
    # - Additional image rows for alt text updates with only Handle + Image Src + Image Alt Text.
    safe_update_rows_product = []
    safe_update_rows_images = []

    accept_all_titles = False

    total = len(grouped)
    for idx, (handle, gdf) in enumerate(grouped, start=1):
        product_payload = build_product_payload(handle, gdf)
        # Preserve original title for rename comparison
        original_title = product_payload.get("title") or handle.replace("-", " ").title()
        if verbose:
            print(f"[{idx}/{total}] Handle={handle} | Original='{original_title}'")

        # Light-touch normalization to hint BDSM context while keeping space for creativity
        product_payload["title"] = flexible_bdsmlanguage(product_payload.get("title") or "")

        # Optional: visual-first analysis to reduce generic or wrong names
        visual_hint: Optional[Dict[str, Any]] = None
        if visual_first:
            visual_hint = analyze_visuals(product_payload, model=model, temperature=0.0)
            if verbose:
                if visual_hint:
                    print("  visual_hint:", json.dumps(visual_hint, ensure_ascii=False))
                else:
                    print("  visual_hint: <none>")

        # Build messages & call model
        messages = build_messages(product_payload, visual_hint=visual_hint)
        if verbose:
            imgs = _select_image_urls(product_payload)
            if imgs:
                print("  images:")
                for u in imgs:
                    print("   -", u)
        try:
            model_json = call_openai_chat(messages, model=model, temperature=temperature)
        except Exception as e:
            # Fallback to conservative rule-based if API fails
            model_json = conservative_rule_based(product_payload)
            if verbose:
                print("  api_error -> using conservative fallback")

        # Post-validate and clamp
        proposed_title = model_json.get("proposed_title") or product_payload.get("title") or handle.replace("-", " ").title()
        if verbose:
            print(f"  proposed_raw='{proposed_title}'")
        if "pet play coffee table" in (proposed_title or "").lower():
            proposed_title = "Pet Play Coffee Table"  # enforce invariant

        # Apply naming rules:
        # 1) Replace standalone 'K9' (or 'K-9') with 'Pet Play'
        before_k9 = proposed_title
        proposed_title = re.sub(r"\bK[- ]?9\b", "Pet Play", proposed_title, flags=re.IGNORECASE)
        k9_changed = (before_k9 != proposed_title)
        if verbose and k9_changed:
            print(f"  rename_k9 -> '{proposed_title}'")
        # 2) Handle any 'Trojan' variants anywhere in the title
        trojan_changed = False
        trojan_pattern = r"\bTrojan(?:s(?:-\d+)?)?\b"
        if re.search(trojan_pattern, proposed_title, flags=re.IGNORECASE):
            fallback_name = _infer_title_from_image_and_type(product_payload, None)
            # Replace the token(s) to avoid condom-brand confusion
            proposed_title = re.sub(trojan_pattern, fallback_name, proposed_title, flags=re.IGNORECASE)
            trojan_changed = True
            if verbose:
                print(f"  rename_trojan -> '{proposed_title}' (fallback='{fallback_name}')")

        # Optional approval for title rename
        # Only prompt for confirmation when a Trojan tag was detected
        if confirm_renames and not accept_all_titles and trojan_changed:
            reason = "Avoid 'Trojan' confusion; descriptive fallback from images"
            final_title, accept_all_flag = _prompt_approve_rename(handle, original_title, proposed_title, reason)
            proposed_title = final_title
            if accept_all_flag:
                accept_all_titles = True

        # Strip 'BDSM' tokens to avoid literal use in titles
        proposed_title = _strip_bdsm_tokens(proposed_title)
        if verbose:
            print(f"  title_final='{proposed_title}'")

        product_type = model_json.get("product_type") or guess_product_type(
            " ".join([proposed_title, product_payload.get("type") or "", product_payload.get("category") or ""])
        ) or "Accessories"
        if verbose:
            print(f"  product_type='{product_type}'")

        seo_title = clamp(model_json.get("seo_title") or proposed_title, MAX_SEO_TITLE)
        seo_description = clamp(model_json.get("seo_description") or f"{proposed_title} by {BRAND_NAME}.", MAX_SEO_DESC)
        body_html = model_json.get("body_html") or ""

        # Ensure body structure roughly matches: intro + 3–5 bullets, optional safety; basic tags only
        if not body_html:
            body_html = conservative_rule_based(product_payload)["body_html"]

        # Alt texts
        alt_texts = model_json.get("alt_texts") or []
        # Repair alt_texts if missing images
        available_images = product_payload.get("images", [])
        if not alt_texts and available_images:
            alt_texts = [{"image_src": img, "alt_text": sanitize_alt_text(f"{proposed_title} — {BRAND_NAME}")} for img in available_images]
        else:
            # sanitize each entry
            repaired = []
            for entry in alt_texts:
                img = entry.get("image_src")
                if img in (None, "", "null"):
                    # skip if no image_src
                    continue
                alt = sanitize_alt_text(entry.get("alt_text") or f"{proposed_title} — {BRAND_NAME}")
                repaired.append({"image_src": img, "alt_text": alt})
            # Ensure we cover all known images
            existing = {e["image_src"] for e in repaired}
            for img in available_images:
                if img not in existing:
                    repaired.append({"image_src": img, "alt_text": sanitize_alt_text(f"{proposed_title} — {BRAND_NAME}")})
            alt_texts = repaired

        primary_img = product_payload.get("primary_image")

        # --- Build review row
        review_rows.append({
            "Handle": handle,
            "Proposed Title": proposed_title,
            "Product Type": product_type,
            "SEO Title": seo_title,
            "SEO Description": seo_description,
            "Primary Image URL": primary_img or ""
        })

        # --- Build proposed full row
        # Consolidate alt texts for human review
        alt_join = "; ".join([f"{a['image_src']} || {a['alt_text']}" for a in alt_texts]) if alt_texts else ""
        full_rows.append({
            "Handle": handle,
            "Proposed Title": proposed_title,
            "Product Type": product_type,
            "SEO Title": seo_title,
            "SEO Description": seo_description,
            "Body (HTML)": body_html,
            "Image Alt Texts (all)": alt_join
        })

        # --- Safe update product row (minimal)
        safe_update_rows_product.append({
            "Handle": handle,
            "Title": proposed_title,
            "Body (HTML)": body_html,
            "SEO Title": seo_title,
            "SEO Description": seo_description
        })

        # --- Safe update image rows (alt text updates)
        for a in alt_texts:
            safe_update_rows_images.append({
                "Handle": handle,
                "Image Src": a["image_src"],
                "Image Alt Text": a["alt_text"]
            })

    review_df = pd.DataFrame(review_rows, columns=[
        "Handle", "Proposed Title", "Product Type", "SEO Title", "SEO Description", "Primary Image URL"
    ])

    full_df = pd.DataFrame(full_rows, columns=[
        "Handle", "Proposed Title", "Product Type", "SEO Title", "SEO Description", "Body (HTML)", "Image Alt Texts (all)"
    ])

    # For safe_update CSV, Shopify accepts rows with only the fields being updated.
    # We concatenate the single product rows + separate image rows (Shopify treats them as part of the same product via Handle).
    safe_product_df = pd.DataFrame(safe_update_rows_product, columns=[
        "Handle", "Title", "Body (HTML)", "SEO Title", "SEO Description"
    ])
    safe_images_df = pd.DataFrame(safe_update_rows_images, columns=[
        "Handle", "Image Src", "Image Alt Text"
    ])
    safe_update_df = pd.concat([safe_product_df, safe_images_df], ignore_index=True)

    return review_df, full_df, safe_update_df

def parse_handles_file(path: str) -> List[str]:
    if not path:
        return []
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
    return [ln for ln in lines if ln]

def main():
    parser = argparse.ArgumentParser(description="Knot Knot CSV SEO — API-connected generator")
    parser.add_argument("--input", required=True, help="Path to Shopify export CSV (products_export.csv)")
    parser.add_argument("--output-dir", default="./out", help="Directory to write the 3 CSVs")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of products (by Handle) to process")
    parser.add_argument("--handles-file", default=None, help="Optional: text file with one Handle per line to process")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model (default: {DEFAULT_MODEL})")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Model temperature (default: 0.2)")
    parser.add_argument("--confirm-renames", action="store_true", help="Prompt to approve or edit proposed title changes")
    parser.add_argument("--visual-first", action="store_true", help="Analyze images first and feed results into naming to avoid generic/wrong titles")
    parser.add_argument("--verbose", action="store_true", help="Print detailed progress and decisions per product")
    args = parser.parse_args()

    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    # Load CSV
    try:
        # Handle BOM in some Shopify exports
        df = pd.read_csv(args.input, encoding="utf-8-sig")
    except Exception as e:
        print(f"ERROR: Failed to read CSV '{args.input}': {e}", file=sys.stderr)
        sys.exit(1)

    # Normalize expected columns
    for col in ["Handle", "Title", "Body (HTML)", "Vendor", "Product Category", "Type", "Tags", "Image Src", "Image Position"]:
        if col not in df.columns:
            # Create missing if needed to avoid KeyErrors (not strictly required)
            df[col] = None

    handles_subset = parse_handles_file(args.handles_file) if args.handles_file else None

    # Process
    review_df, full_df, safe_update_df = process_catalog(
        df=df,
        limit=args.limit,
        handles_subset=handles_subset,
        model=args.model,
        temperature=args.temperature,
        confirm_renames=args.confirm_renames,
        visual_first=args.visual_first,
        verbose=args.verbose,
    )

    # Output
    os.makedirs(args.output_dir, exist_ok=True)
    review_path = os.path.join(args.output_dir, "knotknot_review.csv")
    full_path = os.path.join(args.output_dir, "knotknot_proposed_full.csv")
    safe_path = os.path.join(args.output_dir, "knotknot_safe_update.csv")

    review_df.to_csv(review_path, index=False)
    full_df.to_csv(full_path, index=False)
    safe_update_df.to_csv(safe_path, index=False)

    print("✓ Done.")
    print(f" - Review:        {review_path}")
    print(f" - Proposed Full: {full_path}")
    print(f" - Safe Update:   {safe_path}")
    print("\nNEXT:")
    print("  1) Inspect 'knotknot_review.csv' and confirm titles/types/SEO quickly.")
    print("  2) Spot-check 'knotknot_proposed_full.csv' for Body HTML & Alt Texts.")
    print("  3) When satisfied, import 'knotknot_safe_update.csv' into Shopify to update only the allowed fields.")

if __name__ == "__main__":
    main()
