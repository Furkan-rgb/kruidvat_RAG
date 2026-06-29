import asyncio
import json
import re
import traceback
from datetime import datetime
from urllib import error as urllib_error
from urllib import request as urllib_request

from bs4 import BeautifulSoup

from logging_config import get_logger

# module-level logger
logger = get_logger("kruidvat.extractor")


# Global semaphore to ensure only one LLM request runs at a time.
# Creating the semaphore at import time is fine in modern asyncio.
_llm_semaphore = asyncio.Semaphore(1)

OLLAMA_SYSTEM_PROMPT = """You are an expert data extractor. Your task is to extract ONLY cosmetic/chemical INCI ingredients from Dutch webshop text.

Rules:
1. Return strict JSON only. Schema: {"found": true|false, "ingredients": ["..."]}
2. Only extract actual cosmetic/liquid/powder chemical ingredients (e.g., "Aqua", "Cetearyl Alcohol", "Linalool").
3. CRITICAL: Hardware, electrical appliances, and physical accessories (e.g., hair dryers, straighteners, brushes) DO NOT have cosmetic ingredients. If the product is a tool or device, return {"found": false, "ingredients": []}.
4. DO NOT extract device materials (e.g., "Titanium", "Ceramic", "keramische coating").
5. DO NOT extract features or technologies (e.g., "negatieve ionen", "ThermoProtect", "ionische technologie").
6. DO NOT extract physical parts (e.g., "Diffuser", "Concentrator", "Stijlborstel", "blaasmond").
7. DO NOT extract generic categories or placeholder text (e.g., "NVT", "Haarspray", "Tangle Teezer").
8. Only extract if there is an explicit ingredients section (look for headers like "Ingrediënten", "Ingredients", "Samenstelling"). If missing, return found: false.
9. Exclude prices, marketing text, product descriptions, warnings, instructions, shipping, reviews, and UI text.
"""


def clean_ingredient(s):
    if not s:
        return ""
    s = s.strip()
    # Remove leading labels like 'Ingrediënten:' or 'Samenstelling:'
    s = re.sub(r"(?i)^(?:ingredi[eë]nten|samenstelling)\s*[:\-\s]*", "", s)
    # Strip surrounding punctuation
    s = s.strip(" .;:-")
    # Collapse spaces
    s = re.sub(r"\s+", " ", s)
    return s


def split_ingredient_string(s):
    # Split on commas/•/·/; then clean each
    if not s:
        return []
    # Remove any leading free text before a JSON-like label
    s = re.sub(r"(?i).*?(?:ingredi[eë]nten|samenstelling)[:\-\s]*", "", s)
    s = s.replace("·", ", ").replace("•", ", ")
    parts = [p.strip() for p in re.split(r",|;", s) if p.strip()]
    return [clean_ingredient(p) for p in parts if clean_ingredient(p)]


def parse_llm_ingredients_response(response_text):
    """Normalize LLM response into (raw_text, [ingredients]).

    Accepts either a bare JSON string or wrapped text containing JSON. The
    "ingredients" field can be either a list or a comma-separated string.
    """
    if not response_text:
        return "", []

    # Extract first {...} JSON object if present.
    json_blob = None
    try:
        # Fast path: whole response is JSON
        parsed = json.loads(response_text)
        json_blob = parsed
    except Exception:
        # Try to find JSON object inside text
        m = re.search(r"\{[\s\S]*\}", response_text)
        if m:
            try:
                json_blob = json.loads(m.group(0))
            except Exception:
                json_blob = None

    ingredients_raw = ""
    ingredients_list = []

    if isinstance(json_blob, dict):
        # The LLM may return a boolean 'found' and an 'ingredients' value.
        ing = json_blob.get("ingredients")
        if isinstance(ing, list):
            ingredients_list = [clean_ingredient(i) for i in ing if i]
            ingredients_raw = ", ".join(ingredients_list)
        elif isinstance(ing, str):
            ingredients_raw = ing
            ingredients_list = split_ingredient_string(ing)
    else:
        # If no JSON, try treating the whole response as a CSV-like list
        ingredients_list = split_ingredient_string(response_text)
        ingredients_raw = ", ".join(ingredients_list)

    # Final normalization: remove empty and dedupe while preserving order
    seen = set()
    final = []
    for i in ingredients_list:
        k = i.lower()
        if k and k not in seen:
            seen.add(k)
            final.append(i)
    return ingredients_raw, final


def _call_ollama_extract_sync(prompt, model, ollama_url, timeout):
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "format": "json",
        "system": OLLAMA_SYSTEM_PROMPT,
        "options": {"temperature": 0},
        "prompt": prompt,
    }

    req = urllib_request.Request(
        ollama_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib_request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
    except (urllib_error.URLError, TimeoutError, OSError):
        return "", []

    try:
        outer = json.loads(body)
    except Exception:
        return "", []

    response_text = outer.get("response", "")

    return parse_llm_ingredients_response(response_text)


async def extract_ingredients_with_llm(
    page, *, ollama_model, ollama_url, ollama_timeout
):
    html = await page.content()
    # Produce cleaned plain text from the full HTML: remove scripts/styles and
    # return visible text only. This is intentionally the whole page text so
    # the LLM receives full context (no images or binary data).
    try:
        soup = BeautifulSoup(html, "html.parser")
        # If the product label accordion (ingredients) exists, extract only
        # that block's visible text and use it for the LLM — this localizes
        # sanitization specifically for product pages and avoids losing
        # ingredient text.
        label_accordion = soup.select_one(
            'e2-accordion[data-tab-type="label-info"], e2-accordion[name="LabelInformationTabComponent"]'
        )
        if label_accordion:
            try:
                candidate_text = label_accordion.get_text(separator=" ", strip=True)
            except Exception:
                candidate_text = ""
        else:
            # Preserve the product list block if present so sanitization doesn't
            # strip product entries. We extract it, clean the rest of the page,
            # then re-append its visible text to the candidate text.
            product_list = soup.select_one("#productList")
            product_list_text = ""
            if product_list:
                try:
                    product_list_text = product_list.get_text(separator=" ", strip=True)
                    product_list.extract()
                except Exception:
                    product_list_text = ""

            tags_to_remove = [
                "script",
                "style",
                "noscript",
                "header",
                "footer",
                "nav",
                "svg",
                "form",
                "picture",
                "img",
                "template",
                "iframe",
                "button",
                "input",
                "dialog",
                "e2-navigation-menu",
                "e2-navigation-element",
                "e2-header",
                "e2-smart-banner",
                "e2-customer-service",
                "e2-medical-dialog",
                # Standard HTML additions
                "head",
                "video",
                "audio",
                "source",
                "track",
                "canvas",
                "map",
                "area",
                "object",
                "embed",
                "select",
                "textarea",
                # Custom Site additions
                "e2-impression-tracker",
                "e2-searchbox",
                "e2-product-suggestions",
                "e2-product-thumbnails",
                "e2-carousel",
                "e2-sticky-row",
                "e2-rating",
                "e2-app",
            ]
            for t in soup(tags_to_remove):
                t.decompose()
            # Combine preserved product-list text with the cleaned page text so the
            # LLM still sees product entries even after sanitization.
            cleaned_page_text = soup.get_text(separator=" ", strip=True)
            candidate_text = (product_list_text + " " + cleaned_page_text).strip()
    except Exception as e:
        candidate_text = ""
        logger.error(
            json.dumps(
                {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "event": "html_parse_error",
                    "error": str(e),
                    "page_url": getattr(page, "url", None),
                }
            )
        )

    if not candidate_text:
        logger.info(
            json.dumps(
                {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "event": "no_candidate_text",
                    "reason": "empty page text after cleaning",
                    "page_url": getattr(page, "url", None),
                }
            )
        )
        return []

    try:
        async with _llm_semaphore:
            result = await asyncio.to_thread(
                _call_ollama_extract_sync,
                candidate_text,
                ollama_model,
                ollama_url,
                ollama_timeout,
            )
            if result and result[1]:
                raw, ingredients = result
                logger.info(
                    json.dumps(
                        {
                            "timestamp": datetime.utcnow().isoformat() + "Z",
                            "event": "extraction_accepted",
                            "found": True,
                            "product_url": getattr(page, "url", None),
                            "ingredients": ingredients,
                        }
                    )
                )
                return ingredients
            else:
                logger.info(
                    json.dumps(
                        {
                            "timestamp": datetime.utcnow().isoformat() + "Z",
                            "event": "extraction_rejected",
                            "found": False,
                            "product_url": getattr(page, "url", None),
                            "reason": "no ingredients returned by LLM",
                        }
                    )
                )
    except Exception as e:
        logger.error(
            json.dumps(
                {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "event": "llm_extract_error",
                    "error": traceback.format_exc(),
                    "product_url": getattr(page, "url", None),
                }
            )
        )

    # No LLM result — return empty (no local fallback).
    return []


async def extract_ingredients(page, *, ollama_model, ollama_url, ollama_timeout):
    return await extract_ingredients_with_llm(
        page,
        ollama_model=ollama_model,
        ollama_url=ollama_url,
        ollama_timeout=ollama_timeout,
    )
