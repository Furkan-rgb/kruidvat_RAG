"""Kruidvat catalogue access via the SAP Commerce (OCC) API.

The product grid is a virtualized SPA, so instead of scraping the DOM we call
the same OCC API the site uses. A browser context is still needed once per run
to clear the Akamai bot check / consent (a plain server GET gets a 403); after
that we fetch the API from inside that context, which inherits the cookies.

- search endpoint  -> the paginated product list (URLs)
- products/{code}  -> one product's full record (name, description, ingredients)
"""

import json
from urllib.parse import urljoin, urlencode, urlparse, parse_qs

from .browser import click_cookie_if_present
from .extractor import split_ingredient_string

SEARCH_API = "https://api.kruidvat.nl/api/v2/kvn-spa/search"
PRODUCT_API = "https://api.kruidvat.nl/api/v2/kvn-spa/products/{code}?fields=FULL&lang=nl&curr=EUR"
PAGE_SIZE = 100  # max page size, so we fetch the fewest pages

# Parse the OCC search XML in the page context: product URLs + total page count.
_SEARCH_JS = r"""
async (url) => {
  const r = await fetch(url, {headers:{'Accept':'application/json'}, credentials:'include'});
  const t = await r.text();
  const doc = new DOMParser().parseFromString(t, 'application/xml');
  const pag = doc.querySelector('pagination');
  const totalPages = pag && pag.querySelector('totalPages')
    ? parseInt(pag.querySelector('totalPages').textContent, 10) : 1;
  const dget = (el, tag) => { for (const c of el.children) if (c.tagName === tag) return c.textContent; return null; };
  const urls = [...doc.querySelectorAll('products')].map(p => dget(p, 'url')).filter(Boolean);
  return {status: r.status, totalPages, urls};
}
"""

# Fetch any JSON endpoint from the page context and hand back the raw text.
_FETCH_TEXT_JS = r"""
async (url) => {
  const r = await fetch(url, {headers:{'Accept':'application/json'}, credentials:'include'});
  return JSON.stringify({status: r.status, body: await r.text()});
}
"""


def _search_url(code, current_page, page_size):
    return SEARCH_API + "?" + urlencode({
        "fields": "FULL", "searchType": "PRODUCT",
        "pageSize": page_size, "currentPage": current_page,
        "categoryCode": code, "lang": "nl", "curr": "EUR",
    })


def _code_from_url(url):
    """Product code is the last path segment after /p/ (…/p/5410980)."""
    if "/p/" in url:
        return url.rstrip("/").split("/p/")[-1].split("/")[0].split("?")[0]
    return None


async def open_catalog_page(context, category_url):
    """Open the category, accept consent, and return (page, category_code).

    The categoryCode is read from the SPA's own search request, so callers
    don't hardcode it. Leaves the page open and on the kruidvat origin so
    subsequent API fetches inherit its cookies.
    """
    page = await context.new_page()
    captured = {}

    def on_request(req):
        u = req.url
        if "/kvn-spa/search" in u and "categoryCode=" in u:
            code = parse_qs(urlparse(u).query).get("categoryCode")
            if code:
                captured["code"] = code[0]

    page.on("request", on_request)
    await page.goto(category_url, wait_until="domcontentloaded", timeout=30000)
    await click_cookie_if_present(page)
    # First paint is blank after consent; a reload renders the SPA and fires the
    # category search request we read the categoryCode from.
    await page.reload(wait_until="domcontentloaded", timeout=30000)
    for _ in range(30):
        if captured.get("code"):
            break
        await page.wait_for_timeout(500)
    return page, captured.get("code")


async def collect_links(page, category_url, code, *, page_size=PAGE_SIZE, max_pages=200, limit=None):
    """Page through the search API and return absolute product URLs."""
    links, seen = [], set()
    if not code:
        return links
    total_pages, cur = max_pages, 0
    while cur < min(max_pages, total_pages):
        res = await page.evaluate(_SEARCH_JS, _search_url(code, cur, page_size))
        if not res or res.get("status") != 200:
            break
        total_pages = res.get("totalPages") or total_pages
        for u in res.get("urls", []):
            au = urljoin(category_url, u).split("?")[0]
            if "/p/" in au and au not in seen:
                seen.add(au)
                links.append(au)
        cur += 1
        if limit and len(links) >= limit:
            break
    return links


async def fetch_product(page, product_url):
    """Fetch one product's detail from the OCC API.

    Returns {name, description, ingredients} (ingredients is a cleaned list),
    or None if the product can't be fetched/parsed.
    """
    code = _code_from_url(product_url)
    if not code:
        return None
    try:
        raw = await page.evaluate(_FETCH_TEXT_JS, PRODUCT_API.format(code=code))
        res = json.loads(raw)
        if res.get("status") != 200:
            return None
        d = json.loads(res["body"])
    except Exception:
        return None

    ingredients_raw = ""
    for e in d.get("extendedAttributes") or []:
        if e.get("code") == "drugComposition" and e.get("value"):
            ingredients_raw = e["value"]
            break
    if not ingredients_raw:  # fallback: any attribute that reads like an INCI list
        for e in d.get("extendedAttributes") or []:
            v = str(e.get("value") or "")
            if v.strip().lower().startswith("ingredi"):
                ingredients_raw = v
                break

    return {
        "name": d.get("name"),
        "description": (d.get("description") or "").strip(),
        "ingredients": split_ingredient_string(ingredients_raw) if ingredients_raw else [],
    }
