import re
import asyncio
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse, urljoin
from bs4 import BeautifulSoup

from browser import click_cookie_if_present

# The product grid is client-rendered and virtualized, so instead of scraping the
# DOM we call the same SAP Commerce (OCC) search API the SPA uses. It returns the
# full, paginated product list as XML. We call it from *inside* the browser
# context so it inherits the Akamai/consent cookies (a plain GET gets a 403).
SEARCH_API = "https://api.kruidvat.nl/api/v2/kvn-spa/search"
PAGE_SIZE = 100  # max page size, so we fetch the fewest pages

# Runs in the page context: fetch the search API, parse the XML, return the
# product page URLs (direct-child <url> of each <products>) plus total page count.
_FETCH_JS = r"""
async (url) => {
  const r = await fetch(url, {headers:{'Accept':'application/json'}, credentials:'include'});
  const t = await r.text();
  let doc;
  try { doc = new DOMParser().parseFromString(t, 'application/xml'); }
  catch(e){ return {status:r.status, totalPages:1, urls:[], err:String(e)}; }
  const pag = doc.querySelector('pagination');
  const totalPages = pag && pag.querySelector('totalPages')
    ? parseInt(pag.querySelector('totalPages').textContent, 10) : 1;
  const dget = (el, tag) => { for (const c of el.children) if (c.tagName === tag) return c.textContent; return null; };
  const urls = [...doc.querySelectorAll('products')].map(p => dget(p, 'url')).filter(Boolean);
  return {status:r.status, totalPages, urls};
}
"""


def _search_url(category_code, current_page, page_size):
    q = urlencode({
        "fields": "FULL",
        "searchType": "PRODUCT",
        "pageSize": page_size,
        "currentPage": current_page,
        "categoryCode": category_code,
        "lang": "nl",
        "curr": "EUR",
    })
    return f"{SEARCH_API}?{q}"


# --- legacy HTML helpers (kept for compatibility; no longer the primary path) ---

def build_paginated_url(category_url, page_index, page_size=PAGE_SIZE, sort=None):
    parsed = urlparse(category_url)
    q = parse_qs(parsed.query)
    q["pageSize"] = [str(page_size)]
    q["currentPage"] = [str(page_index)]
    if sort:
        q["sort"] = [sort]
    return urlunparse(parsed._replace(query=urlencode(q, doseq=True)))


def get_total_products_from_text(text):
    if not text:
        return None
    m = re.search(r"(\d[\d.,]*)\s*product(?:\(en\)|en)?\s*gevonden", text, flags=re.I)
    if m:
        return int(m.group(1).replace(".", "").replace(",", ""))
    return None


def extract_total_products_from_html(html):
    try:
        soup = BeautifulSoup(html, "html.parser")
        el = soup.select_one(".total-found__label, .pager__title")
        if el and el.get_text(strip=True):
            n = get_total_products_from_text(el.get_text())
            if n is not None:
                return n
        return get_total_products_from_text(soup.get_text(separator=" \n "))
    except Exception:
        return None


async def fetch_total_products(page):
    try:
        return extract_total_products_from_html(await page.content())
    except Exception:
        return None


def compute_pages_needed(total_count, page_size=PAGE_SIZE):
    if not total_count or total_count <= 0:
        return None
    return (total_count + page_size - 1) // page_size


async def _capture_category_code(context, category_url):
    """Load the category page (accept consent) and read the categoryCode from
    the SPA's own search request, so callers don't need to hardcode it."""
    page = await context.new_page()
    captured = {}

    def on_request(req):
        u = req.url
        if "/kvn-spa/search" in u and "categoryCode=" in u:
            code = parse_qs(urlparse(u).query).get("categoryCode")
            if code:
                captured["code"] = code[0]

    page.on("request", on_request)
    try:
        await page.goto(category_url, wait_until="domcontentloaded", timeout=30000)
        await click_cookie_if_present(page)
        # First paint is blank after accepting consent; a reload renders the SPA
        # and triggers the category search request we're listening for.
        await page.reload(wait_until="domcontentloaded", timeout=30000)
        for _ in range(30):
            if captured.get("code"):
                break
            await page.wait_for_timeout(500)
    finally:
        pass
    return page, captured.get("code")


async def collect_all_product_links(
    context, category_url, *, page_size=PAGE_SIZE, max_pages=200, limit=None, delay=0.3
):
    """Collect product URLs for a category via the OCC search API.

    Opens the category once (to clear consent and discover the categoryCode),
    then pages through the API from the browser context. Returns absolute
    product URLs (deduped, in order).
    """
    page, code = await _capture_category_code(context, category_url)
    links, seen = [], set()
    try:
        if not code:
            return links
        total_pages = max_pages
        cur = 0
        while cur < min(max_pages, total_pages):
            res = await page.evaluate(_FETCH_JS, _search_url(code, cur, page_size))
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
            if delay:
                await page.wait_for_timeout(int(delay * 1000))
    finally:
        try:
            await page.close()
        except Exception:
            pass
    return links
