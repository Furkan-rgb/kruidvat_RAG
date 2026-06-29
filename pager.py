import re
import asyncio
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse, urljoin
from bs4 import BeautifulSoup


def build_paginated_url(category_url, page_index, page_size=20, sort="score"):
    parsed = urlparse(category_url)
    q = parse_qs(parsed.query)
    q["page"] = [str(page_index)]
    q["size"] = [str(page_size)]
    q["sort"] = [sort]
    new_query = urlencode(q, doseq=True)
    return urlunparse(parsed._replace(query=new_query))


def get_total_products_from_text(text):
    if not text:
        return None
    # FIX 2: Tighter regex so it doesn't match prices or random text
    m = re.search(r"(\d+[.,]?\d*)\s*product(?:en)?\s*gevonden", text, flags=re.I)
    if m:
        return int(m.group(1).replace(".", "").replace(",", ""))
    return None


def extract_total_products_from_html(html):
    try:
        soup = BeautifulSoup(html, "html.parser")
        el = soup.select_one(".pager__title")
        if el and el.get_text(strip=True):
            return get_total_products_from_text(el.get_text())
        txt = soup.get_text(separator=" \n ")
        return get_total_products_from_text(txt)
    except Exception:
        return None


async def fetch_total_products(page):
    try:
        html = await page.content()
        return extract_total_products_from_html(html)
    except Exception:
        return None


def compute_pages_needed(total_count, page_size=20):
    if not total_count or total_count <= 0:
        return None
    return (total_count + page_size - 1) // page_size


async def collect_all_product_links(
    context,
    category_url,
    max_pages=200,
    page_size=20,  # Updated default to match typical Kruidvat limit
    delay=0.2,
    open_page_and_prep=None,
    extract_product_links=None,
):
    """Paginate the category and collect product links.

    Optional injectable helpers `open_page_and_prep` and `extract_product_links`
    allow tests or alternate implementations to be passed in.
    """
    links = []
    seen_links = set()
    page = await context.new_page()
    try:
        total = None
        last_page = max(0, max_pages - 1)
        consecutive_empty_pages = 0
        for page_index in range(0, max_pages):
            paged = build_paginated_url(category_url, page_index, page_size=page_size)
            try:
                await page.goto(paged, wait_until="domcontentloaded", timeout=30000)
            except Exception:
                break

            # Run optional prep if provided
            if open_page_and_prep:
                try:
                    await open_page_and_prep(page)
                except Exception:
                    pass

            html = await page.content()

            if total is None:
                total = extract_total_products_from_html(html)
                # FIX 1: Explicitly check for None so that total=0 doesn't break logic
                if total is not None:
                    pages_needed = compute_pages_needed(total, page_size)
                    if pages_needed is not None:
                        last_page = min(max_pages - 1, max(0, pages_needed - 1))

            # Use injected extractor or default to page-based selector
            if extract_product_links:
                new = await extract_product_links(page, category_url)
            else:
                # Strict extraction: only consider direct children of
                # the official product list container `#productList`.
                # For each direct product element, only use the anchor
                # inside that element (`a[href*='/p/']`). No fallbacks.
                containers = await page.query_selector_all("#productList > *")
                new = set()
                for c in containers:
                    try:
                        a = await c.query_selector("a[href*='/p/']")
                        if not a:
                            continue
                        href = await a.get_attribute("href")
                        if not href:
                            continue
                        href = href.strip()
                        if not href or href.lower().startswith("javascript:"):
                            continue
                        href = urljoin(category_url, href)
                        new.add(href)
                    except Exception:
                        pass
                new = sorted(new)

            if not new:
                consecutive_empty_pages += 1
            else:
                consecutive_empty_pages = 0

            for u in new:
                if u not in seen_links:
                    seen_links.add(u)
                    links.append(u)

            if page_index >= last_page:
                break

            if consecutive_empty_pages >= 3:
                break

            if delay:
                await asyncio.sleep(delay)

    finally:
        try:
            await page.close()
        except Exception:
            pass

    return links
