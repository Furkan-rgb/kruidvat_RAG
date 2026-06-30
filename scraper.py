#!/usr/bin/env python3
"""Kruidvat product scraper (OCC API).

Collects products for one or more categories straight from Kruidvat's SAP
Commerce (OCC) API: the search endpoint for the product list, then the
product-detail endpoint for each product's name, description, and ingredients.
A browser context is opened once per run only to clear the bot check / consent;
all product data comes from the API (no per-page rendering, no LLM).

Run with no --category to scrape every category in config.py, or pass one or
more --category flags to override.
"""

import argparse
import asyncio
import json
from datetime import datetime, timezone

from playwright.async_api import async_playwright
from tqdm import tqdm

import config

from lib.browser import make_browser_context
from lib.api import open_catalog_page, collect_links, fetch_product
from lib.db import setup_db, save_products_batch, read_existing_urls


async def scrape_category(context, conn, existing, args, category):
    """Scrape a single category via the OCC API and persist its products.

    `existing` is a set of already-seen URLs; it is updated in place so the
    same product is not scraped twice when it appears in multiple categories.
    """
    # 1) Open the category once (clears consent / bot check) and read its code.
    page, code = await open_catalog_page(context, category)
    try:
        # 2) Collect product URLs from the search API.
        links = await collect_links(
            page, category, code, page_size=100, max_pages=args.max_pages, limit=args.limit
        )
        links = [u for u in links if u not in existing]
        if args.limit:
            links = links[: args.limit]
        existing.update(links)

        # 3) Fetch each product's detail (name, description, ingredients).
        rows = []
        for url in tqdm(links, desc=category):
            d = await fetch_product(page, url)
            if d and d.get("ingredients"):
                rows.append((
                    d.get("name") or None,
                    url,
                    d.get("description") or None,
                    json.dumps(d["ingredients"], ensure_ascii=False),
                    datetime.now(timezone.utc).isoformat(),
                ))
            if len(rows) >= 50:
                save_products_batch(conn, rows)
                rows = []
        if rows:
            save_products_batch(conn, rows)
    finally:
        try:
            await page.close()
        except Exception:
            pass


async def main_async():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--category",
        action="append",
        default=None,
        help="Category URL to scrape (repeatable). Defaults to config.CATEGORIES.",
    )
    parser.add_argument("--db", default=config.DB_PATH)
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--max-pages", type=int, default=200)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N products per category",
    )
    parser.add_argument("--proxy", default=None)
    args = parser.parse_args()

    categories = args.category or config.CATEGORIES

    conn = setup_db(args.db)
    existing = read_existing_urls(conn)

    async with async_playwright() as p:
        browser, context = await make_browser_context(
            p, headless=not args.headed, proxy_url=args.proxy
        )

        for category in categories:
            await scrape_category(context, conn, existing, args, category)

        try:
            await browser.close()
        except Exception:
            pass

    conn.close()


if __name__ == "__main__":
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        pass
