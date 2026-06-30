#!/usr/bin/env python3
"""Kruidvat product scraper (LLM-first extraction).

This is a trimmed, self-contained implementation focused on LLM-first
ingredient extraction. Legacy local parsing/fallbacks were removed; if the
local Ollama call does not return a usable payload the extractor returns
empty so callers can decide to skip or retry.

Run with no --category to scrape every category listed in config.py, or pass
one or more --category flags to override.
"""

import argparse
import asyncio
import json
import re
import sqlite3
from datetime import datetime, timezone
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from playwright_stealth import Stealth
from tqdm import tqdm

import config

DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS products (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    url TEXT UNIQUE,
    ingredients_list TEXT,
    scraped_at TEXT
);
"""


from browser import (
    make_browser_context,
    click_cookie_if_present,
    click_read_more,
    extract_name,
)
from extractor import (
    extract_ingredients_with_llm,
)
from pager import (
    extract_total_products_from_html,
    collect_all_product_links,
)
from db import setup_db, save_products_batch, read_existing_urls


async def open_page_and_prep(context, url):
    """Open a new page, navigate to `url` and run common prep actions.

    Returns the page instance (caller is responsible for closing it).
    """
    page = await context.new_page()
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
    except Exception:
        try:
            await page.close()
        except Exception:
            pass
        raise

    await click_cookie_if_present(page)
    await click_read_more(page)
    return page


async def fetch_total_products(page):
    """Return the total number of products extracted from the currently loaded page.

    Returns an int or None.
    """
    try:
        html = await page.content()
        return extract_total_products_from_html(html)
    except Exception:
        return None


def compute_pages_needed(total_count, page_size=100):
    if not total_count or total_count <= 0:
        return None
    return (total_count + page_size - 1) // page_size


async def scrape_single_product(page, url, *, ollama_model, ollama_url, ollama_timeout):
    await page.goto(url, wait_until="domcontentloaded", timeout=30000)
    # Consent cookie is already set context-wide by the link-collection step.
    await click_cookie_if_present(page, wait=False)

    async def _has_title():
        try:
            await page.wait_for_selector("h1", timeout=10000)
            return True
        except Exception:
            return False

    # Product detail is SPA-rendered and sometimes blank on first paint; reload once.
    if not await _has_title():
        try:
            await page.reload(wait_until="domcontentloaded", timeout=30000)
        except Exception:
            pass
        await _has_title()

    await click_read_more(page)
    name = await extract_name(page)

    ing_list = await extract_ingredients_with_llm(
        page,
        ollama_model=ollama_model,
        ollama_url=ollama_url,
        ollama_timeout=ollama_timeout,
    )

    if not ing_list:
        return None
    return (
        name or None,
        url,
        json.dumps(ing_list, ensure_ascii=False),
        datetime.now(timezone.utc).isoformat(),
    )


async def scrape_category(context, conn, existing, args, category):
    """Scrape a single category page and persist its products.

    `existing` is a set of already-seen URLs; it is updated in place so the
    same product is not scraped twice when it appears in multiple categories.
    """
    # 1) Collect product links via the category search API (the browser context
    #    passes the bot check; the API returns reliable, paginated product URLs).
    links = await collect_all_product_links(
        context,
        category,
        page_size=100,
        max_pages=args.max_pages,
        limit=args.limit,
        delay=args.delay,
    )

    # 2) Filter out already seen URLs and apply optional limit
    links = [u for u in links if u not in existing]
    if args.limit:
        links = links[: args.limit]
    # Mark these as seen so later categories don't re-scrape shared products.
    existing.update(links)

    # Concurrently scrape product pages
    sem = asyncio.Semaphore(args.concurrency)

    async def _scrape_one(u):
        async with sem:
            p = await context.new_page()
            try:
                row = await scrape_single_product(
                    p,
                    u,
                    ollama_model=args.ollama_model,
                    ollama_url=args.ollama_url,
                    ollama_timeout=args.ollama_timeout,
                )
                return row
            finally:
                try:
                    await p.close()
                except Exception:
                    pass

    rows_to_save = []
    if links:
        tasks = [asyncio.create_task(_scrape_one(u)) for u in links]
        for t in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=category):
            try:
                row = await t
            except Exception:
                row = None
            if row:
                rows_to_save.append(row)
            # Flush in batches
            if len(rows_to_save) >= 16:
                save_products_batch(conn, rows_to_save)
                rows_to_save = []

    if rows_to_save:
        save_products_batch(conn, rows_to_save)


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
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--max-pages", type=int, default=200)
    parser.add_argument("--delay", type=float, default=0.2)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N products per category",
    )
    parser.add_argument("--proxy", default=None)
    parser.add_argument(
        "--ollama-model",
        default=config.EXTRACT_MODEL,
        help="Local Ollama model name used for extraction",
    )
    parser.add_argument("--ollama-url", default=config.GENERATE_URL)
    parser.add_argument(
        "--ollama-timeout",
        type=float,
        default=config.OLLAMA_TIMEOUT,
        help="LLM request timeout in seconds (max 60)",
    )
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
