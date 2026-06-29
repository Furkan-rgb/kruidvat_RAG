import asyncio
from urllib.parse import urljoin

from playwright_stealth import Stealth


def normalize_url(base, href, strip_query=True):
    if not href:
        return None
    if href.startswith("http"):
        return href.split("?")[0] if strip_query else href
    absolute = urljoin(base, href)
    return absolute.split("?")[0] if strip_query else absolute


async def click_cookie_if_present(page):
    try:
        locator = page.locator(
            'button:has-text("Akkoord"), button:has-text("Accepteer"), button:has-text("Accept")'
        )
        if await locator.count() > 0:
            await locator.first.click()
            await asyncio.sleep(0.2)
    except Exception:
        pass


async def click_read_more(page):
    try:
        locator = page.locator(
            'button:has-text("Lees meer"), button:has-text("Read more"), a:has-text("Lees meer")'
        )
        count = await locator.count()
        for i in range(min(count, 5)):
            try:
                await locator.nth(i).click(timeout=1000)
                await asyncio.sleep(0.08)
            except Exception:
                pass
    except Exception:
        pass


async def extract_product_links(page, base_url):
    links = set()
    # Only consider anchors that are inside direct children of #productList
    containers = await page.query_selector_all("#productList > *")
    for c in containers:
        try:
            a = await c.query_selector("a[href*='/p/']")
            if not a:
                continue
            href = await a.get_attribute("href")
            if href:
                u = normalize_url(base_url, href)
                if u:
                    links.add(u)
        except Exception:
            pass
    return sorted(links)


async def extract_name(page):
    try:
        h1 = page.locator("h1")
        if await h1.count() > 0:
            text = (await h1.first.inner_text()).strip()
            if text:
                return text
    except Exception:
        pass
    try:
        meta = await page.query_selector("meta[property='og:title']")
        if meta:
            v = await meta.get_attribute("content")
            if v:
                return v.strip()
    except Exception:
        pass
    try:
        t = await page.title()
        if t:
            return t.strip()
    except Exception:
        pass
    return ""


async def make_browser_context(playwright, headless=True, proxy_url=None):
    browser_args = [
        "--disable-blink-features=AutomationControlled",
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--disable-infobars",
        "--window-size=1280,800",
    ]
    launch_kwargs = dict(
        headless=False,
        args=browser_args + ["--headless=new"],
        ignore_default_args=["--enable-automation"],
    )
    if proxy_url:
        launch_kwargs["proxy"] = {"server": proxy_url}

    browser = await playwright.chromium.launch(**launch_kwargs)

    context = await browser.new_context(
        user_agent=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        ),
        locale="nl-NL",
        timezone_id="Europe/Amsterdam",
        viewport={"width": 1280, "height": 800},
        device_scale_factor=1,
        is_mobile=False,
        has_touch=False,
        color_scheme="light",
        extra_http_headers={
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "nl-NL,nl;q=0.9,en-US;q=0.8,en;q=0.7",
        },
    )

    stealth = Stealth()
    await stealth.apply_stealth_async(context)
    return browser, context


async def open_page_and_prep(context, url):
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
