import asyncio
from urllib.parse import urljoin

from playwright_stealth import Stealth

# After the 2026 site redesign the product grid is rendered client-side as
# <a class="product-list-item__link" href=".../p/..."> inside
# <div class="product-grid__products-list">. The grid only renders once the
# OneTrust cookie banner has been accepted.
PRODUCT_LINK_SELECTOR = "a.product-list-item__link[href*='/p/']"


def normalize_url(base, href, strip_query=True):
    if not href:
        return None
    if href.startswith("http"):
        return href.split("?")[0] if strip_query else href
    absolute = urljoin(base, href)
    return absolute.split("?")[0] if strip_query else absolute


async def click_cookie_if_present(page, wait=True):
    """Dismiss the cookie consent wall.

    The site uses a OneTrust banner (#onetrust-accept-btn-handler) that gates
    the product grid. On the first navigation of a context we wait for it to
    attach; once accepted the consent cookie persists for the whole context, so
    later calls (wait=False) just click it if it happens to be visible.
    """
    try:
        ot = page.locator("#onetrust-accept-btn-handler")
        if wait:
            try:
                await ot.wait_for(state="visible", timeout=8000)
            except Exception:
                pass
        if await ot.count() > 0:
            try:
                await ot.first.click(timeout=3000)
                await asyncio.sleep(0.3)
                return
            except Exception:
                pass
    except Exception:
        pass

    # Fallback: generic accept buttons (older layout / other locales).
    try:
        locator = page.locator(
            'button:has-text("Akkoord"), button:has-text("Accepteer"), button:has-text("Accept")'
        )
        if await locator.count() > 0:
            await locator.first.click()
            await asyncio.sleep(0.2)
    except Exception:
        pass


async def wait_for_products(page, timeout=20000):
    """Wait for the client-rendered product grid to populate. Returns bool."""
    try:
        await page.wait_for_selector(PRODUCT_LINK_SELECTOR, timeout=timeout)
        return True
    except Exception:
        return False


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
    """Collect product URLs from the rendered grid (post-redesign markup)."""
    links = set()
    anchors = await page.query_selector_all(PRODUCT_LINK_SELECTOR)
    for a in anchors:
        try:
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
    # Chrome's "new" headless mode is far harder to bot-detect than Playwright's
    # default headless (the site won't render its product grid for the latter),
    # so for headless runs we launch with headless=False and force --headless=new.
    # Passing headless=False (--headed) opens a real visible window.
    launch_kwargs = dict(
        headless=False,
        args=browser_args + (["--headless=new"] if headless else []),
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
    await wait_for_products(page)
    return page
