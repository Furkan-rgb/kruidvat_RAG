"""Stealthed Playwright browser used only to clear the bot check / consent.

The catalogue is read from Kruidvat's API (see lib/api.py); the browser exists
to launch a hard-to-detect Chromium and dismiss the OneTrust cookie banner, so
the API calls made from inside the page context inherit the right cookies.
"""

import asyncio

from playwright_stealth import Stealth


async def click_cookie_if_present(page, wait=True):
    """Dismiss the cookie consent wall.

    The site uses a OneTrust banner (#onetrust-accept-btn-handler) that gates
    the catalogue. On the first navigation of a context we wait for it to
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


async def make_browser_context(playwright, headless=True, proxy_url=None):
    browser_args = [
        "--disable-blink-features=AutomationControlled",
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--disable-infobars",
        "--window-size=1280,800",
    ]
    # Chrome's "new" headless mode is far harder to bot-detect than Playwright's
    # default headless (the site won't serve its API for the latter), so for
    # headless runs we launch with headless=False and force --headless=new.
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
