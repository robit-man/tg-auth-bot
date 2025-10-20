#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools.py — Tool calling framework with a Selenium-powered web search tool and helpers.

Updates
-------
• Headless OFF by default (GUI): Tools.open_browser(headless=False), search_internet() opens GUI.
• Primary content path: capture full DOM HTML via Selenium (authentic browser execution).
• Fallback content path: bs4_scrape now:
    - imports Selenium cookies into requests.Session (domain-matched),
    - applies browser-like headers (UA rotation, Accept/Language/Referer),
    - optional jitter and retries,
    - returns raw HTML or visible text per `verbosity`.

Features
--------
• Tools.search_internet(topic, num_results=5, wait_sec=1, deep_scrape=True, summarize=True, bs4_verbose=False, headless=False)
  - DuckDuckGo search → open top results in tabs → capture DOM → optional bs4 fallback.
  - Returns list[dict]: title, url, snippet, content, aux_summary (if summarize=True).

Dependencies
------------
selenium, webdriver-manager, beautifulsoup4, lxml, requests, ollama
"""

from __future__ import annotations

import base64
import html
import inspect
import json
import os
import platform
import random
import shutil
import subprocess
import textwrap
import time
import traceback
import fnmatch  # <-- added for file helpers
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Set, Callable
from urllib.parse import urlparse, quote_plus

import requests
from bs4 import BeautifulSoup
import ollama  # type: ignore

from selenium import webdriver  # type: ignore
from selenium.webdriver.common.by import By  # type: ignore
from selenium.webdriver.common.keys import Keys  # type: ignore
from selenium.webdriver.chrome.options import Options  # type: ignore
from selenium.webdriver.chrome.service import Service  # type: ignore
from selenium.webdriver.support.ui import WebDriverWait  # type: ignore
from selenium.webdriver.support import expected_conditions as EC  # type: ignore
from selenium.common.exceptions import (  # type: ignore
    TimeoutException,
    NoSuchElementException,
    WebDriverException,
)

from webdriver_manager.chrome import ChromeDriverManager  # type: ignore

from cognition import CognitionChat, CognitionConfig

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = str(SCRIPT_DIR)  # <-- added default workspace root
WORKSPACE_ROOT = Path(WORKSPACE_DIR).resolve()

# ─────────────────────────────────────────────────────────────────────────────
# Autonomous search agent prompts & schemas
# ─────────────────────────────────────────────────────────────────────────────

_AGENT_PLAN_SYSTEM = (
    "You are a senior web research navigator. Suggest the next 1-3 concrete actions to gather "
    "evidence for the objective. Allowed actions:\n"
    "• search — run a web search with a short query (fields: query, focus, max_results 1-10, deep true/false).\n"
    "• visit — open a specific URL and capture targeted information (fields: url, extraction_goal, notes).\n"
    "• synthesize — finish when enough evidence exists (include notes on what to summarize).\n"
    "Always ground decisions in the provided history. Prefer precise, minimal steps. Return ONLY JSON."
)

_AGENT_PLAN_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["steps"],
    "properties": {
        "steps": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": {
                "type": "object",
                "required": ["id", "action", "description"],
                "properties": {
                    "id": {"type": "string"},
                    "action": {"type": "string", "enum": ["search", "visit", "synthesize"]},
                    "description": {"type": "string"},
                    "query": {"type": "string"},
                    "focus": {"type": "string"},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 10},
                    "deep": {"type": "boolean"},
                    "url": {"type": "string"},
                    "extraction_goal": {"type": "string"},
                    "notes": {"type": "string"},
                    "exit_on_completion": {"type": "boolean"},
                },
                "additionalProperties": False,
            },
        },
        "stop_hint": {"type": "string"},
    },
    "additionalProperties": False,
}

_AGENT_EVAL_SYSTEM = (
    "You guard readiness to stop web research. Review the objective, success criteria, and recent "
    "findings. Respond with JSON indicating whether work should stop, whether the objective is met, "
    "and the rationale."
)

_AGENT_EVAL_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["should_stop"],
    "properties": {
        "should_stop": {"type": "boolean"},
        "meets_success": {"type": "boolean"},
        "reason": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "missing": {"type": "string"},
    },
    "additionalProperties": False,
}

_AGENT_SUMMARY_SYSTEM = (
    "You synthesize vetted web research. Given the objective, success criteria, and evidence snippets "
    "with URLs, produce a concise answer. Use bracketed citations like [1] that map to the provided "
    "sources array. If requirements are not met, explain what is missing instead of fabricating."
)

_MAX_AGENT_HISTORY = 5

# ─────────────────────────────────────────────────────────────────────────────
# Light logger
# ─────────────────────────────────────────────────────────────────────────────

def log_message(msg: str, level: str = "INFO") -> None:
    level = level.upper()
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level}] {msg}")

# ─────────────────────────────────────────────────────────────────────────────
# Browser-like header fabrication
# ─────────────────────────────────────────────────────────────────────────────

_UA_CANDIDATES = [
    # Recent Chrome desktop UAs (rotate to reduce signature)
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
]

_LANG_CANDIDATES = [
    "en-US,en;q=0.9",
    "en-GB,en;q=0.9",
    "en-US,en;q=0.8,es;q=0.6",
]

def random_ua() -> str:
    return random.choice(_UA_CANDIDATES)

def build_browser_like_headers(
    *,
    ua: Optional[str] = None,
    referer: Optional[str] = None,
    accept_language: Optional[str] = None,
) -> Dict[str, str]:
    ua = ua or random_ua()
    accept_language = accept_language or random.choice(_LANG_CANDIDATES)
    headers = {
        "User-Agent": ua,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": accept_language,
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "same-origin" if referer else "none",
        "Sec-Fetch-User": "?1",
    }
    if referer:
        headers["Referer"] = referer
    return headers

# ─────────────────────────────────────────────────────────────────────────────
# Tools class
# ─────────────────────────────────────────────────────────────────────────────

class Tools:
    _driver: Optional[webdriver.Chrome] = None
    _runtime_tools: Dict[str, Callable[..., Any]] = {}
    _runtime_docs: Dict[str, str] = {}

    # ===================== Low-level helpers =====================

    @staticmethod
    def describe_tools() -> List[Dict[str, str]]:
        """Return metadata (name, signature, docstring) for public tool methods."""
        catalog: List[Dict[str, str]] = []
        for attr in dir(Tools):
            if attr.startswith("_"):
                continue
            fn = getattr(Tools, attr)
            if callable(fn):
                try:
                    sig = str(inspect.signature(fn))
                except Exception:
                    sig = "(...)"
                doc = inspect.getdoc(fn) or ""
                catalog.append({"name": attr, "signature": sig, "doc": doc})
        for name, fn in Tools._runtime_tools.items():
            try:
                sig = str(inspect.signature(fn))
            except Exception:
                sig = "(...)"
            doc = Tools._runtime_docs.get(name, inspect.getdoc(fn) or "")
            catalog.append({"name": name, "signature": sig, "doc": doc})
        return catalog

    @staticmethod
    def register_runtime_tool(name: str, fn: Callable[..., Any], doc: str = "") -> None:
        Tools._runtime_tools[name] = fn
        Tools._runtime_docs[name] = doc

    @staticmethod
    def runtime_tool_functions() -> Dict[str, Callable[..., Any]]:
        return dict(Tools._runtime_tools)

    @staticmethod
    def _resolve_workspace_path(rel_path: str,
                                base_dir: str = WORKSPACE_DIR) -> Path:
        """
        Resolve a path relative to the workspace and ensure it does not escape.
        """
        base = Path(base_dir).resolve()
        dest = (base / rel_path).resolve()
        try:
            dest.relative_to(WORKSPACE_ROOT)
        except ValueError as exc:
            raise ValueError(f"path '{rel_path}' escapes workspace root") from exc
        return dest

    @staticmethod
    def _find_system_chromedriver() -> str | None:
        candidates: list[str | None] = [
            shutil.which("chromedriver"),
            "/usr/bin/chromedriver",
            "/usr/local/bin/chromedriver",
            "/snap/bin/chromium.chromedriver",
            "/usr/lib/chromium-browser/chromedriver",
            "/opt/homebrew/bin/chromedriver",
        ]
        for path in filter(None, candidates):
            if os.path.isfile(path) and os.access(path, os.X_OK):
                try:
                    subprocess.run([path, "--version"],
                                   check=True,
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL)
                    return path
                except Exception:
                    continue
        return None

    @staticmethod
    def _wait_for_ready(drv, timeout=6):
        WebDriverWait(drv, timeout).until(
            lambda d: d.execute_script("return document.readyState") == "complete"
        )

    @staticmethod
    def _first_present(drv, selectors: list[str], timeout=4):
        for sel in selectors:
            try:
                return WebDriverWait(drv, timeout).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, sel))
                )
            except TimeoutException:
                continue
        return None

    @staticmethod
    def _visible_and_enabled(locator):
        def _cond(drv):
            try:
                el = drv.find_element(*locator)
                return el.is_displayed() and el.is_enabled()
            except Exception:
                return False
        return _cond

    # ===================== Browser lifecycle =====================

    @staticmethod
    def open_browser(headless: bool = False, force_new: bool = False) -> str:
        """
        Launch Chrome/Chromium robustly across platforms.
        NOTE: headless=False by default (GUI ON).
        """
        # Tear down existing driver if requested
        if force_new and Tools._driver:
            try:
                Tools._driver.quit()
            except Exception:
                pass
            Tools._driver = None

        if Tools._driver:
            return "Browser already open"

        chrome_bin = (
            os.getenv("CHROME_BIN")
            or shutil.which("google-chrome")
            or shutil.which("chromium-browser")
            or shutil.which("chromium")
            or "/snap/bin/chromium"
            or "/usr/bin/chromium-browser"
            or "/usr/bin/chromium"
        )

        opts = Options()
        if chrome_bin:
            opts.binary_location = chrome_bin
        if headless:
            opts.add_argument("--headless=new")
        opts.add_argument("--window-size=1920,1080")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--remote-allow-origins=*")
        opts.add_argument(f"--remote-debugging-port={random.randint(45000, 65000)}")

        # 1) Selenium-Manager
        try:
            log_message("[open_browser] Trying Selenium-Manager…", "DEBUG")
            Tools._driver = webdriver.Chrome(options=opts)
            log_message("[open_browser] Launched via Selenium-Manager.", "SUCCESS")
            return "Browser launched (selenium-manager)"
        except WebDriverException as e:
            log_message(f"[open_browser] Selenium-Manager failed: {e}", "WARNING")

        # 2) Snap’s bundled chromedriver
        snap_drv = "/snap/chromium/current/usr/lib/chromium-browser/chromedriver"
        if os.path.exists(snap_drv):
            try:
                log_message(f"[open_browser] Using snap chromedriver at {snap_drv}", "DEBUG")
                Tools._driver = webdriver.Chrome(service=Service(snap_drv), options=opts)
                log_message("[open_browser] Launched via snap chromedriver.", "SUCCESS")
                return "Browser launched (snap chromedriver)"
            except WebDriverException as e:
                log_message(f"[open_browser] Snap chromedriver failed: {e}", "WARNING")

        # 3) System chromedriver
        sys_drv = Tools._find_system_chromedriver()
        if sys_drv:
            try:
                log_message(f"[open_browser] Trying system chromedriver at {sys_drv}", "DEBUG")
                Tools._driver = webdriver.Chrome(service=Service(sys_drv), options=opts)
                log_message("[open_browser] Launched via system chromedriver.", "SUCCESS")
                return "Browser launched (system chromedriver)"
            except WebDriverException as e:
                log_message(f"[open_browser] System chromedriver failed: {e}", "WARNING")

        # 4) ARM64 auto-download (best effort)
        arch = (platform.machine() or "").lower()
        if arch in ("aarch64", "arm64", "armv8l", "armv7l") and chrome_bin:
            try:
                raw = subprocess.check_output([chrome_bin, "--version"]).decode().strip()
                ver = raw.split()[1]
                url = (
                    f"https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/"
                    f"{ver}/linux-arm64/chromedriver-linux-arm64.zip"
                )
                tmp_zip = "/tmp/chromedriver_arm64.zip"
                log_message(f"[open_browser] Downloading ARM64 driver from {url}", "DEBUG")
                subprocess.check_call(["wget", "-qO", tmp_zip, url])
                subprocess.check_call(["unzip", "-o", tmp_zip, "-d", "/tmp"])
                subprocess.check_call(["sudo", "mv", "/tmp/chromedriver", "/usr/local/bin/chromedriver"])
                subprocess.check_call(["sudo", "chmod", "+x", "/usr/local/bin/chromedriver"])
                drv = shutil.which("chromedriver")
                log_message(f"[open_browser] Installed ARM64 driver at {drv}", "DEBUG")
                Tools._driver = webdriver.Chrome(service=Service(drv), options=opts)
                log_message("[open_browser] Launched via downloaded ARM64 chromedriver.", "SUCCESS")
                return "Browser launched (downloaded ARM64 chromedriver)"
            except Exception as e:
                log_message(f"[open_browser] ARM64 download/install failed: {e}", "WARNING")

        # 5) webdriver-manager (x86_64)
        if arch in ("x86_64", "amd64") and chrome_bin:
            try:
                raw = subprocess.check_output([chrome_bin, "--version"]).decode().strip()
                browser_major = raw.split()[1].split(".")[0]
            except Exception:
                browser_major = "latest"
            try:
                log_message(f"[open_browser] Installing ChromeDriver {browser_major} via webdriver-manager", "DEBUG")
                drv_path = ChromeDriverManager(driver_version=browser_major).install()
                Tools._driver = webdriver.Chrome(service=Service(drv_path), options=opts)
                log_message("[open_browser] Launched via webdriver-manager.", "SUCCESS")
                return "Browser launched (webdriver-manager)"
            except Exception as e:
                log_message(f"[open_browser] webdriver-manager failed: {e}", "ERROR")

        # 6) Last-resort: snap install chromium
        try:
            log_message("[open_browser] Attempting `sudo snap install chromium`…", "DEBUG")
            subprocess.check_call(["sudo", "snap", "install", "chromium"])
            Tools._driver = webdriver.Chrome(service=Service(snap_drv), options=opts)
            log_message("[open_browser] Launched via newly-installed snap chromium.", "SUCCESS")
            return "Browser launched (snap install fallback)"
        except Exception as e:
            log_message(f"[open_browser] Auto-snap install failed or Chrome still not found: {e}", "ERROR")

        raise RuntimeError(
            "No usable Chrome/Chromium driver. Install Chrome and a matching chromedriver, "
            "or set CHROME_BIN and ensure chromedriver is on PATH."
        )

    @staticmethod
    def close_browser() -> str:
        if Tools._driver:
            try:
                Tools._driver.quit()
                log_message("[close_browser] Browser closed.", "DEBUG")
            except Exception:
                pass
            Tools._driver = None
            return "Browser closed"
        return "No browser to close"

    @staticmethod
    def is_browser_open() -> bool:
        return Tools._driver is not None

    @staticmethod
    def ensure_browser(headless: bool = True) -> None:
        if not Tools.is_browser_open():
            Tools.open_browser(headless=headless, force_new=True)

    @staticmethod
    def navigate(url: str) -> str:
        if not Tools._driver:
            return "Error: browser not open"
        log_message(f"[navigate] → {url}", "DEBUG")
        Tools._driver.get(url)
        return f"Navigated to {url}"

    @staticmethod
    def click(selector: str, timeout: int = 8) -> str:
        if not Tools._driver:
            return "Error: browser not open"
        try:
            drv = Tools._driver
            el = WebDriverWait(drv, timeout).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
            )
            drv.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
            el.click()
            focused = drv.execute_script("return document.activeElement === arguments[0];", el)
            log_message(f"[click] {selector} clicked (focused={focused})", "DEBUG")
            return f"Clicked {selector}"
        except Exception as e:
            log_message(f"[click] Error clicking {selector}: {e}", "ERROR")
            return f"Error clicking {selector}: {e}"

    @staticmethod
    def input(selector: str, text: str, timeout: int = 8) -> str:
        if not Tools._driver:
            return "Error: browser not open"
        try:
            drv = Tools._driver
            el = WebDriverWait(drv, timeout).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
            )
            drv.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
            el.clear()
            el.send_keys(text + Keys.RETURN)
            log_message(f"[input] Sent {text!r} to {selector}", "DEBUG")
            return f"Sent {text!r} to {selector}"
        except Exception as e:
            log_message(f"[input] Error typing into {selector}: {e}", "ERROR")
            return f"Error typing into {selector}: {e}"

    @staticmethod
    def get_html() -> str:
        if not Tools._driver:
            return "Error: browser not open"
        return Tools._driver.page_source

    @staticmethod
    def screenshot(filename: str = "screenshot.png") -> str:
        if not Tools._driver:
            return "Error: browser not open"
        Tools._driver.save_screenshot(filename)
        return filename

    # ===================== Fetching & LLM helpers =====================

    @staticmethod
    def get_dom_snapshot(max_chars: int = 20000) -> str:
        if not Tools._driver:
            return ""
        try:
            dom = Tools._driver.execute_script("return document.documentElement.outerHTML;")
            if dom and len(dom) > max_chars:
                dom = dom[:max_chars]
            return dom or ""
        except Exception as exc:
            log_message(f"[dom] snapshot failed: {exc}", "WARNING")
            return ""

    @staticmethod
    def scroll(amount: int = 600) -> str:
        if not Tools._driver:
            return "Error: browser not open"
        try:
            Tools._driver.execute_script("window.scrollBy(0, arguments[0]);", amount)
            return f"Scrolled by {amount}"
        except Exception as exc:
            log_message(f"[scroll] failed: {exc}", "WARNING")
            return f"Error scrolling: {exc}"

    @staticmethod
    def _import_cookies_from_browser(sess: requests.Session, url: str) -> None:
        """Copy Selenium cookies for matching domains into the requests.Session."""
        if not Tools._driver:
            return
        try:
            parsed = urlparse(url)
            host = parsed.netloc
            for c in Tools._driver.get_cookies():
                dom = c.get("domain") or ""
                if host.endswith(dom.lstrip(".")):
                    sess.cookies.set(
                        name=c.get("name"),
                        value=c.get("value"),
                        domain=c.get("domain"),
                        path=c.get("path", "/"),
                    )
        except Exception as e:
            log_message(f"[cookies] import failed: {e}", "DEBUG")

    @staticmethod
    def _visible_text_from_html(html_text: str) -> str:
        soup = BeautifulSoup(html_text, "lxml")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        lines = [ln.strip() for ln in text.splitlines()]
        return "\n".join(ln for ln in lines if ln)

    @staticmethod
    def bs4_scrape(
        url: str,
        *,
        verbosity: bool = False,
        timeout: int = 12,
        referer: Optional[str] = None,
        use_browser_cookies: bool = True,
        jitter: bool = True,
        retries: int = 2,
    ) -> str:
        """
        Fetch a URL with obfuscated, browser-like headers and (optionally) Selenium cookies.
        Returns raw HTML (verbosity=True) or visible text only (verbosity=False).
        """
        sess = requests.Session()

        # Bring over cookies from the live browser session (if any)
        if use_browser_cookies:
            Tools._import_cookies_from_browser(sess, url)

        # Seed browser-like headers
        headers = build_browser_like_headers(
            ua=random_ua(),
            referer=referer,
            accept_language=None,
        )
        sess.headers.update(headers)

        last_err: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                if jitter:
                    time.sleep(random.uniform(0.15, 0.6))
                resp = sess.get(url, timeout=timeout, allow_redirects=True)
                resp.raise_for_status()
                html_text = resp.text
                return html_text if verbosity else Tools._visible_text_from_html(html_text)
            except Exception as e:
                last_err = e
                # rotate UA and referer a bit before next try
                sess.headers.update(build_browser_like_headers(
                    ua=random_ua(),
                    referer=referer,
                    accept_language=None,
                ))
                log_message(f"[bs4_scrape] attempt {attempt+1} failed for {url!r}: {e}", "WARNING")

        log_message(f"[bs4_scrape] GET failed for {url!r}: {last_err}", "WARNING")
        return ""

    @staticmethod
    def _duckduckgo_fallback(query: str, *, limit: int = 6, region: str = "us-en") -> List[Dict[str, Any]]:
        """Fallback HTML search using DuckDuckGo's lightweight endpoint."""
        encoded_query = quote_plus(query)
        ddg_url = f"https://duckduckgo.com/html/?q={encoded_query}&kl={region}"
        headers = build_browser_like_headers(referer="https://duckduckgo.com/")
        log_message(f"[duckduckgo_fallback] GET {ddg_url}", "INFO")
        try:
            resp = requests.get(ddg_url, headers=headers, timeout=12)
            resp.raise_for_status()
        except Exception as exc:
            log_message(f"[duckduckgo_fallback] request failed: {exc}", "WARNING")
            return []

        soup = BeautifulSoup(resp.text, "lxml")
        results: List[Dict[str, Any]] = []
        for body in soup.select("div.result__body"):
            link = body.select_one("a.result__a")
            snippet_el = body.select_one("a.result__snippet") or body.select_one("div.result__snippet")
            if not link:
                continue
            title = link.get_text(" ", strip=True)
            href = link.get("href") or ""
            snippet = snippet_el.get_text(" ", strip=True) if snippet_el else ""
            if not href:
                continue
            results.append(
                {
                    "title": title,
                    "url": href,
                    "snippet": snippet,
                    "content": snippet,
                    "aux_summary": snippet,
                    "extracted": snippet,
                }
            )
            if len(results) >= limit:
                break

        log_message(f"[duckduckgo_fallback] scraped {len(results)} entries", "INFO")
        return results

    @staticmethod
    def fetch_webpage(
        url: str,
        *,
        topic: Optional[str] = None,
        extraction_goal: Optional[str] = None,
        summarize: bool = True,
        max_chars: int = 20000,
        retries: int = 2,
        timeout: int = 14,
    ) -> Dict[str, Any]:
        """Fetch a webpage and optionally summarize it relative to an objective."""
        log_message(f"[fetch_webpage] fetching {url}")
        html_text = Tools.bs4_scrape(
            url,
            verbosity=True,
            timeout=timeout,
            referer=None,
            use_browser_cookies=True,
            jitter=True,
            retries=retries,
        )
        if not html_text:
            return {
                "url": url,
                "status": "empty",
                "summary": "",
                "content_preview": "",
                "body": "",
                "title": "",
                "extraction_goal": extraction_goal,
            }

        title = ""
        try:
            soup = BeautifulSoup(html_text, "lxml")
            title = (soup.title.string or "").strip()
        except Exception:
            title = ""

        visible_text = Tools._visible_text_from_html(html_text)
        body = visible_text[:max_chars]
        content_preview = body[:1200]

        summary = ""
        if summarize and body:
            cfg = CognitionConfig.load()
            chat = CognitionChat(
                CognitionConfig(
                    model=cfg.model,
                    temperature=0.0,
                    stream=False,
                    global_system_prelude=cfg.global_system_prelude,
                    ollama_options=cfg.ollama_options,
                    embed_model=cfg.embed_model,
                )
            )
            system_msg = (
                "You extract concise findings from a webpage. Focus on the research objective and "
                "the extraction goal. Use bullet points and include concrete facts, figures, or "
                "quotes when available. Do not speculate."
            )
            chat.set_system(system_msg)
            prompt = textwrap.dedent(
                f"""
                Objective: {topic or '(not provided)'}
                Extraction goal: {extraction_goal or '(not provided)'}
                Document excerpt (trimmed to {len(body)} chars):
                {body}
                """
            ).strip()
            try:
                summary = chat.raw_chat(prompt, stream=False) or ""
            except Exception as exc:
                log_message(f"[fetch_webpage] summarization failed: {exc}", "WARNING")
                summary = ""

        return {
            "url": url,
            "status": "ok",
            "title": title,
            "summary": summary.strip(),
            "content_preview": content_preview,
            "body": body,
            "extraction_goal": extraction_goal,
            "tokens_estimate": len(body.split()),
        }

    @staticmethod
    def describe_image(path: Path, *, model: Optional[str] = None, prompt: Optional[str] = None) -> str:
        model = model or os.environ.get("OLLAMA_VISION_MODEL", "gemma3:4b")
        prompt = prompt or "Provide a concise description of the key elements visible in this browser screenshot."
        try:
            img_b64 = base64.b64encode(path.read_bytes()).decode("ascii")
            buffer: List[str] = []
            print(f"[vision:{model}]", end=" ", flush=True)
            for chunk in ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": "You analyze browser screenshots."},
                    {"role": "user", "content": prompt, "images": [img_b64]},
                ],
                stream=True,
                options={"temperature": 0.0},
            ):
                delta = chunk.get("message", {}).get("content", "")
                if delta:
                    print(delta, end="", flush=True)
                    buffer.append(delta)
            print("", flush=True)
            return "".join(buffer).strip()
        except Exception as exc:
            log_message(f"[describe_image] failed: {exc}", "WARNING")
            return ""

    @staticmethod
    def capture_browser_state(
        label: str,
        out_dir: Path,
        *,
        describe: bool = True,
        vision_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not Tools._driver:
            return {}
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        filename = f"{label.replace(' ', '_')}_{timestamp}.png"
        path = out_dir / filename
        try:
            Tools._driver.save_screenshot(str(path))
        except Exception as exc:
            log_message(f"[capture_browser_state] screenshot failed: {exc}", "WARNING")
            return {}

        description = ""
        if describe:
            description = Tools.describe_image(path, model=vision_model)

        return {
            "screenshot": str(path),
            "vision": description,
        }

    @staticmethod
    def auxiliary_inference(
        *,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        stream: bool = True,
        prefix: str = "[aux] "
    ) -> str:
        """
        Run a small LLM call that optionally streams tokens to console live
        and returns the aggregated text (always). No generators are returned.
        """
        cfg = CognitionConfig.load()
        chat_model = model or cfg.model
        temp = cfg.temperature if temperature is None else temperature

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            pieces: List[str] = []
            print(prefix, end="", flush=True)
            for chunk in ollama.chat(
                model=chat_model,
                messages=messages,
                stream=True,
                options={"temperature": temp},
            ):
                tok = chunk.get("message", {}).get("content", "")
                if tok:
                    print(tok, end="", flush=True)
                    pieces.append(tok)
            print("", flush=True)
            return "".join(pieces).strip()
        except Exception as e:
            log_message(f"[auxiliary_inference] chat error: {e}", "ERROR")
            return ""

            return ""

    # ===================== Main tool: search_internet =====================

    @staticmethod
    def search_internet(topic: str,
                        num_results: int = 1,
                        wait_sec: int = 1,
                        deep_scrape: bool = True,
                        summarize: bool = True,
                        bs4_verbose: bool = False,
                        headless: bool = False,
                        *,
                        extract: bool = True,
                        extract_system: Optional[str] = None,
                        stream_aux: bool = True,
                        retain_browser: bool = False,
                        progress_callback: Optional[callable] = None,
                        **kwargs) -> list:
        """
        DuckDuckGo search → open top results → capture DOM (browser) → optional bs4 fallback →
        (optional) topic-focused summary + topic-focused targeted extraction.

        Returns list of dicts:
        {title, url, snippet, content, aux_summary?, extracted?}
        """

        # ---- Arg aliases (back-compat) -----------------------------------------
        if 'top_n' in kwargs:
            num_results = kwargs.pop('top_n')
        if 'n' in kwargs:
            num_results = kwargs.pop('n')
        if 'limit' in kwargs:
            num_results = kwargs.pop('limit')
        if 'retain_browser' in kwargs:
            retain_browser = bool(kwargs.pop('retain_browser'))
        if kwargs:
            log_message(f"[search_internet] Ignoring unexpected args: {list(kwargs.keys())!r}", "WARNING")

        # ---- Topic-normalized helpers ------------------------------------------
        def _shorten_query(q: str, fallback: str = "news") -> str:
            q = (q or "").strip()
            if not q:
                return fallback
            if len(q) <= 180 and "\n" not in q:
                return q
            import re
            words = re.findall(r"[A-Za-z0-9\-]{3,20}", q)[:10]
            return " ".join(words) if words else fallback

        t_topic = _shorten_query(str(topic or ""))
        log_message(f"[search_internet] ▶ {t_topic!r} (num_results={num_results}, summarize={summarize}, bs4_verbose={bs4_verbose})", "INFO")
        try:
            wait_sec = min(max(int(wait_sec), 1), 5)
        except Exception:
            wait_sec = 1

        # Helper to send progress updates
        def _progress(message: str):
            if progress_callback:
                try:
                    progress_callback(message)
                except Exception as e:
                    log_message(f"[search_internet] progress callback error: {e}", "WARNING")

        # ---- Build SYSTEM prompts derived from the topic ------------------------
        def _summary_system(t: str) -> str:
            return (
                "You write brief, factual summaries that focus STRICTLY on the given topic.\n"
                f"Topic: {t}\n"
                "Instructions:\n"
                "• Prefer 2–6 short bullets (or a tight paragraph if bullets are not natural).\n"
                "• Emphasize details directly relevant to the topic; ignore navigation, ads, and unrelated sections.\n"
                "• Include publication date and the page’s stated source if present in the provided text.\n"
                "• Avoid speculation and meta-commentary."
            )

        def _extract_system_from_topic(t: str) -> str:
            return (
                "You are an information extractor. Work ONLY with the provided page text.\n"
                f"Target topic: {t}\n"
                "Extract:\n"
                "• Key facts, entities, dates, figures, URLs explicitly present in text\n"
                "• Titles/headlines and brief, relevant context\n"
                "Rules:\n"
                "• Ignore ads, cookie banners, nav, subscription prompts, unrelated stories\n"
                "• Prefer a concise bullet list; keep each bullet to one sentence\n"
                "• If present, include canonical URL and publication/date hints\n"
                "• Do NOT invent information; do NOT include instructions or analysis"
            )

        # Helper: call auxiliary_inference and gracefully handle older signature (no stream param)
        def _aux_call(*, prompt: str, system: Optional[str], temperature: Optional[float] = None, stream: bool = True) -> str:
            try:
                return Tools.auxiliary_inference(prompt=prompt, system=system, temperature=temperature, stream=stream)
            except TypeError:
                # Older auxiliary_inference without `stream` param
                return Tools.auxiliary_inference(prompt=prompt, system=system, temperature=temperature)

        # ---- Fresh browser session (GUI by default) -----------------------------
        _progress(f"🌐 Opening browser (GUI mode)...")
        if not retain_browser:
            Tools.close_browser()
        Tools.open_browser(headless=headless, force_new=True)
        drv = Tools._driver
        wait = WebDriverWait(drv, wait_sec, poll_frequency=0.1)
        results: list = []

        serp_url = "https://duckduckgo.com/"
        try:
            # Home
            _progress(f"🔍 Navigating to DuckDuckGo...")
            drv.get(serp_url)
            wait.until(lambda d: d.execute_script("return document.readyState") == "complete")
            log_message("[search_internet] Home page ready.", "DEBUG")
            _progress(f"✓ DuckDuckGo loaded")

            # Cookie banner (best effort)
            try:
                btn = wait.until(EC.element_to_be_clickable(
                    (By.CSS_SELECTOR, "button#onetrust-accept-btn-handler")
                ))
                btn.click()
                log_message("[search_internet] Cookie banner dismissed.", "DEBUG")
            except TimeoutException:
                pass

            # Search box
            selectors = (
                "input#search_form_input_homepage",
                "input#searchbox_input",
                "input[name='q']",
            )
            box = None
            for sel in selectors:
                found = drv.find_elements(By.CSS_SELECTOR, sel)
                if found:
                    box = found[0]
                    break
            if not box:
                raise RuntimeError("Search box not found!")

            # Submit query
            _progress(f"⌨️ Typing search query: '{t_topic}'")
            drv.execute_script(
                "arguments[0].value = arguments[1];"
                "arguments[0].dispatchEvent(new Event('input'));"
                "arguments[0].form.submit();",
                box, t_topic
            )
            log_message("[search_internet] Query submitted.", "DEBUG")
            _progress(f"🔎 Submitting search...")

            # Wait for results
            try:
                wait.until(lambda d: "?q=" in d.current_url)
                wait.until(lambda d: d.find_elements(By.CSS_SELECTOR, "#links .result, #links [data-nr]"))
                log_message("[search_internet] Results detected.", "DEBUG")
                _progress(f"✓ Search results received")
            except TimeoutException:
                log_message("[search_internet] Results timeout; proceeding with whatever is visible.", "WARNING")
                _progress(f"⚠️ Slow response, proceeding with visible results")

            # Top anchors
            anchors = drv.find_elements(
                By.CSS_SELECTOR,
                "a.result__a, a[data-testid='result-title-a']"
            )[: int(num_results)]
            _progress(f"📋 Found {len(anchors)} results, processing each...")

            main_handle = drv.current_window_handle

            for idx, a in enumerate(anchors, 1):
                try:
                    href = a.get_attribute("href")
                    title = a.text.strip() or html.unescape(drv.execute_script("return arguments[0].innerText;", a))
                    _progress(f"\n📄 Result {idx}/{len(anchors)}: {title[:60]}...")

                    # snippet (best effort)
                    try:
                        parent = a.find_element(By.XPATH, "./ancestor::*[contains(@class,'result')][1]")
                        sn = parent.find_element(By.CSS_SELECTOR, ".result__snippet, span[data-testid='result-snippet']")
                        snippet = sn.text.strip()
                    except NoSuchElementException:
                        snippet = ""

                    page_content = ""
                    final_url = href

                    # ---- Deep scrape in new tab (primary: capture DOM) -----------
                    if deep_scrape and href:
                        _progress(f"🌐 Opening page in new tab: {href[:50]}...")
                        drv.switch_to.new_window("tab")
                        TAB_DEADLINE = time.time() + 12
                        drv.set_page_load_timeout(12)
                        try:
                            drv.get(href)
                            _progress(f"⏳ Page loading...")
                        except TimeoutException:
                            log_message(f"[search_internet] page-load timeout for {href!r}", "WARNING")
                            _progress(f"⚠️ Page load timeout, capturing partial content")

                        # Allow JS to settle a bit; watch for lull
                        _progress(f"⏱️ Waiting for page to stabilize (JavaScript execution)...")
                        drv.execute_script("""
                            window._lastMut = Date.now();
                            try {
                                new MutationObserver(function(){ window._lastMut = Date.now(); })
                                .observe(document, {childList:true,subtree:true,attributes:true});
                            } catch(e) {}
                        """)
                        STABLE_MS = 600
                        while True:
                            if time.time() >= TAB_DEADLINE:
                                log_message(f"[search_internet] hard deadline for {href!r}", "WARNING")
                                break
                            last = drv.execute_script("return window._lastMut || Date.now();")
                            try:
                                if (time.time()*1000) - float(last) > STABLE_MS:
                                    break
                            except Exception:
                                break
                            time.sleep(0.12)

                        # Capture DOM BEFORE closing tab
                        _progress(f"📸 Capturing page DOM and extracting text...")
                        try:
                            final_url = drv.current_url or href
                            html_dom = drv.execute_script("return document.documentElement.outerHTML;")
                            page_content = html_dom if bs4_verbose else Tools._visible_text_from_html(html_dom)
                            _progress(f"✓ Captured {len(page_content)} characters from page")
                        except Exception as e:
                            log_message(f"[search_internet] DOM capture failed: {e}", "WARNING")
                            _progress(f"❌ DOM capture failed: {e}")

                        # Close tab & switch back
                        try:
                            drv.close()
                        except Exception:
                            pass
                        drv.switch_to.window(main_handle)
                        _progress(f"🔙 Returned to search results")

                        # Fallback: if empty (blocked, JS empty shell), use obfuscated bs4 fetch with cookies
                        if not page_content:
                            page_content = Tools.bs4_scrape(
                                final_url,
                                verbosity=bs4_verbose,
                                timeout=12,
                                referer=drv.current_url if "duckduckgo.com" in drv.current_url else serp_url,
                                use_browser_cookies=True,
                                jitter=True,
                                retries=2,
                            )

                    # ---- Ensure we have some text to work with -------------------
                    if not page_content and final_url:
                        page_content = Tools.bs4_scrape(
                            final_url,
                            verbosity=bs4_verbose,
                            timeout=12,
                            referer=serp_url,
                            use_browser_cookies=False,
                            jitter=True,
                            retries=1,
                        )
                    if not page_content:
                        page_content = snippet or title or ""

                    # Trim long bodies for LLM calls
                    body_for_llm = (page_content or "")[:18000]

                    # ---- Topic-focused summary (optional) ------------------------
                    aux_summary = ""
                    if summarize and body_for_llm:
                        try:
                            _progress(f"🤖 Generating AI summary of page content...")
                            aux_prompt = f"{body_for_llm}"
                            log_message("[search_internet] aux summary (stream)", "DEBUG")
                            aux_summary = _aux_call(
                                prompt=aux_prompt,
                                system=_summary_system(t_topic),
                                stream=stream_aux,
                            )
                            _progress(f"✓ Summary generated ({len(aux_summary)} chars)")
                        except Exception as ex:
                            log_message(f"[search_internet] auxiliary_inference summary error: {ex}", "WARNING")
                            _progress(f"⚠️ Summary generation failed: {ex}")

                    # ---- Topic-focused targeted extraction (optional) ------------
                    extracted = ""
                    if extract and body_for_llm:
                        try:
                            _progress(f"🔍 Extracting key information for topic: '{t_topic}'...")
                            sys_for_extract = extract_system or _extract_system_from_topic(t_topic)
                            log_message("[search_internet] targeted extraction (stream)", "DEBUG")
                            extracted = _aux_call(
                                prompt=body_for_llm,
                                system=sys_for_extract,
                                temperature=0.2,
                                stream=stream_aux,
                            )
                            _progress(f"✓ Extracted key information ({len(extracted)} chars)")
                        except Exception as ex:
                            log_message(f"[search_internet] auxiliary_inference extraction error: {ex}", "WARNING")
                            _progress(f"⚠️ Extraction failed: {ex}")

                    # Basic fallback to avoid empty fields when requested
                    if extract and not (extracted or "").strip():
                        extracted = (aux_summary or snippet or title or "")[:4000]

                    # ---- Assemble result ----------------------------------------
                    entry = {
                        "title": title,
                        "url": final_url,
                        "snippet": snippet,
                        "content": page_content,
                    }
                    if summarize:
                        entry["aux_summary"] = aux_summary
                    if extract:
                        entry["extracted"] = extracted

                    results.append(entry)
                    _progress(f"✅ Result {idx} complete: {title[:60]}")

                except Exception as ex:
                    log_message(f"[search_internet] result error: {ex}", "WARNING")
                    _progress(f"❌ Error processing result {idx}: {ex}")
                    continue

            _progress(f"\n✨ Search complete! Processed {len(results)}/{len(anchors)} results successfully")

        except Exception as e:
            log_message(f"[search_internet] Fatal: {e}\n{traceback.format_exc()}", "ERROR")
            _progress(f"❌ Fatal error: {e}")
        finally:
            if not retain_browser:
                _progress(f"🔒 Closing browser...")
                Tools.close_browser()

        if not results:
            fallback = Tools._duckduckgo_fallback(t_topic, limit=num_results)
            if fallback:
                log_message("[search_internet] Browser search empty; using HTML fallback.", "WARNING")
                results = fallback
            else:
                log_message("[search_internet] No results found via browser or fallback.", "WARNING")

        if retain_browser:
            log_message(f"[search_internet] Collected {len(results)} results (browser retained).", "SUCCESS")
        else:
            log_message(f"[search_internet] Collected {len(results)} results.", "SUCCESS")
        return results


    @staticmethod
    def complex_search_agent(
        objective: str,
        *,
        success_criteria: Optional[str] = None,
        background: Optional[Sequence[str]] = None,
        max_iterations: int = 5,
        max_results: int = 6,
        headless: bool = True,
        deep_scrape: bool = True,
    ) -> Dict[str, Any]:
        """Autonomous multi-step web search with navigation and early stopping."""

        try:
            max_iterations = max(1, int(max_iterations))
        except Exception:
            max_iterations = 5
        try:
            max_results = max(1, int(max_results))
        except Exception:
            max_results = 6

        def _as_bool(value: Any, default: bool) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                val = value.strip().lower()
                if val in {"true", "1", "yes", "y"}:
                    return True
                if val in {"false", "0", "no", "n"}:
                    return False
            return default

        headless = _as_bool(headless, False)
        deep_scrape = _as_bool(deep_scrape, True)

        if isinstance(background, str):
            background_iter = [background]
        else:
            try:
                background_iter = list(background or [])
            except TypeError:
                background_iter = []
        background_list = [snip.strip()[:600] for snip in background_iter if isinstance(snip, str) and snip.strip()][:3]
        cfg = CognitionConfig.load()

        def _new_cfg(temp: float) -> CognitionConfig:
            return CognitionConfig(
                model=cfg.model,
                temperature=temp,
                stream=False,
                global_system_prelude=cfg.global_system_prelude,
                ollama_options=dict(cfg.ollama_options),
                embed_model=cfg.embed_model,
            )

        plan_chat = CognitionChat(_new_cfg(0.2))
        eval_chat = CognitionChat(_new_cfg(0.0))
        summary_chat = CognitionChat(_new_cfg(0.1))

        history: List[Dict[str, Any]] = []
        sources: Dict[str, Dict[str, str]] = {}
        plan_queue: List[Dict[str, Any]] = []
        executed_ids: Set[str] = set()
        stop_hint = ""
        stop_reason = ""
        success = False
        iterations = 0
        evaluation_snapshot: Optional[Dict[str, Any]] = None
        issues: List[str] = []
        plan_failures = 0

        def _history_for_model() -> List[Dict[str, Any]]:
            data: List[Dict[str, Any]] = []
            for item in history[-_MAX_AGENT_HISTORY:]:
                data.append(
                    {
                        "step_id": item.get("step_id"),
                        "action": item.get("action"),
                        "query": item.get("query"),
                        "url": item.get("url"),
                        "summary": (item.get("key_findings") or item.get("summary") or "")[:600],
                        "sources": item.get("sources", []),
                    }
                )
            return data

        def _generate_plan(remaining: int) -> None:
            nonlocal stop_hint, plan_failures
            payload = {
                "objective": objective,
                "success_criteria": success_criteria,
                "history": _history_for_model(),
                "background": background_list,
                "remaining_iterations": max(0, remaining),
                "issues": issues[-6:],
            }
            try:
                plan_chat.set_system(_AGENT_PLAN_SYSTEM)
                raw = plan_chat.structured_json(
                    user_message=json.dumps(payload, ensure_ascii=False),
                    json_schema=_AGENT_PLAN_SCHEMA,
                    stream=False,
                )
                if isinstance(raw, dict):
                    plan_resp = raw
                else:
                    plan_resp = json.loads(str(raw))
                plan_failures = 0
            except Exception as exc:
                plan_failures += 1
                msg = f"plan_generation_failed({plan_failures}): {exc}"
                log_message(f"[complex_search_agent] {msg}", "WARNING")
                issues.append(msg)
                fallback_step = {
                    "id": f"fallback_search_{len(history)+1}",
                    "action": "search",
                    "description": "Retry broad search after planner failure using HTML fallback backend.",
                    "query": objective,
                    "focus": success_criteria or "",
                    "max_results": max_results,
                    "deep": True,
                }
                plan_queue.append(fallback_step)
                return

            steps = plan_resp.get("steps") if isinstance(plan_resp, dict) else []
            if isinstance(steps, list) and steps:
                plan_queue.extend(steps)
            else:
                plan_failures += 1
                msg = f"plan_returned_no_steps({plan_failures})"
                issues.append(msg)
                log_message(f"[complex_search_agent] {msg}", "WARNING")
                fallback_step = {
                    "id": f"fallback_search_{len(history)+1}",
                    "action": "search",
                    "description": "Planner returned no steps; retrying manual search.",
                    "query": objective,
                    "focus": success_criteria or "",
                    "max_results": max_results,
                    "deep": True,
                }
                plan_queue.append(fallback_step)
                return
            hint = plan_resp.get("stop_hint") if isinstance(plan_resp, dict) else None
            if hint:
                stop_hint = str(hint)[:400]

        def _evaluate(latest: Dict[str, Any], remaining: int) -> Dict[str, Any]:
            payload = {
                "objective": objective,
                "success_criteria": success_criteria,
                "history": _history_for_model(),
                "latest": {
                    "action": latest.get("action"),
                    "summary": (latest.get("key_findings") or latest.get("summary") or "")[:600],
                    "sources": latest.get("sources", []),
                },
                "remaining_iterations": max(0, remaining),
                "issues": issues[-6:],
            }
            try:
                eval_chat.set_system(_AGENT_EVAL_SYSTEM)
                raw = eval_chat.structured_json(
                    user_message=json.dumps(payload, ensure_ascii=False),
                    json_schema=_AGENT_EVAL_SCHEMA,
                    stream=False,
                )
                return raw if isinstance(raw, dict) else json.loads(str(raw))
            except Exception as exc:
                log_message(f"[complex_search_agent] evaluation failed: {exc}", "WARNING")
                return {"should_stop": False, "reason": "evaluation_error", "meets_success": False, "confidence": 0.0}

        _generate_plan(max_iterations)

        while iterations < max_iterations:
            if not plan_queue:
                _generate_plan(max_iterations - iterations)
                if not plan_queue:
                    break

            step = plan_queue.pop(0)
            action = str(step.get("action") or "").lower()
            step_id = str(step.get("id") or f"step-{len(history)+1}")
            description = str(step.get("description") or "")

            if step_id in executed_ids:
                continue
            executed_ids.add(step_id)

            if action == "synthesize":
                stop_reason = description or "synthesize"
                success = True
                evaluation_snapshot = {
                    "should_stop": True,
                    "meets_success": True,
                    "reason": stop_reason,
                    "confidence": 0.6,
                }
                break

            entry: Dict[str, Any] = {
                "step_id": step_id,
                "action": action,
                "description": description,
                "plan_step": step,
                "issues": issues[-6:],
            }

            if action == "search":
                iterations += 1
                query = step.get("query") or objective
                focus = step.get("focus") or ""
                requested_max = step.get("max_results")
                try:
                    limit = int(requested_max) if requested_max else max_results
                except Exception:
                    limit = max_results
                limit = max(1, min(int(limit), max_results))

                desired_deep = bool(step.get("deep") if step.get("deep") is not None else deep_scrape)
                log_message(f"[complex_search_agent] search '{query}' (max_results={limit}, deep={desired_deep})")

                attempts = 0
                results: List[Dict[str, Any]] = []
                current_deep = desired_deep
                while attempts < 2:
                    attempts += 1
                    try:
                        results = Tools.search_internet(
                            topic=str(query),
                            num_results=limit,
                            wait_sec=1,
                            deep_scrape=current_deep,
                            summarize=True,
                            bs4_verbose=False,
                            headless=headless,
                            extract=True,
                            stream_aux=False,
                        )
                    except Exception as exc:
                        error_msg = f"browser_search_error attempt={attempts}: {exc}"
                        log_message(f"[complex_search_agent] {error_msg}", "ERROR")
                        issues.append(error_msg)
                        results = []

                    if results:
                        break

                    issue_note = f"browser_search_empty attempt={attempts} query={query!r} deep={current_deep}"
                    log_message(f"[complex_search_agent] {issue_note}", "WARNING")
                    issues.append(issue_note)

                    if not current_deep:
                        current_deep = True
                        continue
                    break

                if not results:
                    fallback = Tools._duckduckgo_fallback(query, limit=limit)
                    if fallback:
                        issues.append("duckduckgo_fallback_used")
                        results = fallback
                    else:
                        issues.append("duckduckgo_fallback_failed")

                digest: List[Dict[str, Any]] = []
                findings: List[str] = []
                for res in results[: min(len(results), 4)]:
                    url = res.get("url") or ""
                    summary = (res.get("extracted") or res.get("aux_summary") or res.get("snippet") or res.get("content") or "")
                    summary = summary.replace("\n", " ").strip()
                    if summary:
                        findings.append(f"- {summary[:280]}")
                    digest.append({
                        "title": res.get("title"),
                        "url": url,
                        "summary": summary[:600],
                    })
                    if url and url not in sources:
                        sources[url] = {
                            "url": url,
                            "title": res.get("title") or url,
                            "summary": summary[:400],
                        }

                entry.update(
                    {
                        "query": query,
                        "focus": focus,
                        "results": digest,
                        "summary": digest[0]["summary"] if digest else description,
                        "key_findings": "\n".join(findings)[:900],
                        "sources": [d.get("url") for d in digest if d.get("url")],
                    }
                )
                if not digest:
                    entry.setdefault("error", "no_results_collected")

            elif action == "visit":
                iterations += 1
                url = str(step.get("url") or "").strip()
                if not url:
                    entry.update({"error": "missing_url"})
                else:
                    extraction_goal = step.get("extraction_goal") or step.get("notes") or description
                    page = Tools.fetch_webpage(
                        url,
                        topic=objective,
                        extraction_goal=extraction_goal,
                        summarize=True,
                        max_chars=20000,
                    )
                    entry.update(
                        {
                            "url": url,
                            "summary": page.get("summary") or "",
                            "key_findings": (page.get("summary") or "")[:900],
                            "title": page.get("title"),
                            "content_preview": page.get("content_preview"),
                            "sources": [url],
                        }
                    )
                    if url not in sources:
                        sources[url] = {
                            "url": url,
                            "title": page.get("title") or url,
                            "summary": (page.get("summary") or "")[:400],
                        }

            else:
                entry.update({"error": f"unsupported_action:{action}"})

            history.append(entry)

            if action not in {"search", "visit"}:
                continue

            evaluation = _evaluate(entry, max_iterations - iterations)
            evaluation_snapshot = evaluation
            if evaluation.get("should_stop"):
                success = bool(evaluation.get("meets_success"))
                stop_reason = str(evaluation.get("reason") or "stop_condition_met")
                break

        if not stop_reason:
            if success:
                stop_reason = "completed"
            elif iterations >= max_iterations:
                stop_reason = "max_iterations"
            elif stop_hint:
                stop_reason = stop_hint
            elif history:
                stop_reason = "plan_exhausted"
            else:
                stop_reason = "no_actions_executed"

        source_list: List[Dict[str, Any]] = []
        for idx, (url, info) in enumerate(sources.items(), start=1):
            if not url:
                continue
            source_list.append(
                {
                    "id": idx,
                    "url": url,
                    "title": info.get("title") or url,
                    "summary": info.get("summary") or "",
                }
            )

        evidence = []
        for item in history[-_MAX_AGENT_HISTORY:]:
            evidence.append(
                {
                    "step_id": item.get("step_id"),
                    "action": item.get("action"),
                    "summary": (item.get("key_findings") or item.get("summary") or "")[:800],
                    "sources": item.get("sources", []),
                }
            )

        summary_payload = {
            "objective": objective,
            "success_criteria": success_criteria,
            "stop_reason": stop_reason,
            "sources": source_list,
            "evidence": evidence,
            "issues": issues[-6:],
        }

        final_summary = ""
        if source_list:
            summary_chat.set_system(_AGENT_SUMMARY_SYSTEM)
            try:
                final_summary = summary_chat.raw_chat(
                    json.dumps(summary_payload, ensure_ascii=False),
                    stream=False,
                ) or ""
            except Exception as exc:
                msg = f"summary_failed: {exc}"
                log_message(f"[complex_search_agent] {msg}", "WARNING")
                issues.append(msg)
                final_summary = ""
        else:
            issues.append("no_sources_collected")
            final_summary = "No sources were collected. Issues encountered:\n" + "\n".join(f"- {msg}" for msg in issues[-6:])
            success = False

        return {
            "objective": objective,
            "success": bool(success),
            "reason": stop_reason,
            "summary": final_summary.strip(),
            "sources": source_list,
            "iterations": iterations,
            "history": history,
            "evaluation": evaluation_snapshot,
            "stop_hint": stop_hint,
            "issues": issues,
        }



    # ===================== NEW: Filesystem & system utilities =====================

    @staticmethod
    def find_file(filename, search_path="."):
        log_message(f"Searching for file: {filename} in path: {search_path}", "PROCESS")
        for root, dirs, files in os.walk(search_path):
            if filename in files:
                log_message(f"File found in directory: {root}", "SUCCESS")
                return root
        log_message("File not found.", "WARNING")
        return None

    @staticmethod
    def get_current_location():
        """
        Resolve the users current location from a json object that also gets IP address and internet provider.
        """
        try:
            log_message("Retrieving current location based on IP.", "PROCESS")
            response = requests.get("http://ip-api.com/json", timeout=5)
            if response.status_code == 200:
                log_message("Current location retrieved.", "SUCCESS")
                return response.json()
            else:
                log_message("Error retrieving location: HTTP " + str(response.status_code), "ERROR")
                return {"error": f"HTTP error {response.status_code}"}
        except Exception as e:
            log_message("Error retrieving location: " + str(e), "ERROR")
            return {"error": str(e)}

    @staticmethod
    def get_system_utilization():
        try:
            import psutil  # type: ignore
        except Exception:
            return {"error": "psutil not installed; add 'psutil' to your dependencies."}
        utilization = {
            "cpu_usage": psutil.cpu_percent(interval=1),
            "memory_usage": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent
        }
        log_message("System utilization retrieved.", "DEBUG")
        return utilization

    @staticmethod
    def get_cwd() -> str:
        """
       Process current working directory.

        Returns:
            str: Result of os.getcwd().

        Example:
            print("Running from", Tools.get_cwd())
        """
        return os.getcwd()

    @staticmethod
    def find_files(pattern: str,
                   path: str = WORKSPACE_DIR) -> str:
        """
       Recursive file search (glob).

        Args:
            pattern (str, required): Unix glob (e.g. "*.md").
            path    (str, optional): Directory root for the walk.

        Returns:
            str (JSON array): Each element = {"file": "<name>", "dir": "<abs dir>"}.

        Example:
            matches_json = Tools.find_files("*.py")
        """
        matches = []
        for root, _, files in os.walk(path):
            for fname in files:
                if fnmatch.fnmatch(fname, pattern):
                    matches.append({"file": fname, "dir": root})
        return json.dumps(matches)

    @staticmethod
    def list_dir(path: str = WORKSPACE_DIR) -> str:
        """
       Non-recursive listing (JSON string).

        Args:
            path (str, optional): Directory to list.

        Returns:
            str (JSON array | JSON object{"error":…})

        Example:
            Tools.list_dir("data")
        """
        try:
            return json.dumps(os.listdir(path))
        except Exception as e:
            return json.dumps({"error": str(e)})

    @staticmethod
    def list_files(path: str = WORKSPACE_DIR,
                   pattern: str = "*") -> list:
        """
       Pythonic wrapper around `find_files()`.

        Args:
            path    (str, optional): Search root.
            pattern (str, optional): Glob (default "*").

        Returns:
            list[dict]: Same objects as `find_files`, but decoded.

        Example:
            py_files = Tools.list_files("src", "*.py")
        """
        return json.loads(Tools.find_files(pattern, path))

    # ────────────────────────────────────────
    #  File-reading / writing helpers
    # ────────────────────────────────────────

    @staticmethod
    def read_files(path: str, *filenames: str) -> dict:
        """
       Read multiple files in one call.

        Args:
            path       (str, required): Directory containing the files.
            *filenames (str, required): One or more filenames.

        Returns:
            dict[str,str]: {filename: content | error string}.

        Example:
            texts = Tools.read_files("logs", "out.txt", "err.txt")
        """
        out = {}
        for fn in filenames:
            out[fn] = Tools.read_file(fn, path)
        return out

    @staticmethod
    def read_file(filepath: str,
                  base_dir: str = WORKSPACE_DIR) -> str:
        """
       Read a single text file.

        Args:
            filepath (str, required): Absolute path **or** path relative to `base_dir`.
            base_dir (str | None, optional): If provided, `filepath` is resolved
                under it.

        Returns:
            str: File contents on success, or
                 "Error reading '<absolute_path>': <reason>".

        Example:
            body = Tools.read_file("notes/poem.txt", Tools.WORKSPACE_DIR)
        """
        path = os.path.join(base_dir, filepath) if base_dir else filepath
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"Error reading {path!r}: {e}"

    @staticmethod
    def write_file(filepath: str,
                   content: str,
                   base_dir: str = WORKSPACE_DIR) -> str:
        """
       Write (overwrite) a text file. **This is the canonical write helper.**

        Args:
            filepath (str, required): Destination file path (use forward slashes).
            content  (str, required): Full text to write.  
            base_dir (str | None, optional): Prefix directory; if `None`, `filepath`
                must be absolute.

        Returns:
            str: "Wrote <n> chars to '<absolute_path>'" or
                 "Error writing '<absolute_path>': <reason>".

        Example:
            Tools.write_file("docs/readme.md", "# Intro\\n")
        """
        path = os.path.join(base_dir, filepath) if base_dir else filepath
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Wrote {len(content)} chars to {path!r}"
        except Exception as e:
            return f"Error writing {path!r}: {e}"

    # ────────────────────────────────────────
    #  Rename / copy helpers
    # ────────────────────────────────────────

    @staticmethod
    def rename_file(old: str,
                    new: str,
                    base_dir: str = WORKSPACE_DIR) -> str:
        """
       Rename (move) a file **within** the workspace.

        Args:
            old (str, required): Existing relative path.
            new (str, required): New relative path.
            base_dir (str, optional): Workspace root.

        Returns:
            str: "Renamed <old> → <new>" or "Error renaming file: <reason>".

        Security:
            • Both paths are `os.path.normpath()`’d and must stay under `base_dir`.

        Example:
            Tools.rename_file("tmp.txt", "archive/tmp.txt")
        """
        safe_old = os.path.normpath(old)
        safe_new = os.path.normpath(new)
        if safe_old.startswith("..") or safe_new.startswith(".."):
            return "Error: Invalid path"
        src = os.path.join(base_dir, safe_old)
        dst = os.path.join(base_dir, safe_new)
        try:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.rename(src, dst)
            return f"Renamed {safe_old} → {safe_new}"
        except Exception as e:
            return f"Error renaming file: {e}"

    @staticmethod
    def copy_file(src: str,
                  dst: str,
                  base_dir: str = WORKSPACE_DIR) -> str:
        """
       Copy a file inside the workspace.

        Args:
            src (str, required): Existing file path.
            dst (str, required): Destination path.
            base_dir (str, optional): Workspace root.

        Returns:
            str: "Copied <src> → <dst>" or "Error copying file: <reason>".

        Example:
            Tools.copy_file("data/raw.csv", "backup/raw.csv")
        """
        safe_src = os.path.normpath(src)
        safe_dst = os.path.normpath(dst)
        if safe_src.startswith("..") or safe_dst.startswith(".."):
            return "Error: Invalid path"
        src_path = os.path.join(base_dir, safe_src)
        dst_path = os.path.join(base_dir, safe_dst)
        try:
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy(src_path, dst_path)
            return f"Copied {safe_src} → {safe_dst}"
        except Exception as e:
            return f"Error copying file: {e}"

    # ────────────────────────────────────────
    #  Metadata helpers
    # ────────────────────────────────────────

    @staticmethod
    def file_exists(filename: str,
                    base_dir: str = WORKSPACE_DIR) -> bool:
        """
       Check existence of a file.

        Args:
            filename (str, required): Relative path.
            base_dir (str, optional)

        Returns:
            bool: True if present and within workspace, False otherwise.

        Example:
            if Tools.file_exists("output/result.json"): ...
        """
        safe = os.path.normpath(filename)
        if safe.startswith(".."):
            return False
        return os.path.exists(os.path.join(base_dir, safe))

    @staticmethod
    def file_info(filename: str,
                  base_dir: str = WORKSPACE_DIR) -> dict:
        """
       Stat a file (size & mtime).

        Args:
            filename (str, required)
            base_dir (str, optional)

        Returns:
            dict:
                • "size" (int) in bytes  
                • "modified" (float) UNIX epoch seconds  
              or {"error": "<reason>"}.

        Example:
            meta = Tools.file_info("report.pdf")
        """
        safe = os.path.normpath(filename)
        if safe.startswith(".."):
            return {"error": "Invalid path"}
        path = os.path.join(base_dir, safe)
        try:
            st = os.stat(path)
            return {"size": st.st_size, "modified": st.st_mtime}
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def create_file(filename: str,
                    content: str,
                    base_dir: str = WORKSPACE_DIR) -> str:
        """
       Create (or overwrite) a new text file.

        Args:
            filename (str, required): Name **or relative path** of the file to create under
                `base_dir`. Use forward slashes for sub-dirs (“notes/poem.txt”).
            content  (str, required): Full text to write.  Pass "" for an empty file.
            base_dir (str, optional, default=WORKSPACE_DIR): Root folder for the workspace.
                Should normally be left as default.

        Returns:
            str: "Created file: <absolute_path>" on success, otherwise
                 "Error creating file '<absolute_path>': <reason>".

        Errors:
            • Intermediate directories that cannot be created.
            • I/O permission problems.

        Example:
            Tools.create_file("german_poem.txt",
                              "Goldener Abend, still und rein…")
        """
        import os
        path = os.path.join(base_dir, filename)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Created file: {path}"
        except Exception as e:
            return f"Error creating file {path!r}: {e}"

    @staticmethod
    def append_file(filename: str,
                    content: str,
                    base_dir: str = WORKSPACE_DIR) -> str:
        """
       Append text to the end of a file (create if missing).

        Args:
            filename (str, required): Relative path under `base_dir`.
            content  (str, required): Text to append (no newline automatically added).
            base_dir (str, optional): Defaults to WORKSPACE_DIR.

        Returns:
            str: "Appended to file: <absolute_path>" or
                 "Error appending to file '<absolute_path>': <reason>".

        Example:
            Tools.append_file("log/run.txt", "\\nFinished at 21:03")
        """
        import os
        path = os.path.join(base_dir, filename)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(content)
            return f"Appended to file: {path}"
        except Exception as e:
            return f"Error appending to file {path!r}: {e}"

    @staticmethod
    def delete_file(filename: str,
                    base_dir: str = WORKSPACE_DIR) -> str:
        """
       Delete a file.

        Args:
            filename (str, required): Relative path under `base_dir`.
            base_dir (str, optional): Workspace root.

        Returns:
            str: "Deleted file: <absolute_path>", "File not found: <absolute_path>",
                 or "Error deleting file '<absolute_path>': <reason>".

        Example:
            Tools.delete_file("old/tmp.txt")
        """
        import os
        path = os.path.join(base_dir, filename)
        try:
            os.remove(path)
            return f"Deleted file: {path}"
        except FileNotFoundError:
            return f"File not found: {path}"
        except Exception as e:
            return f"Error deleting file {path!r}: {e}"

    @staticmethod
    def list_workspace(path: str = ".",
                       pattern: str = "**/*",
                       include_files: bool = True,
                       include_dirs: bool = False,
                       max_results: int = 400) -> List[str]:
        """
        List workspace paths matching a glob pattern.
        """
        root = Tools._resolve_workspace_path(path)
        results: List[str] = []
        for entry in sorted(root.glob(pattern)):
            if entry.is_file() and not include_files:
                continue
            if entry.is_dir() and not include_dirs:
                continue
            rel = entry.relative_to(WORKSPACE_ROOT)
            results.append(str(rel))
            if len(results) >= max_results:
                break
        return results

    @staticmethod
    def run_bash(command: str,
                 cwd: str = ".",
                 timeout: int = 20) -> Dict[str, Any]:
        """
        Execute a bash command inside the workspace and capture output.
        """
        working_dir = Tools._resolve_workspace_path(cwd)
        try:
            proc = subprocess.run(
                ["/bin/bash", "-lc", command],
                capture_output=True,
                text=True,
                cwd=str(working_dir),
                timeout=timeout,
            )
            return {
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
        except subprocess.TimeoutExpired as exc:
            return {"returncode": None, "stdout": exc.stdout or "", "stderr": f"Timeout: {exc}"}
        except Exception as exc:
            return {"returncode": None, "stdout": "", "stderr": str(exc)}

    @staticmethod
    def search_text(pattern: str,
                    path: str = ".",
                    regex: bool = True,
                    max_results: int = 200) -> List[Dict[str, Any]]:
        """
        Search for a pattern across text files in the workspace.
        """
        root = Tools._resolve_workspace_path(path)
        matcher = re.compile(pattern) if regex else None
        results: List[Dict[str, Any]] = []
        for file_path in sorted(root.rglob("*")):
            if not file_path.is_file():
                continue
            try:
                text = file_path.read_text(encoding="utf-8")
            except Exception:
                continue
            for idx, line in enumerate(text.splitlines(), start=1):
                found = matcher.search(line) if regex else (pattern in line)
                if found:
                    results.append({
                        "file": str(file_path.relative_to(WORKSPACE_ROOT)),
                        "line": idx,
                        "text": line.strip(),
                    })
                    if len(results) >= max_results:
                        return results
        return results

    @staticmethod
    def replace_lines(filepath: str,
                      start_line: int,
                      end_line: int,
                      new_content: str) -> str:
        """
        Replace a range of lines in a file with the provided text.
        """
        if start_line < 1 or end_line < start_line:
            return "Invalid line range supplied."

        path = Tools._resolve_workspace_path(filepath)
        try:
            existing = path.read_text(encoding="utf-8").splitlines(keepends=True)
        except Exception as exc:
            return f"Error reading {path}: {exc}"

        start_idx = start_line - 1
        end_idx = end_line
        if start_idx > len(existing):
            return "Start line exceeds file length."

        replacement = new_content.splitlines(keepends=True)
        existing[start_idx:end_idx] = replacement

        try:
            path.write_text("".join(existing), encoding="utf-8")
        except Exception as exc:
            return f"Error writing {path}: {exc}"

        return (f"Replaced lines {start_line}-{end_line} in "
                f"{path.relative_to(WORKSPACE_ROOT)} with {len(replacement)} new line(s).")
