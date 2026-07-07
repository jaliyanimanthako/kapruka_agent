"""
Catalog-focused Playwright crawler for Kapruka.

The crawler is intentionally biased toward product discovery:
- keeps the crawl inside Kapruka catalog/product paths
- expands "see more" style listings before parsing links
- prioritizes product detail pages over generic site pages
"""

from __future__ import annotations

import asyncio
import json
import re
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from markdownify import markdownify as md
from playwright.async_api import async_playwright

DEFAULT_CRAWL_MAX_PAGES = 20
DEFAULT_LISTING_EXPANSION_ROUNDS = 8
DEFAULT_CATALOG_BURST = 1
DEFAULT_MAX_CATALOG_PAGES = 30
DEFAULT_MAX_CATALOG_LINKS_PER_PAGE = 25
PRODUCT_LINK_SELECTOR = 'a[href*="/buyonline/"][href*="/kid/"]'
LOAD_MORE_SELECTORS = [
    "text=See More Products",
    "text=Load More",
    "text=See More",
    'button:has-text("See More Products")',
    'button:has-text("Load More")',
    'a:has-text("See More Products")',
]


class KaprukaWebCrawler:
    """Playwright-based crawler for Kapruka product discovery."""

    def __init__(
        self,
        base_url: str,
        max_depth: int,
        exclude_patterns: List[str],
        max_pages: Optional[int] = None,
        max_catalog_pages: Optional[int] = None,
        max_product_pages: Optional[int] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.max_depth = max_depth
        self.exclude_patterns = exclude_patterns
        self.max_pages = DEFAULT_CRAWL_MAX_PAGES if max_pages is None else max_pages
        self.max_listing_expansions = DEFAULT_LISTING_EXPANSION_ROUNDS
        self.catalog_burst = DEFAULT_CATALOG_BURST
        if max_catalog_pages is None and max_product_pages is None:
            self.max_catalog_pages = min(DEFAULT_MAX_CATALOG_PAGES, self.max_pages)
            self.max_product_pages = max(0, self.max_pages - self.max_catalog_pages)
        else:
            self.max_catalog_pages = DEFAULT_MAX_CATALOG_PAGES if max_catalog_pages is None else max_catalog_pages
            remaining_pages = max(0, self.max_pages - self.max_catalog_pages)
            self.max_product_pages = remaining_pages if max_product_pages is None else max_product_pages
        self._base_parsed = urlparse(self.base_url)
        self._base_host = self._normalize_host(self._base_parsed.hostname)
        self.visited: Set[str] = set()
        self.documents: List[Dict[str, Any]] = []
        self.errors: List[Dict[str, Any]] = []
        self.crawled_product_ids: Set[str] = set()

    @staticmethod
    def _normalize_host(host: Optional[str]) -> Optional[str]:
        if not host:
            return None

        host = host.lower()
        if host.startswith("www."):
            return host[4:]
        return host

    @staticmethod
    def _clean_text(value: Optional[str]) -> str:
        if not value:
            return ""
        return re.sub(r"\s+", " ", value).strip()

    def _normalize_url(self, url: str) -> str:
        url = url.split("#", 1)[0].strip()
        if not url:
            return ""

        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return url

        normalized = parsed._replace(fragment="")
        normalized_url = normalized.geturl()

        if normalized.path == "/" and normalized_url.endswith("/"):
            return normalized_url[:-1]

        return normalized_url

    def _is_same_site(self, url: str) -> bool:
        parsed_url = urlparse(url)
        if parsed_url.scheme not in {"http", "https"}:
            return False

        return self._normalize_host(parsed_url.hostname) == self._base_host

    def _is_product_page(self, url: str) -> bool:
        return bool(re.search(r"/buyonline/[^/]+/kid/[^/]+", url))

    def _extract_product_id(self, url: str) -> Optional[str]:
        match = re.search(r"/kid/([^/?#]+)", self._normalize_url(url), re.IGNORECASE)
        if not match:
            return None
        return match.group(1).lower()

    def _is_catalog_page(self, url: str) -> bool:
        normalized = self._normalize_url(url)
        if normalized in {self.base_url, f"{self.base_url}/"}:
            return True

        path = urlparse(normalized).path.lower()
        return (
            path in {"", "/"}
            or path.startswith("/online/")
            or path.startswith("/shops/")
        )

    def _is_low_value_catalog_page(self, url: str) -> bool:
        path = urlparse(self._normalize_url(url)).path.lower()
        low_value_prefixes = (
            "/shops/blog",
            "/shops/blog_posts.jsp",
            "/shops/popular_searches.jsp",
            "/shops/events_home.jsp",
            "/online/services/",
        )
        return path.startswith(low_value_prefixes)

    def _is_catalog_related(self, url: str) -> bool:
        return self._is_product_page(url) or self._is_catalog_page(url)

    def _reset_run_state(self) -> None:
        self.visited.clear()
        self.documents.clear()
        self.errors.clear()
        self.crawled_product_ids.clear()

    def _seed_urls(self, start_urls: List[str]) -> List[str]:
        seeds: List[str] = []
        for url in start_urls:
            normalized = self._normalize_url(url)
            if normalized and normalized not in seeds:
                seeds.append(normalized)

        known_catalog_seeds = [
            self.base_url,
            urljoin(f"{self.base_url}/", "shops/deliveryCatalogCompact_wide.jsp"),
            urljoin(f"{self.base_url}/", "online/newadditions"),
        ]
        for url in known_catalog_seeds:
            normalized = self._normalize_url(url)
            if normalized not in seeds:
                seeds.append(normalized)

        return seeds

    def should_crawl(self, url: str) -> bool:
        url = self._normalize_url(url)
        if not url or url in self.visited:
            return False

        if not self._is_same_site(url):
            return False

        for pattern in self.exclude_patterns:
            if pattern in url:
                return False

        if re.search(r"\.(jpg|jpeg|png|gif|pdf|zip|exe)$", urlparse(url).path, re.I):
            return False

        if self._is_low_value_catalog_page(url):
            return False

        if self._is_product_page(url):
            product_id = self._extract_product_id(url)
            if product_id and product_id in self.crawled_product_ids:
                return False

        return self._is_catalog_related(url)

    def _extract_internal_links(self, soup: Any, url: str) -> List[str]:
        links: List[str] = []
        seen: Set[str] = set()

        for anchor in soup.find_all("a", href=True):
            href = anchor.get("href", "").strip()
            if not href or href.startswith(("mailto:", "tel:", "javascript:")):
                continue

            absolute_url = urljoin(url, href)
            normalized = self._normalize_url(absolute_url)
            if not normalized or normalized == self._normalize_url(url):
                continue

            if self._is_same_site(normalized) and normalized not in seen:
                seen.add(normalized)
                links.append(normalized)

        return links

    def _find_primary_content(self, soup: BeautifulSoup) -> Any:
        return (
            soup.find("main")
            or soup.find("div", {"id": "root"})
            or soup.find("article")
            or soup.find("section", {"class": re.compile("product|listing|grid|items", re.I)})
            or soup.find("div", {"class": re.compile("content|main|container|product|listing|grid", re.I)})
            or soup.body
        )

    def _ordered_unique(self, values: List[str]) -> List[str]:
        seen: Set[str] = set()
        ordered: List[str] = []
        for value in values:
            if value not in seen:
                seen.add(value)
                ordered.append(value)
        return ordered

    def _product_link_roots(self, soup: BeautifulSoup) -> List[Any]:
        roots: List[Any] = []
        primary_content = self._find_primary_content(soup)
        if primary_content is not None:
            roots.append(primary_content)

        for element in soup.find_all(
            ["section", "div", "ul"],
            attrs={"class": re.compile("product|related|similar|recommend|grid|list|item", re.I)},
        ):
            roots.append(element)

        for element in soup.find_all(
            ["section", "div"],
            attrs={"id": re.compile("product|related|similar|recommend", re.I)},
        ):
            roots.append(element)

        return roots[:20]

    def extract_product_links(self, soup: BeautifulSoup, url: str) -> List[str]:
        links: List[str] = []
        seen_product_ids: Set[str] = set()

        for root in self._product_link_roots(soup):
            for link in self._extract_internal_links(root, url):
                if not self._is_product_page(link):
                    continue

                product_id = self._extract_product_id(link)
                if product_id and product_id not in seen_product_ids:
                    seen_product_ids.add(product_id)
                    links.append(link)

        return self._ordered_unique(links)

    def extract_catalog_links(self, soup: BeautifulSoup, url: str) -> List[str]:
        primary_content = self._find_primary_content(soup)
        if primary_content is None:
            return []

        current_normalized_url = self._normalize_url(url)
        current_path = urlparse(self._normalize_url(url)).path.lower()
        category_prefix = ""
        current_parts = [part for part in current_path.split("/") if part]
        if len(current_parts) >= 2 and current_parts[0] == "online":
            category_prefix = f"/online/{current_parts[1]}"

        scored_links: List[tuple[int, str]] = []
        for link in self._extract_internal_links(primary_content, url):
            if not self._is_catalog_page(link) or self._is_product_page(link) or self._is_low_value_catalog_page(link):
                continue

            link_path = urlparse(link).path.lower()
            score = 0
            if category_prefix and link_path.startswith(category_prefix):
                score += 4
            if "/price/" in link_path:
                score += 3
            if urlparse(link).query:
                score += 2
            if link_path.startswith("/online/"):
                score += 1
            if self._normalize_url(link) == current_normalized_url:
                continue

            scored_links.append((score, link))

        scored_links.sort(key=lambda item: (-item[0], item[1]))
        ordered_links = self._ordered_unique([link for _, link in scored_links])
        return ordered_links[:DEFAULT_MAX_CATALOG_LINKS_PER_PAGE]

    def _enqueue_catalog_links(
        self,
        queue: deque[tuple[str, int]],
        queued_urls: Set[str],
        current_depth: int,
        links: List[str],
    ) -> int:
        links_added = 0
        for link in links:
            if link not in self.visited and link not in queued_urls:
                queue.append((link, current_depth + 1))
                queued_urls.add(link)
                links_added += 1
        return links_added

    def _enqueue_product_links(
        self,
        queue: deque[tuple[str, int]],
        queued_product_ids: Set[str],
        current_depth: int,
        links: List[str],
    ) -> int:
        links_added = 0
        for link in links:
            product_id = self._extract_product_id(link)
            if not product_id:
                continue
            if product_id in self.crawled_product_ids or product_id in queued_product_ids:
                continue

            queue.append((link, current_depth + 1))
            queued_product_ids.add(product_id)
            links_added += 1

        return links_added

    def extract_product_data(self, soup: BeautifulSoup, url: str) -> Optional[Dict[str, Any]]:
        try:
            name = None
            price_text = None
            description = None
            availability = None

            name_selectors = [
                "h1",
                "h2",
                '[data-testid="product-title"]',
                ".product-name",
                ".product-title",
            ]
            for selector in name_selectors:
                elem = soup.select_one(selector)
                text = self._clean_text(elem.get_text()) if elem else ""
                if text:
                    name = text
                    break

            price_selectors = [
                ".price",
                '[data-testid="product-price"]',
                ".product-price",
                ".amount",
                ".currency",
                '[class*="price"]',
            ]
            for selector in price_selectors:
                elem = soup.select_one(selector)
                text = self._clean_text(elem.get_text()) if elem else ""
                if text:
                    price_text = text
                    break

            if not price_text:
                body_text = soup.get_text(" ", strip=True)
                price_match = re.search(
                    r"(?:US\$|Rs\.?|LKR)[\s]*([\d,]+(?:\.\d+)?.*?)($|[A-Z][a-z]{1,3}\$)",
                    body_text,
                    re.IGNORECASE,
                )
                if price_match:
                    price_text = self._clean_text(price_match.group(0))

            description_selectors = [
                ".product-description",
                '[data-testid="product-description"]',
                ".description",
                ".details",
            ]
            for selector in description_selectors:
                elem = soup.select_one(selector)
                text = self._clean_text(elem.get_text()) if elem else ""
                if text:
                    description = text[:500]
                    break

            if not description:
                meta_desc = soup.find("meta", attrs={"name": "description"})
                if meta_desc and meta_desc.get("content"):
                    description = self._clean_text(meta_desc["content"])[:500]

            availability_selectors = [
                ".availability",
                '[data-testid="availability"]',
                ".stock-status",
                ".in-stock",
                ".sold-out",
            ]
            for selector in availability_selectors:
                elem = soup.select_one(selector)
                text = self._clean_text(elem.get_text()) if elem else ""
                if text:
                    availability = text
                    break

            if not availability:
                body_text = soup.get_text(" ", strip=True).lower()
                if any(token in body_text for token in ("sold out", "out of stock", "unavailable")):
                    availability = "Out of Stock"
                elif any(token in body_text for token in ("in stock", "available")):
                    availability = "In Stock"
                else:
                    availability = "Unknown"

            return {
                "name": name or "Unknown Product",
                "price": price_text or "Price not found",
                "description": description or "No description available",
                "availability": availability or "Unknown",
                "url": url,
            }
        except Exception:
            return None

    def extract_content(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        for element in soup(["script", "style", "nav", "footer", "aside", "noscript", "iframe"]):
            element.decompose()

        title = soup.title.string if soup.title else url.split("/")[-1]
        title = self._clean_text(title) or "Untitled"
        headings = [
            self._clean_text(heading.get_text())
            for heading in soup.find_all(["h1", "h2", "h3", "h4"])
            if self._clean_text(heading.get_text())
        ]
        links = self._extract_internal_links(soup, url)

        main_content = (
            soup.find("div", {"id": "root"})
            or soup.find("main")
            or soup.find("article")
            or soup.find("div", {"class": re.compile("content|main|container", re.I)})
            or soup.body
        )

        if main_content:
            content_md = md(str(main_content), heading_style="ATX")
        else:
            content_md = md(str(soup), heading_style="ATX")

        content_md = re.sub(r"You need to enable JavaScript.*?\.", "", content_md, flags=re.IGNORECASE)
        content_md = re.sub(r"\n{3,}", "\n\n", content_md).strip()

        return {
            "title": title,
            "headings": headings,
            "content": content_md,
            "links": links,
        }

    async def _count_product_links(self, page: Any) -> int:
        return await page.locator(PRODUCT_LINK_SELECTOR).count()

    async def _progressive_scroll(self, page: Any, rounds: int = 4) -> None:
        previous_height = -1
        for _ in range(rounds):
            current_height = await page.evaluate("document.body.scrollHeight")
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(1200)
            if current_height == previous_height:
                break
            previous_height = current_height

    async def _expand_listing_page(self, page: Any) -> None:
        previous_link_count = await self._count_product_links(page)

        for _ in range(self.max_listing_expansions):
            await self._progressive_scroll(page, rounds=2)
            clicked = False

            for selector in LOAD_MORE_SELECTORS:
                locator = page.locator(selector).first
                try:
                    if await locator.count() == 0 or not await locator.is_visible():
                        continue

                    await locator.scroll_into_view_if_needed()
                    await locator.click(timeout=3000)
                    await page.wait_for_timeout(1800)
                    clicked = True
                    break
                except Exception:
                    continue

            current_link_count = await self._count_product_links(page)
            if current_link_count <= previous_link_count and not clicked:
                break

            previous_link_count = max(previous_link_count, current_link_count)

    async def _load_page_html(self, page: Any, url: str) -> str:
        await page.goto(url, wait_until="domcontentloaded", timeout=60000)
        await page.wait_for_selector("body", timeout=10000)

        try:
            await page.wait_for_load_state("networkidle", timeout=7000)
        except Exception:
            await page.wait_for_timeout(2500)

        await self._progressive_scroll(page)
        if self._is_catalog_page(url):
            await self._expand_listing_page(page)

        return await page.content()

    async def crawl_async(self, start_urls: List[str], request_delay: float = 2.0) -> List[Dict[str, Any]]:
        self._reset_run_state()
        catalog_queue = deque((url, 0) for url in self._seed_urls(start_urls))
        product_queue: deque[tuple[str, int]] = deque()
        queued_catalog_urls = {url for url, _ in catalog_queue}
        queued_product_ids: Set[str] = set()
        pages_crawled = 0
        catalog_pages_crawled = 0
        product_pages_crawled = 0
        catalog_streak = 0

        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=True)
            page = await browser.new_page()
            page.set_default_timeout(30000)

            try:
                while catalog_queue or product_queue:
                    if self.max_pages is not None and pages_crawled >= self.max_pages:
                        print(f"   🛑 Reached max pages limit ({self.max_pages})")
                        break

                    can_use_catalog = bool(catalog_queue) and catalog_pages_crawled < self.max_catalog_pages
                    can_use_product = bool(product_queue) and product_pages_crawled < self.max_product_pages

                    if not can_use_catalog and not can_use_product:
                        print(
                            "   🛑 Reached configured catalog/product crawl budgets "
                            f"({catalog_pages_crawled}/{self.max_catalog_pages} catalog, "
                            f"{product_pages_crawled}/{self.max_product_pages} product)"
                        )
                        break

                    if can_use_product and (not can_use_catalog or catalog_streak >= self.catalog_burst):
                        active_queue = product_queue
                        catalog_streak = 0
                        queue_kind = "product"
                    else:
                        active_queue = catalog_queue if catalog_queue else product_queue
                        if active_queue is catalog_queue:
                            catalog_streak += 1
                            queue_kind = "catalog"
                        else:
                            queue_kind = "product"

                    url, depth = active_queue.popleft()
                    if queue_kind == "catalog":
                        queued_catalog_urls.discard(url)
                    else:
                        product_id = self._extract_product_id(url)
                        if product_id:
                            queued_product_ids.discard(product_id)

                    if depth > self.max_depth or not self.should_crawl(url):
                        continue

                    try:
                        print(f"🔍 [{depth}] {url}")
                        self.visited.add(url)
                        pages_crawled += 1
                        is_product_page = self._is_product_page(url)
                        if is_product_page:
                            product_pages_crawled += 1
                            product_id = self._extract_product_id(url)
                            if product_id:
                                self.crawled_product_ids.add(product_id)
                        else:
                            catalog_pages_crawled += 1

                        html = await self._load_page_html(page, url)
                        soup = BeautifulSoup(html, "html.parser")
                        product_links = self.extract_product_links(soup, url)
                        catalog_links = self.extract_catalog_links(soup, url)

                        if is_product_page:
                            doc_data = self.extract_product_data(soup, url)
                            if doc_data:
                                self.documents.append(doc_data)
                                print(
                                    f"   ✅ Product saved: {doc_data.get('name', 'Unknown')} - "
                                    f"{doc_data.get('price', 'N/A')}"
                                )
                            else:
                                print("   ⚠️  Failed to extract product data")
                        else:
                            content_data = self.extract_content(soup, url)
                            print(
                                f"   ✅ Listing parsed ({len(content_data['content'])} chars, "
                                f"{len(content_data['links'])} links found)"
                            )

                        if depth < self.max_depth:
                            catalog_links_added = 0
                            product_links_added = 0
                            if catalog_pages_crawled < self.max_catalog_pages:
                                catalog_links_added = self._enqueue_catalog_links(
                                    queue=catalog_queue,
                                    queued_urls=queued_catalog_urls,
                                    current_depth=depth,
                                    links=catalog_links,
                                )
                            if product_pages_crawled < self.max_product_pages:
                                product_links_added = self._enqueue_product_links(
                                    queue=product_queue,
                                    queued_product_ids=queued_product_ids,
                                    current_depth=depth,
                                    links=product_links,
                                )
                            links_added = catalog_links_added + product_links_added
                            if links_added > 0:
                                print(
                                    f"   📎 Added {catalog_links_added} catalog and "
                                    f"{product_links_added} product URLs to queue "
                                    f"(depth {depth + 1})"
                                )

                        print(
                            f"   📊 Progress: {len(self.documents)} products saved, "
                            f"{len(self.visited)} visited, "
                            f"{catalog_pages_crawled}/{self.max_catalog_pages} catalog pages, "
                            f"{product_pages_crawled}/{self.max_product_pages} product pages, "
                            f"{len(catalog_queue)} catalog queued, {len(product_queue)} product queued"
                        )
                        await asyncio.sleep(request_delay)
                    except Exception as exc:
                        self.errors.append(
                            {
                                "url": url,
                                "depth_level": depth,
                                "stage": "crawl",
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                        print(f"   ❌ Error: {type(exc).__name__}: {str(exc)[:100]}")
                        continue
            finally:
                await browser.close()

        return self.documents

    def export_to_json(self, output_file: str = "catalog.json") -> str:
        output_path = Path(output_file)
        products = [doc for doc in self.documents if "price" in doc and "name" in doc]

        catalog_data = {
            "metadata": {
                "total_products": len(products),
                "total_pages_crawled": len(self.visited),
                "errors_encountered": len(self.errors),
                "export_timestamp": datetime.now(timezone.utc).isoformat(),
            },
            "products": products,
            "errors": self.errors,
        }

        with open(output_path, "w", encoding="utf-8") as file_obj:
            json.dump(catalog_data, file_obj, indent=2, ensure_ascii=False)

        print(f"✅ Catalog exported to {output_path}")
        return str(output_path)

    def crawl(self, start_urls: List[str], request_delay: float = 2.0) -> List[Dict[str, Any]]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.crawl_async(start_urls, request_delay))

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, self.crawl_async(start_urls, request_delay))
            return future.result()


__all__ = ["KaprukaWebCrawler"]
