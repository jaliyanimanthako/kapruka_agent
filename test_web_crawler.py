"""Smoke test for the Kapruka web crawler.

Run this from the repository root:

    python test_web_crawler.py
"""

from __future__ import annotations

import sys
import unittest
from collections import deque
from pathlib import Path

from bs4 import BeautifulSoup


REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from web_crawler import KaprukaWebCrawler  # noqa: E402


class KaprukaWebCrawlerUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.crawler = KaprukaWebCrawler(
            base_url="https://www.kapruka.com/",
            max_depth=2,
            exclude_patterns=["/login", "/admin", "/search"],
            max_pages=50,
            max_catalog_pages=10,
            max_product_pages=40,
        )

    def test_should_crawl_only_catalog_related_urls(self) -> None:
        self.assertTrue(
            self.crawler.should_crawl("https://www.kapruka.com/buyonline/sample-product/kid/item001")
        )
        self.assertTrue(self.crawler.should_crawl("https://www.kapruka.com/online/newadditions?page=2"))
        self.assertFalse(self.crawler.should_crawl("https://www.kapruka.com/contactus"))
        self.assertFalse(self.crawler.should_crawl("https://www.kapruka.com/assets/logo.png"))
        self.assertFalse(self.crawler.should_crawl("https://www.kapruka.com/shops/blog_posts.jsp"))

    def test_extract_catalog_and_product_links(self) -> None:
        html = """
        <html>
          <body>
            <main>
            <a href="/buyonline/sample-product/kid/item001">Product</a>
            <a href="/buyonline/sample-product-two/kid/item001">Duplicate Product Id</a>
            <a href="/online/newadditions?page=2">Next Page</a>
            <a href="/contactus">Contact</a>
            <a href="javascript:void(0)">Ignore</a>
            </main>
          </body>
        </html>
        """
        soup = BeautifulSoup(html, "html.parser")

        product_links = self.crawler.extract_product_links(soup, "https://www.kapruka.com/online/newadditions")
        catalog_links = self.crawler.extract_catalog_links(soup, "https://www.kapruka.com/online/newadditions")

        self.assertEqual(
            product_links,
            ["https://www.kapruka.com/buyonline/sample-product/kid/item001"],
        )
        self.assertEqual(
            catalog_links,
            ["https://www.kapruka.com/online/newadditions?page=2"],
        )

    def test_extract_product_data_prefers_meta_description_and_stock_status(self) -> None:
        html = """
        <html>
          <head>
            <meta name="description" content="A compact mixer grinder for daily kitchen use.">
          </head>
          <body>
            <h1>Compact Mixer Grinder</h1>
            <div class="price">US$99.00</div>
            <div class="sold-out">Sold Out</div>
          </body>
        </html>
        """
        soup = BeautifulSoup(html, "html.parser")
        product = self.crawler.extract_product_data(
            soup,
            "https://www.kapruka.com/buyonline/compact-mixer-grinder/kid/item001",
        )

        self.assertIsNotNone(product)
        self.assertEqual(product["name"], "Compact Mixer Grinder")
        self.assertEqual(product["price"], "US$99.00")
        self.assertEqual(product["description"], "A compact mixer grinder for daily kitchen use.")
        self.assertEqual(product["availability"], "Sold Out")

    def test_enqueue_catalog_links_skips_duplicates_and_visited(self) -> None:
        queue = deque([("https://www.kapruka.com/online/newadditions?page=2", 1)])
        queued_urls = {"https://www.kapruka.com/online/newadditions?page=2"}
        self.crawler.visited.add("https://www.kapruka.com/online/newadditions?page=1")

        links_added = self.crawler._enqueue_catalog_links(
            queue=queue,
            queued_urls=queued_urls,
            current_depth=1,
            links=[
                "https://www.kapruka.com/online/newadditions?page=1",
                "https://www.kapruka.com/online/newadditions?page=3",
                "https://www.kapruka.com/online/newadditions?page=2",
            ],
        )

        self.assertEqual(links_added, 1)
        self.assertEqual(
            list(queue),
            [
                ("https://www.kapruka.com/online/newadditions?page=2", 1),
                ("https://www.kapruka.com/online/newadditions?page=3", 2),
            ],
        )

    def test_enqueue_product_links_dedupes_by_product_id(self) -> None:
        queue = deque()
        queued_product_ids = set()
        self.crawler.crawled_product_ids.add("item001")

        links_added = self.crawler._enqueue_product_links(
            queue=queue,
            queued_product_ids=queued_product_ids,
            current_depth=1,
            links=[
                "https://www.kapruka.com/buyonline/already-crawled/kid/item001",
                "https://www.kapruka.com/buyonline/sample-product-a/kid/item002",
                "https://www.kapruka.com/buyonline/sample-product-b/kid/item002",
                "https://www.kapruka.com/buyonline/sample-product-c/kid/item003",
            ],
        )

        self.assertEqual(links_added, 2)
        self.assertEqual(
            list(queue),
            [
                ("https://www.kapruka.com/buyonline/sample-product-a/kid/item002", 2),
                ("https://www.kapruka.com/buyonline/sample-product-c/kid/item003", 2),
            ],
        )


def main() -> int:
    crawler = KaprukaWebCrawler(
        base_url="https://www.kapruka.com/",
        max_depth=3,
        exclude_patterns=["/login", "/admin", "/search"],
        max_pages=200,
        max_catalog_pages=30,
        max_product_pages=170,
    )

    start_urls = ["https://www.kapruka.com"]

    try:
        documents = crawler.crawl(start_urls=start_urls, request_delay=1.0)
    except Exception as exc:
        print(f"Crawl failed: {type(exc).__name__}: {exc}")
        return 1

    print(f"\n📊 Crawl Summary:")
    print(f"   Documents saved: {len(documents)}")
    print(f"   Pages visited: {len(crawler.visited)}")
    print(f"   Errors captured: {len(crawler.errors)}")

    if documents:
        first_document = documents[0]
        print(f"\n🛍️ First document extracted:")
        print(f"   Title/Name: {first_document.get('title', first_document.get('name', 'Untitled'))}")
        print(f"   URL: {first_document.get('url', '')}")
        
        # Check if product data was extracted
        if 'price' in first_document:
            print(f"   Price: {first_document.get('price', 'N/A')}")
            print(f"   Availability: {first_document.get('availability', 'N/A')}")
            print(f"   Description: {first_document.get('description', 'N/A')[:100]}...")

    # Export to catalog.json
    try:
        output_path = crawler.export_to_json("catalog.json")
        print(f"\n✅ Catalog exported to: {output_path}")
    except Exception as exc:
        print(f"❌ Export failed: {type(exc).__name__}: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
