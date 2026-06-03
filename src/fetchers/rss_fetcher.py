"""Fetch articles from RSS feeds."""

import asyncio
import re
from datetime import datetime
from typing import List

import feedparser
from dateutil import parser as date_parser

from src.fetchers.base_fetcher import BaseFetcher
from src.models.article import Article


class RSSFetcher(BaseFetcher):
    """
    Fetches articles from RSS feeds.

    Uses feedparser library for RSS parsing.
    """

    def __init__(self, feed_url: str, transformer, storage):
        """
        Initialize RSS fetcher.

        Args:
            feed_url: URL of RSS feed
            transformer: ArticleTransformer instance
            storage: MarkdownStorage instance
        """
        super().__init__(transformer, storage)
        self.feed_url = feed_url

    async def fetch_articles(self) -> List[Article]:
        """
        Fetch articles from RSS feed.

        Returns:
            List of Article objects
        """
        print(f"📰 Fetching from RSS: {self.feed_url}")

        # feedparser is sync, run in executor
        loop = asyncio.get_event_loop()
        feed = await loop.run_in_executor(
            None, feedparser.parse, self.feed_url
        )

        # Parse entries
        articles = []
        for entry in feed.entries:
            article = self._parse_entry(entry)
            if article:
                articles.append(article)

        print(f"✅ Fetched {len(articles)} RSS articles")
        return articles

    def get_source_name(self) -> str:
        """Return source name."""
        return "rss"

    def _parse_entry(self, entry) -> Article:
        """Parse RSS entry to Article."""
        try:
            # Get title
            title = entry.get("title", "No Title")

            # Get URL
            url = entry.get("link", "")
            if not url:
                return None

            # Parse date
            published_str = entry.get("published", entry.get("updated", ""))
            if published_str:
                try:
                    published_at = date_parser.parse(published_str)
                except:
                    published_at = datetime.now()
            else:
                published_at = datetime.now()

            # Get summary
            summary = entry.get("summary", entry.get("description", ""))
            # Clean HTML tags (basic)
            summary = re.sub("<.*?>", "", summary)[:200]

            return Article(
                title=title,
                url=url,
                published_at=published_at,
                source="rss",
                summary=summary,
            )
        except Exception as e:
            print(f"⚠️  Failed to parse entry: {e}")
            return None


async def test_rss():
    """Test RSS fetcher."""
    from src.transformers.article_transformer import ArticleTransformer
    from src.storage.markdown_storage import MarkdownStorage
    
    transformer = ArticleTransformer()
    storage = MarkdownStorage()
    
    # HackerNews RSS feed
    fetcher = RSSFetcher("https://hnrss.org/frontpage", transformer, storage)
    articles = await fetcher.fetch_articles()

    print(f"\n📊 Fetched {len(articles)} articles")
    for article in articles[:3]:
        print(f"  - {article.title[:50]}...")


if __name__ == "__main__":
    asyncio.run(test_rss())