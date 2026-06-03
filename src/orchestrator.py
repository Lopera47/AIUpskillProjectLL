"""Orchestrate multiple news fetchers."""

import asyncio
from typing import List

from src.fetchers.hackernews_fetcher import HackerNewsFetcher
from src.fetchers.rss_fetcher import RSSFetcher
from src.fetchers.github_trending_fetcher import GitHubTrendingFetcher  # ➕ ADD
from src.models.article import Article
from src.storage.markdown_storage import MarkdownStorage
from src.transformers.article_transformer import ArticleTransformer


class FetchOrchestrator:
    """
    Orchestrates fetching from multiple sources.

    Coordinates HackerNews, RSS, GitHub, and other fetchers.
    """

    def __init__(self, transformer, storage):
        """Initialize orchestrator with injected dependencies."""
        self.storage = storage
        self.fetchers = [
            HackerNewsFetcher(transformer, storage),
            RSSFetcher("https://hnrss.org/frontpage", transformer, storage),
            GitHubTrendingFetcher(transformer, storage),
        ]

    async def fetch_all(self) -> List[Article]:
        """
        Fetch from all sources concurrently.

        Returns:
            Combined list of all articles
        """
        print("\n🚀 Starting fetch from all sources...")
        print(f"   Sources: {len(self.fetchers)}")

        # Create tasks for all fetchers (all have fetch_and_save now)
        tasks = [fetcher.fetch_and_save() for fetcher in self.fetchers]

        # Fetch all concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine articles
        all_articles = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"⚠️  Fetcher {i} failed: {result}")
            else:
                print(f"✅ Fetcher {i}: {len(result)} articles")
                all_articles.extend(result)

        print(
            f"\n🎉 Total: {len(all_articles)} articles from {len(self.fetchers)} sources"
        )
        return all_articles


# Test it
async def main():
    """Test orchestrator."""
    
    transformer = ArticleTransformer()
    storage = MarkdownStorage()
    
    orchestrator = FetchOrchestrator(transformer, storage)
    articles = await orchestrator.fetch_all()

    print(f"\n📊 Sample articles:")
    for article in articles[:5]:
        print(f"  [{article.source}] {article.title[:60]}...")


if __name__ == "__main__":
    asyncio.run(main())