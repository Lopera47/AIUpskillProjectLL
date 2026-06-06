"""Main entry point for news fetcher."""

import asyncio
import sys

from src.orchestrator import FetchOrchestrator
from src.fetchers.hackernews_fetcher import HackerNewsFetcher  # ➕ ADD
from src.fetchers.rss_fetcher import RSSFetcher  # ➕ ADD
from src.fetchers.github_trending_fetcher import GitHubTrendingFetcher  # ➕ ADD
from src.transformers.article_transformer import ArticleTransformer
from src.storage.markdown_storage import MarkdownStorage


async def main():
    """Main function."""
    print("=" * 60)
    print("  AI Agent Onboarding - News Fetcher")
    print("  Milestone 2: SOLID Refactoring")
    print("=" * 60)

    try:
        # Create dependencies
        transformer = ArticleTransformer()
        storage = MarkdownStorage()
        
        # ➕ ADD: Create fetchers list
        fetchers = [
            HackerNewsFetcher(transformer, storage),
            RSSFetcher("https://hnrss.org/frontpage", transformer, storage),
            GitHubTrendingFetcher(transformer, storage),
        ]
        
        # ✅ CHANGE: Pass fetchers to orchestrator
        orchestrator = FetchOrchestrator(fetchers, storage, transformer)
        articles = await orchestrator.fetch_all()

        print("\n" + "=" * 60)
        print(f"✅ Success! Fetched {len(articles)} articles total")
        print(f"📁 Saved to: data/articles/all_articles.md")
        print("=" * 60)

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)