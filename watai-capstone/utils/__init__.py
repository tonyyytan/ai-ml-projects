"""
Utility modules for the Wikipedia Q&A System.
"""
from .wiki_fetcher import fetch_wikipedia_article, fetch_multiple_topics, search_wikipedia_topics

__all__ = [
    "fetch_wikipedia_article",
    "fetch_multiple_topics",
    "search_wikipedia_topics",
]

