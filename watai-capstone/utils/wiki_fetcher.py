"""
Wikipedia API fetcher for retrieving and processing Wikipedia articles.
"""
import wikipedia
from typing import List, Dict
from langchain_core.documents import Document


def fetch_wikipedia_article(topic: str, sentences: int = 3) -> str:
    """
    Fetch a Wikipedia article summary for a given topic.
    
    Args:
        topic: The topic to search for on Wikipedia
        sentences: Number of sentences to return in the summary
        
    Returns:
        Summary text from Wikipedia
    """
    try:
        wikipedia.set_lang("en")
        page = wikipedia.page(topic, auto_suggest=True)
        summary = wikipedia.summary(topic, sentences=sentences)
        return summary
    except wikipedia.exceptions.DisambiguationError as e:
        # If disambiguation, use the first option
        return fetch_wikipedia_article(e.options[0], sentences)
    except wikipedia.exceptions.PageError:
        return f"Could not find Wikipedia page for: {topic}"
    except Exception as e:
        return f"Error fetching Wikipedia article: {str(e)}"


def fetch_multiple_topics(topics: List[str], sentences: int = 5) -> List[Document]:
    """
    Fetch multiple Wikipedia topics and convert them to LangChain Documents.
    
    Args:
        topics: List of topics to fetch from Wikipedia
        sentences: Number of sentences per summary
        
    Returns:
        List of Document objects with content and metadata
    """
    documents = []
    for topic in topics:
        try:
            content = fetch_wikipedia_article(topic, sentences)
            if not content.startswith("Could not find") and not content.startswith("Error"):
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": "wikipedia",
                        "topic": topic,
                        "type": "wiki"
                    }
                )
                documents.append(doc)
        except Exception as e:
            print(f"Error processing topic {topic}: {str(e)}")
            continue
    return documents


def search_wikipedia_topics(query: str, results: int = 5) -> List[str]:
    """
    Search Wikipedia for topics related to a query.
    
    Args:
        query: Search query
        results: Number of results to return
        
    Returns:
        List of topic titles
    """
    try:
        wikipedia.set_lang("en")
        search_results = wikipedia.search(query, results=results)
        return search_results
    except Exception as e:
        print(f"Error searching Wikipedia: {str(e)}")
        return []

