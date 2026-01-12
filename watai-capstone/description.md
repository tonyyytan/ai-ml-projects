# Technical Description: RAG + Memory Q&A System

## Overview

This project implements a sophisticated **Retrieval-Augmented Generation (RAG)** system with persistent conversational memory, designed to answer questions about Wikipedia content, mathematics, and science while maintaining context across sessions and enabling historical conversation search.

## Architecture

The system follows a **hybrid RAG architecture** combining:
- **Vector Search**: Semantic similarity search over knowledge embeddings
- **Conversational Memory**: Session-based chat history management
- **Historical Search**: Cross-session conversation retrieval
- **Dynamic Knowledge Expansion**: Runtime Wikipedia article addition

### System Flow

```
User Query → History Detection → Historical Search (if applicable)
    ↓
Contextual Query Rewriting (History-Aware)
    ↓
Vector Search (MongoDB Atlas) → Top-k Document Retrieval
    ↓
Document Combination (Stuff Chain)
    ↓
LLM Generation (with context + history)
    ↓
Response + History Persistence (MongoDB)
```

## Core Technologies

### Frameworks & Libraries
- **LangChain**: Orchestration framework for LLM applications
  - `langchain-core`: Core abstractions (prompts, runnables, documents)
  - `langchain-classic`: Legacy chain implementations
  - `langchain-openai`: OpenAI integrations (embeddings, chat models)
  - `langchain-mongodb`: MongoDB vector store and chat history

### Infrastructure
- **MongoDB Atlas**: Cloud-hosted database with vector search capabilities
  - Vector search index (1536 dimensions, cosine similarity)
  - Chat history collection (session-based storage)
  - Knowledge base collection (document embeddings)

### AI Models
- **OpenAI Text Embeddings**: `text-embedding-3-small` (1536 dimensions)
- **OpenAI Chat Model**: `gpt-3.5-turbo` (configurable via environment)

### APIs & Data Sources
- **Wikipedia API**: Real-time article fetching via `wikipedia` Python library
- **OpenAI API**: Embeddings and chat completion

### Python Libraries
- `pymongo`: MongoDB driver
- `python-dotenv`: Environment variable management
- `wikipedia`: Wikipedia content retrieval

## Key Technical Components

### 1. Vector Store (`MongoDBAtlasVectorSearch`)

**Purpose**: Semantic search engine for knowledge base

**Implementation**:
- Uses MongoDB Atlas Vector Search index
- 1536-dimensional embeddings (OpenAI `text-embedding-3-small`)
- Cosine similarity for relevance scoring
- Top-k retrieval (k=3 by default)

**Technical Details**:
```python
vector_store = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=embeddings,
    index_name=VECTOR_INDEX_NAME,
    relevance_score_fn="cosine",
)
```

### 2. History-Aware Retriever

**Purpose**: Contextualizes user queries using conversation history

**Technical Implementation**:
- Two-stage retrieval process
- **Stage 1**: LLM rewrites query using chat history
- **Stage 2**: Rewritten query performs vector search

**Prompt Engineering**:
- System prompt instructs LLM to reformulate queries without answering
- Uses `MessagesPlaceholder` to inject chat history dynamically

### 3. RAG Chain Architecture

**Components**:
1. **History-Aware Retriever**: `create_history_aware_retriever()`
   - Rewrites queries with context
   - Performs vector search

2. **Question-Answer Chain**: `create_stuff_documents_chain()`
   - Combines retrieved documents
   - Uses "stuff" strategy (all context in single prompt)

3. **Retrieval Chain**: `create_retrieval_chain()`
   - Chains retriever + QA chain
   - Handles document retrieval and generation

4. **Conversational RAG Chain**: `RunnableWithMessageHistory`
   - Wraps RAG chain with message history
   - Manages session-based memory

### 4. Conversational Memory System

**Storage**: MongoDB Atlas (`MongoDBChatMessageHistory`)

**Features**:
- **Session-based isolation**: Each session maintains separate history
- **Persistent storage**: Survives application restarts
- **Automatic management**: LangChain handles save/load operations
- **Message serialization**: Converts to/from LangChain message types

**Implementation**:
```python
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    return MongoDBChatMessageHistory(
        MONGO_URI, 
        session_id, 
        database_name=DATABASE_NAME, 
        collection_name=CHAT_HISTORY_COLLECTION
    )
```

### 5. Historical Conversation Search

**Purpose**: Cross-session conversation retrieval

**Algorithm**:
1. **Keyword Extraction**: Filters query for meaningful keywords (>3 chars, excludes stop words)
2. **Session Discovery**: Retrieves all session IDs from MongoDB
3. **Message Matching**: Searches all sessions for keyword matches
4. **Context Window**: Includes 2 messages before/after matches
5. **Deduplication**: Removes duplicate messages while preserving order
6. **Context Formatting**: Structures snippets for LLM consumption

**Technical Features**:
- Cross-session search capability
- Context windowing for better relevance
- Error handling for inaccessible sessions
- Configurable result limits

### 6. Wikipedia Integration (`utils/wiki_fetcher.py`)

**Functions**:
- `fetch_wikipedia_article()`: Single article retrieval with disambiguation handling
- `fetch_multiple_topics()`: Batch processing with error handling
- `search_wikipedia_topics()`: Topic search functionality

**Features**:
- Automatic disambiguation resolution
- Configurable summary length
- LangChain Document conversion
- Metadata tagging (source, topic, type)

## RAG Implementation Details

### Retrieval Strategy

**Method**: Similarity Search with History-Aware Rewriting

**Parameters**:
- Search type: `similarity`
- Top-k documents: 3 (`k=3`)
- Similarity metric: Cosine similarity
- Embedding dimensions: 1536

### Generation Strategy

**Method**: Stuff Documents Chain

**Process**:
1. All retrieved documents concatenated
2. Context injected into system prompt
3. Single LLM call with full context
4. Chat history included via `MessagesPlaceholder`

**Prompt Structure**:
```
System: [Instructions + {context}]
Chat History: [Previous messages]
Human: [Current question]
```

### Temperature & Model Settings

- **Temperature**: 0 (deterministic responses)
- **Model**: GPT-3.5-turbo (configurable)
- **Max tokens**: Model default

## Memory Management

### Session Management

**Session ID Strategy**: 
- Default: `"user_123"` (hardcoded in main)
- Configurable per user/conversation
- Used for history isolation

### History Persistence

**Storage Format**: MongoDB documents
- Structure: `{session_id, messages: [...]}`
- Message types: `HumanMessage`, `AIMessage`
- Automatic serialization/deserialization

### History Integration

**Current Session**: 
- Automatically loaded by `RunnableWithMessageHistory`
- Included in every RAG call
- Limited by session context window

**Historical Conversations**:
- Searched on-demand for history-related queries
- Prepend to prompt when detected
- Cross-session retrieval capability

## Vector Search Configuration

### MongoDB Atlas Vector Index

**Index Configuration** (JSON):
```json
{
  "fields": [
    {
      "numDimensions": 1536,
      "path": "embedding",
      "similarity": "cosine",
      "type": "vector"
    }
  ]
}
```

**Key Settings**:
- Dimensions: 1536 (OpenAI embedding size)
- Similarity: Cosine (normalized dot product)
- Index type: Vector search
- Path: `embedding` field

### Document Structure

**Knowledge Base Documents**:
```python
{
    "page_content": "Article text...",
    "metadata": {
        "source": "wikipedia",
        "topic": "Machine Learning",
        "type": "wiki"
    },
    "embedding": [1536-dimensional vector]
}
```

## API Integrations

### OpenAI API

**Endpoints Used**:
1. **Embeddings API**: Text to vector conversion
   - Model: `text-embedding-3-small`
   - Input: Text strings
   - Output: 1536-dimensional vectors

2. **Chat Completions API**: LLM inference
   - Model: `gpt-3.5-turbo`
   - Input: Prompt + history + context
   - Output: Generated text

### Wikipedia API

**Library**: `wikipedia` (Python wrapper)

**Functions**:
- `wikipedia.summary()`: Article summaries
- `wikipedia.page()`: Full page content
- `wikipedia.search()`: Topic search
- `wikipedia.set_lang()`: Language configuration

**Error Handling**:
- DisambiguationError: Auto-selects first option
- PageError: Returns error message
- Network errors: Graceful degradation

## Data Flow

### Question Processing Pipeline

1. **Input**: User question string
2. **History Detection**: Keyword matching for history queries
3. **Historical Search** (if applicable):
   - Extract keywords
   - Search all sessions
   - Retrieve relevant snippets
4. **Query Enhancement**: Prepend historical context if found
5. **History-Aware Rewriting**: LLM reformulates query using session history
6. **Vector Search**: Retrieve top-k documents
7. **Document Combination**: Concatenate retrieved context
8. **LLM Generation**: Generate answer with context + history
9. **Response**: Return answer
10. **Persistence**: Save Q&A pair to MongoDB

### Knowledge Base Population

1. **Initial Seed**: Default documents + Wikipedia articles
2. **Dynamic Addition**: `add_wikipedia_article()` function
3. **Embedding Generation**: OpenAI embeddings API
4. **Vector Storage**: MongoDB Atlas with vector index
5. **Metadata Tagging**: Source, topic, type information

## Technical Features

### 1. Multi-Source Knowledge

- **Wikipedia**: Dynamic article fetching
- **Custom Knowledge**: User-provided documents
- **Historical Conversations**: Past discussion context

### 2. Contextual Understanding

- **Current Session History**: Automatic context injection
- **Historical Context**: On-demand cross-session search
- **Query Rewriting**: History-aware query enhancement

### 3. Scalability Features

- **Cloud Database**: MongoDB Atlas (managed service)
- **Vector Indexing**: Optimized similarity search
- **Efficient Embeddings**: Small model (1536 dims) for speed
- **Caching**: Reused embeddings for stored documents

### 4. Error Resilience

- **API Error Handling**: Graceful degradation
- **Wikipedia Errors**: Disambiguation resolution
- **MongoDB Errors**: Fallback session discovery
- **Validation**: Input validation and error messages

### 5. Developer Experience

- **Environment Variables**: `.env` configuration
- **Modular Design**: Separated utilities (`utils/wiki_fetcher.py`)
- **Type Hints**: Function signatures
- **Documentation**: Docstrings and comments
- **Interactive CLI**: User-friendly console interface

## Configuration

### Environment Variables

All configuration via `.env` file:

```env
OPENAI_API_KEY=sk-...
MONGODB_URI=mongodb+srv://...
DATABASE_NAME=ai_db
KNOWLEDGE_BASE_COLLECTION=knowledge_base
CHAT_HISTORY_COLLECTION=chat_history
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_LLM_MODEL=gpt-3.5-turbo
VECTOR_INDEX_NAME=default
```

### Configurable Parameters

- **Retrieval**: `k` (number of documents), search type
- **Embeddings**: Model, dimensions
- **LLM**: Model, temperature
- **History**: Session ID, collection names
- **Wikipedia**: Summary length, batch size

## Performance Characteristics

### Embedding Generation
- **Model**: `text-embedding-3-small` (fast, cost-effective)
- **Dimensions**: 1536 (balanced performance/quality)
- **Batch Processing**: Supported for multiple documents

### Vector Search
- **Index**: MongoDB Atlas Vector Search (optimized)
- **Query Time**: O(log n) with index
- **Result Limit**: Top-k (configurable)

### LLM Inference
- **Model**: GPT-3.5-turbo (fast, cost-effective)
- **Temperature**: 0 (deterministic, faster)
- **Context Window**: Limited by model (4096 tokens for GPT-3.5)

## Security Considerations

- **API Keys**: Environment variables (not hardcoded)
- **MongoDB**: Connection string with authentication
- **Input Validation**: Query sanitization
- **Error Messages**: No sensitive data leakage

## Extensibility

### Easy Extensions

1. **Additional Data Sources**: Extend `utils/wiki_fetcher.py`
2. **Custom Retrievers**: Implement different search strategies
3. **Multiple LLMs**: Swap OpenAI for other providers
4. **Advanced Memory**: Implement semantic memory search
5. **Web Interface**: Add Flask/FastAPI wrapper
6. **Multi-user Support**: Session management enhancements

## Technical Strengths

1. **Production-Ready Architecture**: MongoDB Atlas, managed services
2. **Advanced RAG**: History-aware retrieval + generation
3. **Persistent Memory**: Cross-session conversation retention
4. **Dynamic Knowledge**: Runtime Wikipedia integration
5. **Error Resilience**: Comprehensive error handling
6. **Modular Design**: Separated concerns, reusable components
7. **Scalable**: Cloud infrastructure, vector indexing
8. **Configurable**: Environment-based configuration

## Limitations & Considerations

1. **Session-based History**: Requires session ID management
2. **Keyword Search**: Historical search uses simple keyword matching
3. **Context Window**: Limited by LLM token limits
4. **Cost**: API calls for embeddings and LLM inference
5. **Wikipedia Rate Limits**: API throttling possible
6. **MongoDB Index**: Requires manual vector index creation

## Future Enhancements

1. **Semantic History Search**: Embedding-based conversation search
2. **Multi-modal Support**: Image/document understanding
3. **Streaming Responses**: Real-time token generation
4. **Advanced Memory**: Long-term memory compression
5. **Query Optimization**: Better query rewriting strategies
6. **Analytics**: Conversation analytics and insights

