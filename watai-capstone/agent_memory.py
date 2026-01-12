"""
RAG + Memory System using LangChain, MongoDB Atlas, OpenAI, and Wikipedia API.

This system:
- Retrieves Wikipedia articles and stores them as vector embeddings in MongoDB
- Answers questions using RAG (Retrieval-Augmented Generation) with tool support
- Maintains conversation history in MongoDB
- Can answer questions about Wikipedia content and mathematical/scientific topics
- Supports multi-tooling with LangChain tools for dynamic Wikipedia access
"""
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.documents import Document
from langchain.tools import tool
from utils.wiki_fetcher import fetch_multiple_topics, fetch_wikipedia_article, search_wikipedia_topics

# Load environment variables
load_dotenv()

# 1. SETUP API KEYS & MONGO CONNECTION
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MONGO_URI = os.getenv("MONGODB_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME", "ai_db")
KNOWLEDGE_BASE_COLLECTION = os.getenv("KNOWLEDGE_BASE_COLLECTION", "knowledge_base")
CHAT_HISTORY_COLLECTION = os.getenv("CHAT_HISTORY_COLLECTION", "chat_history")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-3.5-turbo")
VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME", "default")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

client = MongoClient(MONGO_URI)
collection = client[DATABASE_NAME][KNOWLEDGE_BASE_COLLECTION]
chat_history_collection = client[DATABASE_NAME][CHAT_HISTORY_COLLECTION]

# 2. SETUP VECTOR STORE (The "Brain")
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
vector_store = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=embeddings,
    index_name=VECTOR_INDEX_NAME,
    relevance_score_fn="cosine",
)

# 3. POPULATE DATABASE (Run this once to "teach" the AI)
def seed_database(use_wikipedia: bool = True):
    """
    Seed the database with initial knowledge.
    
    Args:
        use_wikipedia: If True, fetch Wikipedia articles for common topics
    """
    if collection.count_documents({}) > 0:
        print("Database already contains data. Skipping seed.")
        return
    
    print("Seeding database...")
    docs = []
    
    # Add some default knowledge
    default_docs = [
        Document(
            page_content="Tony Tan acts as a Resume Reviewer AI.",
            metadata={"source": "user_bio", "type": "bio"}
        ),
        Document(
            page_content="Waterloo Engineering is known for its co-op program.",
            metadata={"source": "wiki", "type": "general"}
        ),
        Document(
            page_content="LangChain is a framework for developing applications powered by LLMs.",
            metadata={"source": "tech_docs", "type": "tech"}
        ),
    ]
    docs.extend(default_docs)
    
    # Optionally fetch Wikipedia articles
    if use_wikipedia:
        print("Fetching Wikipedia articles...")
        wiki_topics = ["Machine Learning", "Artificial Intelligence", "Mathematics", "Physics"]
        wiki_docs = fetch_multiple_topics(wiki_topics, sentences=7)
        docs.extend(wiki_docs)
        print(f"Fetched {len(wiki_docs)} Wikipedia articles")
    
    if docs:
        vector_store.add_documents(docs)
        print(f"Database seeded with {len(docs)} documents!")

# 4. ADD WIKIPEDIA ARTICLE TO DATABASE
@tool
def add_wikipedia_article(topic: str, sentences: int = 7):
    """
    Useful for adding a Wikipedia article to the knowledge base for future reference.
    Use this when the user wants to save information about a topic to the database or when you need to permanently store knowledge about a subject.
    Fetches a Wikipedia article and stores it as a vector embedding in MongoDB for future retrieval.
    
    Args:
        topic: Wikipedia topic to fetch and add
        sentences: Number of sentences to fetch (default: 7)
        
    Returns:
        True if successful, False otherwise
    """
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
            vector_store.add_documents([doc])
            print(f"Successfully added Wikipedia article: {topic}")
            return True
        else:
            print(f"Failed to add Wikipedia article: {topic}")
            return False
    except Exception as e:
        print(f"Error adding Wikipedia article: {str(e)}")
        return False

# 5. SETUP THE RAG CHAIN (Retrieval + Generation)
llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

# Retriever: Finds relevant info in MongoDB
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Contextualize Question Prompt: Rewrites user query based on history
contextualize_q_system_prompt = """Given a chat history and the latest user question 
which might reference context in the chat history, formulate a standalone question 
which can be understood without the chat history. Do NOT answer the question, 
just reformulate it if needed and otherwise return it as is."""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Answer Prompt: Uses retrieved docs to answer
qa_system_prompt = """You are an intelligent assistant specialized in answering questions 
about Wikipedia content, mathematics, and science. Use the following pieces of retrieved 
context to answer the question. 

The context may include:
1. Knowledge base information (Wikipedia articles, technical docs)
2. Historical conversation snippets (previous discussions with the user)

When the user asks about past conversations (e.g., "do you remember our conversation about X" 
or "what did we talk about earlier"), use the historical conversation context to recall 
and reference those discussions. Be conversational and acknowledge what was discussed previously.

If you don't know the answer, just say that you don't know. Use three sentences maximum 
and keep the answer concise. For mathematical questions, provide clear explanations and 
step-by-step reasoning when helpful.

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# 5.5. HISTORICAL CONVERSATION SEARCH
def search_historical_conversations(query: str, current_session_id: str, limit: int = 3):
    """
    Search through historical conversations in MongoDB to find relevant past discussions.
    Searches across all sessions to find conversations matching the query.
    
    Args:
        query: Search query
        current_session_id: Current session ID
        limit: Maximum number of conversation snippets to return
        
    Returns:
        String containing relevant historical conversation snippets
    """
    try:
        # Extract keywords from query (filter out common words and short words)
        query_lower = query.lower()
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "about", "remember", "conversation", "discussed", "talked"}
        keywords = [word for word in query_lower.split() if len(word) > 3 and word not in stop_words]
        
        # If no meaningful keywords, return empty
        if not keywords:
            return ""
        
        # Get all sessions from MongoDB
        try:
            all_sessions = chat_history_collection.distinct("session_id")
        except Exception:
            # If distinct fails, try to get all documents and extract session_ids
            all_docs = chat_history_collection.find({}, {"session_id": 1})
            all_sessions = list(set([doc.get("session_id") for doc in all_docs if doc.get("session_id")]))
        
        historical_contexts = []
        
        # Search through all sessions (including current one)
        for session_id in all_sessions:
            try:
                history = get_session_history(session_id)
                messages = history.messages
                
                if not messages:
                    continue
                
                # Check if any message contains keywords from the query
                relevant_messages = []
                for i, msg in enumerate(messages):
                    if hasattr(msg, 'content'):
                        content = msg.content.lower()
                        # Check if message contains any keywords
                        if any(keyword in content for keyword in keywords):
                            # Include surrounding context (2 messages before and after)
                            start_idx = max(0, i - 2)
                            end_idx = min(len(messages), i + 3)
                            context_messages = messages[start_idx:end_idx]
                            relevant_messages.extend(context_messages)
                
                # Remove duplicates while preserving order
                seen = set()
                unique_messages = []
                for msg in relevant_messages:
                    msg_id = id(msg)
                    if msg_id not in seen:
                        seen.add(msg_id)
                        unique_messages.append(msg)
                
                # If we found relevant messages, create a context snippet
                if unique_messages:
                    context_text = f"From session '{session_id}':\n"
                    # Include up to 8 messages for context
                    for msg in unique_messages[:8]:
                        if hasattr(msg, 'content'):
                            role = "Human" if "Human" in msg.__class__.__name__ else "AI"
                            context_text += f"{role}: {msg.content}\n"
                    historical_contexts.append(context_text)
                    
                    if len(historical_contexts) >= limit:
                        break
            except Exception as e:
                # Skip sessions that can't be accessed
                continue
        
        if historical_contexts:
            return "\n\n".join(historical_contexts)
        return ""
    except Exception as e:
        print(f"Error searching historical conversations: {str(e)}")
        return ""

def get_recent_conversation_summary(session_id: str, num_messages: int = 10):
    """
    Get a summary of recent conversations from current session to provide context.
    
    Args:
        session_id: Session ID to get history for
        num_messages: Number of recent messages to include
        
    Returns:
        String summary of recent conversations
    """
    try:
        history = get_session_history(session_id)
        messages = history.messages[-num_messages:] if len(history.messages) > num_messages else history.messages
        
        if not messages:
            return ""
        
        summary = "Recent conversation context:\n"
        for msg in messages:
            if hasattr(msg, 'content'):
                role = "Human" if "Human" in msg.__class__.__name__ else "AI"
                summary += f"{role}: {msg.content}\n"
        
        return summary
    except Exception as e:
        print(f"Error getting conversation summary: {str(e)}")
        return ""

# 6. ADD MEMORY (Chat History Management)
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Get or create chat message history for a session from MongoDB."""
    history = MongoDBChatMessageHistory(
        MONGO_URI, 
        session_id, 
        database_name=DATABASE_NAME, 
        collection_name=CHAT_HISTORY_COLLECTION
    )
    return history

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# Helper function to view chat history
def view_chat_history(session_id: str, limit: int = 10):
    """
    View chat history for a session.
    
    Args:
        session_id: Session ID to view history for
        limit: Maximum number of messages to display
    """
    history = get_session_history(session_id)
    messages = history.messages
    if not messages:
        print(f"No chat history found for session: {session_id}")
        return
    
    print(f"\nChat History for session '{session_id}' (last {min(limit, len(messages))} messages):")
    print("-" * 60)
    for i, msg in enumerate(messages[-limit:], 1):
        if hasattr(msg, 'content'):
            role = msg.__class__.__name__.replace('Message', '').lower()
            content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
            print(f"{i}. [{role.upper()}]: {content}")
    print("-" * 60)
    print()

# Helper function to clear chat history
def clear_chat_history(session_id: str):
    """Clear chat history for a session."""
    history = get_session_history(session_id)
    history.clear()
    print(f"Chat history cleared for session: {session_id}\n")

# 7. MAIN INTERACTION FUNCTION
def ask_question(question: str, session_id: str = "default_session"):
    """
    Ask a question to the RAG system with historical context support.
    
    Args:
        question: The question to ask
        session_id: Session ID for maintaining conversation history
        
    Returns:
        The answer from the AI
    """
    # Check if question is about past conversations
    history_keywords = ["remember", "conversation", "discussed", "talked about", "earlier", "before", "previous", "what did we", "did we talk"]
    is_history_question = any(keyword in question.lower() for keyword in history_keywords)
    
    # If asking about history, search for relevant historical conversations
    historical_context = ""
    if is_history_question:
        historical_context = search_historical_conversations(question, session_id, limit=3)
    
    # The RAG chain automatically includes current session history via RunnableWithMessageHistory
    # For history questions, we prepend historical context to help the model recall past conversations
    enhanced_question = question
    if historical_context:
        enhanced_question = f"""Historical conversation context:
{historical_context}

Current question: {question}

Please reference the historical context above when answering questions about past conversations."""
    
    response = conversational_rag_chain.invoke(
        {"input": enhanced_question},
        config={"configurable": {"session_id": session_id}},
    )
    return response["answer"]

# 8. INTERACTIVE CONSOLE INTERFACE
def main():
    """Main function to run the interactive console interface."""
    print("=" * 60)
    print("Wikipedia & Science Q&A System")
    print("Using LangChain, MongoDB, OpenAI, and Wikipedia API")
    print("=" * 60)
    print()
    
    # Seed database if empty
    print("Initializing database...")
    seed_database(use_wikipedia=True)
    print()
    
    session_id = "user_123"
    
    # Check if there's existing chat history
    history = get_session_history(session_id)
    if history.messages:
        print(f"Found {len(history.messages)} previous messages in chat history.")
        print("The bot will remember previous conversations!\n")
    else:
        print("Starting new conversation session.\n")
    
    print("You can now ask questions! Type 'quit' or 'exit' to exit.")
    print("Type 'add <topic>' to add a Wikipedia article to the knowledge base.")
    print("Type 'history' to view chat history.")
    print("Type 'clear' to clear chat history.")
    print("Type 'help' for more commands.")
    print("-" * 60)
    print()
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            
            if user_input.lower() == "help":
                print("\nCommands:")
                print("  <question> - Ask a question")
                print("  add <topic> - Add a Wikipedia article (e.g., 'add Quantum Mechanics')")
                print("  history - View chat history")
                print("  clear - Clear chat history for this session")
                print("  quit/exit - Exit the program")
                print()
                continue
            
            if user_input.lower() == "history":
                view_chat_history(session_id)
                continue
            
            if user_input.lower() == "clear":
                clear_chat_history(session_id)
                continue
            
            if user_input.lower().startswith("add "):
                topic = user_input[4:].strip()
                if topic:
                    print(f"\nAdding Wikipedia article: {topic}...")
                    add_wikipedia_article(topic)
                    print()
                else:
                    print("Please specify a topic. Example: 'add Machine Learning'")
                continue
            
            # Ask the question
            print("\nAI: ", end="", flush=True)
            answer = ask_question(user_input, session_id)
            print(answer)
            print()
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            print()

if __name__ == "__main__":
    main()

