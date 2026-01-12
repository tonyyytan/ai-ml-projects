"""
RAG + Memory System using LangChain, MongoDB Atlas, OpenAI, and Wikipedia API.

This system:
- Retrieves Wikipedia articles and stores them as vector embeddings in MongoDB
- Answers questions using RAG (Retrieval-Augmented Generation)
- Maintains conversation history in MongoDB
- Can answer questions about Wikipedia content and mathematical/scientific topics
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
from utils.wiki_fetcher import fetch_multiple_topics, fetch_wikipedia_article

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
def add_wikipedia_article(topic: str, sentences: int = 7):
    """
    Add a Wikipedia article to the knowledge base.
    
    Args:
        topic: Wikipedia topic to fetch and add
        sentences: Number of sentences to fetch
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
context to answer the question. If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise. For mathematical questions, 
provide clear explanations and step-by-step reasoning when helpful.

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

# 6. ADD MEMORY (Chat History Management)
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    return MongoDBChatMessageHistory(
        MONGO_URI, 
        session_id, 
        database_name=DATABASE_NAME, 
        collection_name=CHAT_HISTORY_COLLECTION
    )

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# 7. MAIN INTERACTION FUNCTION
def ask_question(question: str, session_id: str = "default_session"):
    """
    Ask a question to the RAG system.
    
    Args:
        question: The question to ask
        session_id: Session ID for maintaining conversation history
        
    Returns:
        The answer from the AI
    """
    response = conversational_rag_chain.invoke(
        {"input": question},
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
    
    print("You can now ask questions! Type 'quit' or 'exit' to exit.")
    print("Type 'add <topic>' to add a Wikipedia article to the knowledge base.")
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
                print("  ask <question> - Ask a question")
                print("  add <topic> - Add a Wikipedia article (e.g., 'add Quantum Mechanics')")
                print("  quit/exit - Exit the program")
                print()
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

