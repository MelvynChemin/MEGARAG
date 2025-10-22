import os
import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Loaders & Vector store
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma

# Core utilities
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# LLMs & Embeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings

from dotenv import load_dotenv
load_dotenv()  # reads .env and adds vars to os.environ

# Optional (cleaner: you can set these in your shell instead)
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_ENDPOINT'] = os.getenv("LANGCHAIN_ENDPOINT")
os.environ['LANGCHAIN_TRACING_V2'] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ['LANGCHAIN_PROJECT'] = os.getenv("LANGCHAIN_PROJECT")

os.environ['USER_AGENT'] = os.getenv("USER_AGENT")

# === INDEXING ===

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

# Split into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Embed using your local Ollama model
embedder = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")

# Store embeddings locally
vectorstore = Chroma.from_documents(documents=splits, embedding=embedder)
retriever = vectorstore.as_retriever()

# === RETRIEVAL & GENERATION ===

# Pull an RAG-ready prompt template from LangChain Hub
prompt = hub.pull("rlm/rag-prompt")

# Choose your model
# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
llm = ChatOllama(model="gemma3:1b", temperature=0, base_url="http://localhost:11434")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

response = rag_chain.invoke("What is AI AGENTS?")
print(response)
