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
from fct import _rrf_fuse, _dedupe
load_dotenv()  # reads .env and adds vars to os.environ

# Optional (cleaner: you can set these in your shell instead)
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_ENDPOINT'] = os.getenv("LANGCHAIN_ENDPOINT")
os.environ['LANGCHAIN_TRACING_V2'] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ['LANGCHAIN_PROJECT'] = os.getenv("LANGCHAIN_PROJECT")

os.environ['USER_AGENT'] = os.getenv("USER_AGENT")

from langchain.prompts import ChatPromptTemplate


from langchain.load import dumps, loads

def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]

from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from pathlib import Path
import pickle
def multi_query(name : str, question: str):

    BASE = Path("chroma-store") / name
    emb = OllamaEmbeddings(model="nomic-embed-text") 
    clean_vs = Chroma(
    persist_directory=str(BASE / "clean"),
    embedding_function=emb,
    collection_name="clean_docs",
    )
    # path = Path(BASE / "clean")
    # print("📁 DB directory:", path.resolve())
    # print("📄 Files in dir:", [p.name for p in path.glob('*')])
    # from chromadb import PersistentClient
    # client = PersistentClient(path=str(BASE / "clean"))
    # print([ (c.name, c.count()) for c in client.list_collections() ])

    # # 2️⃣ Collection info
    # try:
    #     count = clean_vs._collection.count()  # number of vectors
    #     print(f"📊 Number of vectors stored: {count}")
    # except Exception as e:
    #     print("⚠️ Could not query Chroma collection:", e)

    # # 3️⃣ Sanity query
    # try:
    #     test_docs = clean_vs.similarity_search("test", k=1)
    #     print(f"✅ Sample query returned {len(test_docs)} docs.")
    #     if len(test_docs):
    #         print("🧩 First document snippet:", test_docs[0].page_content[:200])
    # except Exception as e:
    #     print("⚠️ Search error:", e)
    clean_ret = clean_vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 40, "lambda_mult": 0.7},
    )
    # Multi Query: Different Perspectives
    template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
    The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
    Generate multiple search queries related to: {question} \n
    Output (3 queries):"""
    prompt_decomposition = ChatPromptTemplate.from_template(template)

    from langchain_core.output_parsers import StrOutputParser
    from langchain_openai import ChatOpenAI
    llm = ChatOllama(model="gemma3:1b", temperature=0,base_url="http://localhost:11434")

    generate_queries = (
        prompt_decomposition 
        | llm
        | StrOutputParser() 
        | (lambda x: x.split("\n"))
    )

    retrieval_chain = generate_queries | clean_ret.map() | get_unique_union
    docs = retrieval_chain.invoke({"question":question})
    print(len(docs))

     # RAG promp
    template_rag = """Answer the following question based on this context:

    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template_rag)

    llm = ChatOllama(model="gemma3:1b", temperature=0,base_url="http://localhost:11434")

    final_rag_chain = (
        {"context": retrieval_chain, 
        "question": itemgetter("question")} 
        | prompt
        | llm
        | StrOutputParser()
    )


    return final_rag_chain


if __name__ == "__main__":
    mq_chain = multi_query("ai", "What is AI AGENTS?")
    response = mq_chain.invoke({"question": "What is AI AGENTS?"})
    print(response)