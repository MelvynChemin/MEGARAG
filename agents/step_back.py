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
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

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
    examples = [
    {
        "input": "Could the members of The Police perform lawful arrests?",
        "output": "what can the members of The Police do?",
    },
    {
        "input": "Jan Sindel’s was born in what country?",
        "output": "what is Jan Sindel’s personal history?",
    },
    ]
    example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:""",
            ),
            # Few shot examples
            few_shot_prompt,
            # New question
            ("user", "{question}"),
        ]
    )
    llm = ChatOllama(model="gemma3:1b", temperature=0,base_url="http://localhost:11434")

    generate_queries_step_back = prompt | llm | StrOutputParser()
    question = "What is task decomposition for LLM agents?"
    generate_queries_step_back.invoke({"question": question})
     # RAG promp

    llm = ChatOllama(model="gemma3:1b", temperature=0,base_url="http://localhost:11434")

    # Response prompt 
    response_prompt_template = """You are an expert of world knowledge. I am going to ask you a question. Your response should be comprehensive and not contradicted with the following context if they are relevant. Otherwise, ignore them if they are not relevant.

    # {normal_context}
    # {step_back_context}

    # Original Question: {question}
    # Answer:"""
    response_prompt = ChatPromptTemplate.from_template(response_prompt_template)

    chain = (
        {
            # Retrieve context using the normal question
            "normal_context": RunnableLambda(lambda x: x["question"]) | clean_ret,
            # Retrieve context using the step-back question
            "step_back_context": generate_queries_step_back | clean_ret,
            # Pass on the question
            "question": lambda x: x["question"],
        }
        | response_prompt
        | llm
        | StrOutputParser()
    )

    


    return chain


if __name__ == "__main__":
    mq_chain = multi_query("ai", "What is AI AGENTS?")
    response = mq_chain.invoke({"question": "What is AI AGENTS?"})
    print(response)