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


from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from pathlib import Path
import pickle
def multi_query(name : str, question: str):

    
    template_rag = """Answer the following question 

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template_rag)

    llm = ChatOllama(model="gemma3:1b", temperature=0,base_url="http://localhost:11434")

    final_rag_chain = (
        {
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