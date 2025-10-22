from pathlib import Path
import pickle
from collections import defaultdict
from typing import List, Sequence
from operator import itemgetter

from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def _rrf_fuse(ranked_lists: Sequence[Sequence[Document]], k: int = 60, c: int = 60) -> List[Document]:
    scores, order, id2doc = defaultdict(float), [], {}
    for docs in ranked_lists:
        for rank, d in enumerate(docs[:k], start=1):
            did = d.metadata.get("id") or (d.metadata.get("source"), d.metadata.get("chunk"))
            if did not in id2doc:
                id2doc[did] = d
            if did not in order:
                order.append(did)
            scores[did] += 1.0 / (c + rank)
    fused_ids = sorted(order, key=lambda i: scores[i], reverse=True)
    return [id2doc[i] for i in fused_ids]

def _dedupe(docs: Sequence[Document]) -> List[Document]:
    seen, out = set(), []
    for d in docs:
        key = d.metadata.get("id") or (d.metadata.get("source"), d.metadata.get("chunk"))
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out