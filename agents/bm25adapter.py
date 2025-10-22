# --- BM25s adapter retriever (wraps your pickled bm25s model) ---
from typing import List, Any, Optional
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import Field


class BM25sAdapterRetriever(BaseRetriever):
    """BM25s retriever adapter for LangChain."""
    
    bm25: Any = Field(description="BM25s model instance")
    corpus: List[str] = Field(description="List of raw texts (same order as bm25 index)")
    metas: Optional[List[dict]] = Field(default=None, description="Optional list of metadata aligned to corpus")
    k: int = Field(default=8, description="Number of documents to retrieve")

    class Config:
        arbitrary_types_allowed = True  # Allows the 'Any' type for bm25

    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """Retrieve relevant documents using BM25s."""
        # bm25s API variants:
        # - scores, indices = bm25.retrieve(query, k=k)
        # - or returns list of (index, score)
        res = self.bm25.retrieve(query, k=self.k)
        
        if isinstance(res, tuple) and len(res) == 2:
            scores, indices = res
            # Handle numpy arrays or lists
            if hasattr(indices, 'flatten'):
                indices = indices.flatten().tolist()
            if hasattr(scores, 'flatten'):
                scores = scores.flatten().tolist()
        else:
            # assume list[tuple[int,float]]
            indices = [r[0] for r in res]
            scores = [r[1] for r in res]

        docs: List[Document] = []
        for rank, (idx, score) in enumerate(zip(indices, scores), start=1):
            # Skip invalid indices
            if idx >= len(self.corpus):
                continue
                
            meta = {}
            if self.metas and idx < len(self.metas) and isinstance(self.metas[idx], dict):
                meta.update(self.metas[idx])
            meta.update({"bm25_score": float(score), "rank": rank, "retriever": "bm25s"})
            docs.append(Document(page_content=self.corpus[idx], metadata=meta))
        
        return docs