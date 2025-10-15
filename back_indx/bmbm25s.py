import bm25s
# from .ocr import perform_ocr
from pathlib import Path
import pickle
import os
# Create corpus from your chunks
# clean_chunks, all_pages = perform_ocr(Path("Uploads"))
# corpus = [ch.page_content for ch in clean_chunks]
# chunk_ids = [ch.metadata["chunk_id"] for ch in clean_chunks]

# # Train BM25
# retriever = bm25s.BM25(corpus=corpus)
# retriever.index(bm25s.tokenize(corpus))

# # Save the objects (optional: pickle to disk if you want persistence)
# import pickle, os
# os.makedirs("data", exist_ok=True)
# with open("data/bm25_acme.pkl", "wb") as f:
#     pickle.dump((retriever, chunk_ids, corpus), f)


def train_and_save_bm25(clean_chunks, output_dir) -> None:
    """
    Trains a BM25 model on the provided clean chunks and saves it to disk.
    
    The BM25 model is indexed with the content of the chunks, allowing for efficient retrieval based on relevance.
    The trained model, along with chunk IDs and corpus, is saved as a pickle file for future use.
    
    Args:
        clean_chunks (List[Document]): List of LangChain Document objects with metadata.
    """
    # Create corpus from your chunks
    corpus = [ch.page_content for ch in clean_chunks]
    chunk_ids = [ch.metadata["chunk_id"] for ch in clean_chunks]

    # Train BM25
    retriever = bm25s.BM25(corpus=corpus)
    retriever.index(bm25s.tokenize(corpus))

    # Save the objects (optional: pickle to disk if you want persistence)
    os.makedirs("data", exist_ok=True)
    with open(f"{output_dir}/bm25_acme.pkl", "wb") as f:
        pickle.dump((retriever, chunk_ids, corpus), f)
    print(f"✅ BM25 model saved at {output_dir}/bm25_acme.pkl")
    