# Pipeline on call
from pathlib import Path
# OCR PART
from back_indx.ocr import perform_ocr, save_vector_db, load_to_sqlite
# BM25 PART
from back_indx.bmbm25s import train_and_save_bm25
# RAPTOR PART
from back_indx.raptor import perform_raptor_summarization
# Other imports
from typing import Sequence, Union
from dataclasses import dataclass
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma


# ---------- Paths helper ----------
@dataclass
class TenantPaths:
    base: Path
    clean: Path
    bm25: Path
    raptor: Path


def get_paths(tenant_slug: str, base_dir: str = "chroma-store") -> TenantPaths:
    base = Path(base_dir) / tenant_slug
    paths = TenantPaths(
        base=base,
        clean=base / "clean",
        bm25=base / "bm25",
        raptor=base / "raptor",
    )
    for p in (paths.clean, paths.bm25, paths.raptor):
        p.mkdir(parents=True, exist_ok=True)
    return paths


def get_embed(model_name: str = "nomic-embed-text"):
    # swap this to HuggingFaceEmbeddings(...) if you prefer
    return OllamaEmbeddings(model=model_name)



# ---------- Save CLEAN chunks to Chroma ----------
def save_clean_vector_db(
    chunks: Sequence[Union[str, Document]],
    persist_dir: Path,
    collection_name: str = "clean_docs",
    embed_model: str = "nomic-embed-text",
):
    embed = get_embed(embed_model)
    # Accept either raw strings or Documents
    texts, metas = [], []
    for c in chunks:
        if isinstance(c, Document):
            texts.append(c.page_content)
            metas.append(c.metadata)
        else:
            texts.append(str(c))
            metas.append({})
    vdb = Chroma.from_texts(
        texts=texts,
        embedding=embed,
        metadatas=metas,
        collection_name=collection_name,
        persist_directory=str(persist_dir),
    )
    vdb.persist()
    return vdb




def run_pipeline(tenant_slug: str):
    # Configuration
    embed_model = "nomic-embed-text" 
    paths = get_paths(tenant_slug, "chroma-store")
    # Step 1: Perform OCR and get clean chunks
    clean_chunks, all_pages = perform_ocr(Path("uploads" + "/" + tenant_slug))
    
    # Step 2: Save to Vector DB
    # mettre les clean chunks dans la vector db
    save_clean_vector_db(
        clean_chunks,
        persist_dir=paths.clean,
        collection_name="clean_docs",
        embed_model="nomic-embed-text",
    ) # replace with just embeding clean chunks (so text and metadata) in chroma vector db
    
    # Optional Step Load to SQLite
    # load_to_sqlite(clean_chunks, db_path='chunks.db')
    
    # Step 3: Train and Save BM25
    train_and_save_bm25(clean_chunks,output_dir=paths.bm25)
    
    # Step 4: Perform Raptor Summarization
    all_texts = perform_raptor_summarization(all_pages)
    embed = get_embed(embed_model)
    vdb = Chroma.from_texts(
        texts=all_texts,
        embedding=embed,
        collection_name="raptor_docs",
        persist_directory=str(paths.raptor),
    )
    vdb.persist()
    print("✅ Pipeline completed successfully.")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python indx_pipeline.py <tenant_slug>")
        sys.exit(1)
    
    tenant_slug = sys.argv[1]
    run_pipeline(tenant_slug)