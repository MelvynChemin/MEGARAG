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
    print(f"\n💾 Saving clean chunks to vector DB...")
    print(f"   📊 Received {len(chunks)} chunks")
    
    embed = get_embed(embed_model)
    
    # Accept either raw strings or Documents
    texts, metas = [], []
    empty_count = 0
    
    for i, c in enumerate(chunks):
        if isinstance(c, Document):
            content = c.page_content.strip()
            if content:  # Only add non-empty content
                texts.append(content)
                metas.append(c.metadata)
            else:
                empty_count += 1
                print(f"   ⚠️  Skipping empty document at index {i}")
        else:
            content = str(c).strip()
            if content:  # Only add non-empty content
                texts.append(content)
                metas.append({})
            else:
                empty_count += 1
                print(f"   ⚠️  Skipping empty chunk at index {i}")
    
    # Validate we have content to embed
    if not texts:
        error_msg = (
            f"❌ No valid text chunks found after filtering {empty_count} empty chunks.\n\n"
            "🔍 Possible causes:\n"
            "  1. PDF text extraction failed (scanned images without OCR)\n"
            "  2. Documents contain only images/graphics\n"
            "  3. Text normalization removed all content\n"
            "  4. Boilerplate removal was too aggressive\n\n"
            "💡 Solutions:\n"
            "  - Check if PDF has selectable text (not scanned images)\n"
            "  - Try a simple text-based PDF first\n"
            "  - Review ingestion.py's text cleaning rules\n"
            "  - Check if deduplication is removing everything\n"
        )
        raise ValueError(error_msg)
    
    print(f"   ✅ Prepared {len(texts)} valid text chunks for embedding")
    if empty_count > 0:
        print(f"   ℹ️  Filtered out {empty_count} empty chunks")
    
    # Show statistics
    avg_length = sum(len(t) for t in texts) / len(texts)
    min_length = min(len(t) for t in texts)
    max_length = max(len(t) for t in texts)
    print(f"   📏 Chunk lengths - Avg: {avg_length:.0f}, Min: {min_length}, Max: {max_length}")
    
    # Show sample of first chunk
    if texts:
        sample = texts[0][:200] if len(texts[0]) > 200 else texts[0]
        print(f"   📝 Sample chunk: {sample}...")
    
    try:
        print(f"   🔄 Creating embeddings with model: {embed_model}")
        vdb = Chroma.from_texts(
            texts=texts,
            embedding=embed,
            metadatas=metas,
            collection_name=collection_name,
            persist_directory=str(persist_dir),
        )
        vdb.persist()
        collection_count = vdb._collection.count()
        print(f"   ✅ Vector DB created successfully with {collection_count} embeddings")
        return vdb
    except Exception as e:
        print(f"   ❌ Error creating vector DB: {str(e)}")
        print(f"   ℹ️  Make sure Ollama is running and '{embed_model}' model is available")
        print(f"   💡 Run: ollama pull {embed_model}")
        raise


def run_pipeline(tenant_slug: str):
    print("=" * 70)
    print(f"🚀 Starting RAG Pipeline for tenant: {tenant_slug}")
    print("=" * 70)
    
    try:
        # Configuration
        embed_model = "nomic-embed-text" 
        paths = get_paths(tenant_slug, "chroma-store")
        upload_path = Path("uploads") / tenant_slug
        
        # Validate upload directory exists
        if not upload_path.exists():
            raise FileNotFoundError(f"❌ Upload directory not found: {upload_path}")
        
        # Check for files
        files = list(upload_path.iterdir())
        if not files:
            raise ValueError(f"❌ No files found in: {upload_path}")
        
        print(f"\n📂 Found {len(files)} file(s) in upload directory")
        total_size = 0
        for f in files:
            size = f.stat().st_size
            total_size += size
            print(f"   - {f.name} ({size:,} bytes / {size/1024:.1f} KB)")
        print(f"   📦 Total size: {total_size:,} bytes / {total_size/1024/1024:.2f} MB")
        
        # Step 1: Perform OCR and get clean chunks
        print("\n" + "=" * 70)
        print("Step 1: Performing OCR and extracting text...")
        print("=" * 70)
        
        try:
            clean_chunks, all_pages = perform_ocr(upload_path)
        except Exception as e:
            print(f"❌ OCR failed: {str(e)}")
            print("\n💡 Common OCR issues:")
            print("   - Missing dependencies (pypdf, pytesseract, pdf2image)")
            print("   - Invalid or corrupted PDF files")
            print("   - Permissions issues reading files")
            raise
        
        print(f"\n✅ OCR completed:")
        print(f"   - Clean chunks: {len(clean_chunks)}")
        print(f"   - Total pages: {len(all_pages)}")
        
        # Validate we got chunks
        if not clean_chunks:
            error_msg = (
                "❌ OCR produced no chunks.\n\n"
                "🔍 Debug steps:\n"
                "  1. Check if PDFs contain actual text (not just images)\n"
                "  2. Test with: PyPDFLoader('your_file.pdf').load()\n"
                "  3. Review ingestion.py cleaning/deduplication logic\n"
                "  4. Check boilerplate removal isn't too aggressive\n"
            )
            raise ValueError(error_msg)
        
        # Show sample chunk info
        if clean_chunks:
            first_chunk = clean_chunks[0]
            if isinstance(first_chunk, Document):
                content_preview = first_chunk.page_content[:150]
                print(f"   📄 First chunk preview: {content_preview}...")
                print(f"   🏷️  Metadata keys: {list(first_chunk.metadata.keys())}")
        
        # Step 2: Save to Vector DB
        print("\n" + "=" * 70)
        print("Step 2: Creating vector database from clean chunks...")
        print("=" * 70)
        
        save_clean_vector_db(
            clean_chunks,
            persist_dir=paths.clean,
            collection_name="clean_docs",
            embed_model=embed_model,
        )
        
        # Optional Step: Load to SQLite
        # load_to_sqlite(clean_chunks, db_path='chunks.db')
        
        # Step 3: Train and Save BM25
        print("\n" + "=" * 70)
        print("Step 3: Training BM25 model...")
        print("=" * 70)
        
        try:
            train_and_save_bm25(clean_chunks, output_dir=paths.bm25)
            print("✅ BM25 model trained and saved")
        except Exception as e:
            print(f"⚠️  BM25 training failed: {str(e)}")
            print("   Continuing with pipeline...")
        
        # Step 4: Perform Raptor Summarization
        print("\n" + "=" * 70)
        print("Step 4: Performing RAPTOR summarization...")
        print("=" * 70)
        
        if not all_pages:
            print("⚠️  No pages found, skipping RAPTOR")
        else:
            try:
                all_texts = perform_raptor_summarization(all_pages, n_levels=3)
                print(f"   ✅ Generated {len(all_texts)} summary texts")
                
                if all_texts:
                    embed = get_embed(embed_model)
                    vdb = Chroma.from_texts(
                        texts=all_texts,
                        embedding=embed,
                        collection_name="raptor_docs",
                        persist_directory=str(paths.raptor),
                    )
                    vdb.persist()
                    print("   ✅ RAPTOR summaries saved to vector DB")
                else:
                    print("   ⚠️  No summaries generated, skipping RAPTOR DB creation")
            except Exception as e:
                print(f"⚠️  RAPTOR summarization failed: {str(e)}")
                print("   Continuing with pipeline...")
        
        print("\n" + "=" * 70)
        print("✅ Pipeline completed successfully!")
        print("=" * 70)
        print(f"📍 Vector stores created in: {paths.base}")
        print(f"   - Clean chunks: {paths.clean}")
        print(f"   - BM25 index: {paths.bm25}")
        print(f"   - RAPTOR summaries: {paths.raptor}")
        print("=" * 70)
        
    except Exception as e:
        print("\n" + "=" * 70)
        print(f"❌ Pipeline failed!")
        print("=" * 70)
        print(f"Error: {str(e)}")
        print("=" * 70)
        
        import traceback
        print("\n📋 Full traceback:")
        traceback.print_exc()
        
        print("\n💡 Quick troubleshooting:")
        print("   1. Check if Ollama is running: ollama list")
        print("   2. Pull embedding model: ollama pull nomic-embed-text")
        print("   3. Verify PDF files are readable and contain text")
        print("   4. Check ingestion.py for aggressive filtering")
        
        import sys
        sys.exit(1)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python indx_pipeline.py <tenant_slug>")
        sys.exit(1)
    
    tenant_slug = sys.argv[1]
    run_pipeline(tenant_slug)