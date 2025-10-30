from enum import unique
import os, re, unicodedata, html, hashlib, datetime
from pathlib import Path
from typing import List, Dict, Tuple, Set
from collections import Counter, defaultdict

from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -------------------------
# 1) Filename metadata parser (customize to your org)
# -------------------------
FILENAME_DATE_RE = re.compile(r"(20\d{2}[-_.]?(?:0[1-9]|1[0-2])[-_.]?(?:0[1-9]|[12]\d|3[01]))", re.I)
FILENAME_VERSION_RE = re.compile(r"(?:v|ver|version)[-_ ]?(\d+(?:\.\d+)*)", re.I)
FILENAME_DEPT_RE = re.compile(r"(hr|legal|finance|it|engineering|sales|marketing)", re.I)

def parse_filename_metadata(path: Path) -> Dict:
    name = path.stem
    meta = {}
    # effective_date from filename, fallback to file mtime
    m = FILENAME_DATE_RE.search(name)
    if m:
        s = re.sub(r"[-_.]", "-", m.group(1))
        try:
            meta["effective_date"] = datetime.datetime.fromisoformat(
                s if len(s) == 10 else f"{s[:4]}-{s[5:7]}-{s[8:10]}"
            ).date().isoformat()
        except Exception:
            meta["effective_date"] = datetime.date.fromtimestamp(path.stat().st_mtime).isoformat()
    else:
        meta["effective_date"] = datetime.date.fromtimestamp(path.stat().st_mtime).isoformat()

    # version
    m = FILENAME_VERSION_RE.search(name)
    meta["version"] = m.group(1) if m else "1.0"

    # dept guess
    m = FILENAME_DEPT_RE.search(name)
    meta["dept"] = m.group(1).upper() if m else "GENERAL"

    meta["source_filename"] = path.name
    meta["source_path"] = str(path.resolve())
    meta["doc_id"] = hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:12]
    return meta

# -------------------------
# 2) Text sanitizer
#    - normalize unicode
#    - strip HTML entities
#    - collapse whitespace
#    - remove page headers/footers learned per file
# -------------------------
def normalize_text(t: str) -> str:
    t = html.unescape(t)
    t = unicodedata.normalize("NFKC", t)
    # common PDF artifacts
    t = t.replace("\u00ad", "")  # soft hyphen
    # collapse whitespace
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\s+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def learn_boilerplate_lines(pages: List[str], top_k: int = 3, bottom_k: int = 3, freq_threshold: float = 0.8) -> Set[str]:
    """
    LESS AGGRESSIVE: Changed freq_threshold from 0.5 to 0.8
    Heuristic: grab top/bottom K lines of each page; if a line repeats in >= freq_threshold of pages, treat as boilerplate
    """
    # For single-page documents, don't remove any boilerplate
    if len(pages) <= 1:
        print("   ℹ️  Single page document - skipping boilerplate removal")
        return set()
    
    counter = Counter()
    total = len(pages)
    for p in pages:
        lines = [ln.strip() for ln in p.splitlines() if ln.strip()]
        if not lines:
            continue
        heads = lines[:min(top_k, len(lines))]
        tails = lines[-min(bottom_k, len(lines)):]
        for ln in (heads + tails):
            if len(ln) <= 120:  # avoid counting long paragraphs
                counter[ln] += 1
    
    boilerplate = {ln for ln, c in counter.items() if c / max(1, total) >= freq_threshold}
    
    if boilerplate:
        print(f"   ℹ️  Found {len(boilerplate)} boilerplate lines to remove")
    
    return boilerplate
def strip_boilerplate(text: str, boilerplate: Set[str]) -> str:
    if not boilerplate:
        return text
    kept = []
    for ln in text.splitlines():
        s = ln.strip()
        if s and s not in boilerplate:
            kept.append(ln)
    return "\n".join(kept)

# -------------------------
# 3) PII / secret redaction (lightweight starters; expand as needed)
# -------------------------
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)?\d{3,4}[\s.-]?\d{3,4}\b")
APIKEY_RE = re.compile(r"\b(sk-[A-Za-z0-9]{20,})\b", re.I)

def redact_pii(text: str) -> str:
    # text = EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    # text = PHONE_RE.sub("[REDACTED_PHONE]", text)
    text = APIKEY_RE.sub("[REDACTED_KEY]", text)
    return text

# -------------------------
# 4) Dedupe (exact + simple near-duplicate)
#    - exact: SHA1 of normalized text
#    - near: Jaccard on 5-gram shingles (fast, dependency-free)
#      (upgrade path: SimHash/MinHash if needed)
# -------------------------
def shingles(text: str, n: int = 5) -> Set[str]:
    tok = re.findall(r"\w+", text.lower())
    return {" ".join(tok[i:i+n]) for i in range(0, max(0, len(tok)-n+1))}

def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

class Deduper:
    def __init__(self, near_thresh: float = 0.9):
        self.exact_hashes: Set[str] = set()
        self.shingle_index: Dict[str, Set[str]] = defaultdict(set)  # key: doc_idhash -> shingles
        self.near_thresh = near_thresh

    def is_duplicate(self, text: str) -> bool:
        norm = normalize_text(text)
        if not norm:
            return True
        h = hashlib.sha1(norm.encode()).hexdigest()
        if h in self.exact_hashes:
            return True
        # near-dup check
        sh = shingles(norm, n=5)
        for other_h, other_sh in self.shingle_index.items():
            if jaccard(sh, other_sh) >= self.near_thresh:
                return True
        # not a dup → register
        self.exact_hashes.add(h)
        self.shingle_index[h] = sh
        return False

# -------------------------
# 5) PDF → clean page Documents (with metadata)
# -------------------------
def load_and_clean_pdf(pdf_path: Path, tenant: str = "default", acl: str = "internal") -> List[Document]:
    """
    Updated with better debugging
    """
    base_meta = parse_filename_metadata(pdf_path)
    base_meta.update({"tenant": tenant, "acl": acl})

    # Load raw pages
    loader = PyPDFLoader(str(pdf_path))
    raw_docs = loader.load()  # one Document per page, metadata has {"source": path, "page": int}

    # Learn boilerplate lines across pages
    pages_text = [d.page_content or "" for d in raw_docs]
    
    # DEBUG: Show raw content
    print(f"   📄 Raw page content ({len(pages_text)} pages):")
    for i, page in enumerate(pages_text):
        print(f"      Page {i+1}: {len(page)} chars")
    
    boiler = learn_boilerplate_lines(pages_text)

    cleaned_docs = []
    for d in raw_docs:
        text_original = d.page_content or ""
        
        # Step by step cleaning with debug output
        text = normalize_text(text_original)
        print(f"   After normalize: {len(text)} chars")
        
        text = strip_boilerplate(text, boiler)
        print(f"   After boilerplate removal: {len(text)} chars")
        
        text = redact_pii(text)
        print(f"   After PII redaction: {len(text)} chars")

        page_meta = dict(base_meta)
        page_meta.update({
            "source": d.metadata.get("source", str(pdf_path)),
            "page": d.metadata.get("page", None),
            "doc_id": base_meta["doc_id"],
            "chunk_type": "page",
        })

        cleaned_docs.append(Document(page_content=text, metadata=page_meta))
    
    return cleaned_docs

# -------------------------
# 6) Chunking (semantic + carry strong anchors)
# -------------------------
def chunk_docs(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150,
        separators=["\n## ", "\n# ", "\n- ", "\n• ", "\n", " ", ""],  # soft preference for headings/bullets
    )
    chunks = splitter.split_documents(docs)
    # Attach stable chunk ids + anchors
    out = []
    for i, ch in enumerate(chunks):
        m = dict(ch.metadata)
        content = ch.page_content.strip()
        # store anchors for precise citations
        m["chunk_id"] = hashlib.sha1(f'{m.get("doc_id")}:{m.get("page")}:{i}:{content[:80]}'.encode()).hexdigest()[:12]
        m["chunk_index"] = i
        out.append(Document(page_content=content, metadata=m))
    return out

from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain.schema import Document

# def semantic_then_window(docs: list[Document]) -> list[Document]:
#     out = []
#     header_splitter = MarkdownHeaderTextSplitter(
#         headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
#     )
#     window_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000, chunk_overlap=150,
#         separators=["\n## ", "\n# ", "\n- ", "\n• ", "\n", " ", ""],
#     )
#     for d in docs:
#         # Try semantic split; if it fails (not markdown), fall back to windowed
#         sections = header_splitter.split_text(d.page_content)
#         if not sections:
#             out.extend(window_splitter.split_documents([d]))
#             continue
#         # Wrap sections back into Documents, carry metadata + section titles
#         sec_docs = []
#         for s in sections:
#             meta = dict(d.metadata)
#             # concatenate detected headers for stronger anchors
#             title = " / ".join([v for k, v in s.metadata.items() if k in ("h1","h2","h3") and v])
#             meta["section_title"] = title
#             sec_docs.append(Document(page_content=s.page_content, metadata=meta))
#         out.extend(window_splitter.split_documents(sec_docs))
#     return out
def semantic_then_window(docs: list[Document]) -> list[Document]:
    pre_chunks: List[Document] = []
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
    )
    window_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150,
        separators=["\n## ", "\n# ", "\n- ", "\n• ", "\n", " ", ""],
    )
    
    for d in docs:
        # Try semantic split; if it fails (not markdown), fall back to windowed
        sections = header_splitter.split_text(d.page_content)
        if not sections:
            # Fallback to windowed split on the whole page document
            windowed_chunks = window_splitter.split_documents([d])
            pre_chunks.extend(windowed_chunks)
            continue
            
        # Wrap sections back into Documents, carry metadata + section titles
        sec_docs = []
        for s in sections:
            meta = dict(d.metadata)
            # concatenate detected headers for stronger anchors
            title = " / ".join([v for k, v in s.metadata.items() if k in ("h1","h2","h3") and v])
            meta["section_title"] = title
            sec_docs.append(Document(page_content=s.page_content, metadata=meta))
            
        # Apply windowed split to the semantically sectioned documents
        windowed_chunks = window_splitter.split_documents(sec_docs)
        pre_chunks.extend(windowed_chunks)

    # --- Add stable chunk_id and chunk_index to all final chunks ---
    out = []
    for i, ch in enumerate(pre_chunks):
        m = dict(ch.metadata)
        content = ch.page_content.strip()
        
        # Ensure doc_id is available for stable chunk_id generation
        doc_id = m.get("doc_id", hashlib.sha1(m.get("source", "").encode()).hexdigest()[:12])

        # Store a stable chunk_id for indexing/citation
        # The ID is generated from doc_id, page, chunk index, and the first 80 chars of content
        m["chunk_id"] = hashlib.sha1(
            f'{doc_id}:{m.get("page")}:{i}:{content[:80]}'.encode()
        ).hexdigest()[:12]
        
        m["chunk_index"] = i
        m["chunk_type"] = "chunk" # Clarify this is a final chunk, not just a page

        out.append(Document(page_content=content, metadata=m))
        
    return out

# -------------------------
# 7) Pipeline function: PDF folder → clean, annotated, deduped chunks
# -------------------------
def ingest_pdf_folder(folder: Path, tenant: str = "default", acl: str = "internal") -> List[Document]:
    all_pages: List[Document] = []
    for pdf in sorted(folder.glob("*.pdf")):
        all_pages.extend(load_and_clean_pdf(pdf, tenant=tenant, acl=acl))

    # Chunk
    # chunks = chunk_docs(all_pages)
    chunks = semantic_then_window(all_pages)

    # Dedupe
    deduper = Deduper(near_thresh=0.98)
    unique: List[Document] = []
    for d in chunks:
        if not deduper.is_duplicate(d.page_content):
            unique.append(d)

    # SAFETY CHECK: If deduplication removes everything, return original chunks
    if len(unique) == 0 and len(chunks) > 0:
        print("⚠️  WARNING: Deduplication removed all chunks! Returning original chunks.")
        return chunks, all_pages

    
    return unique, all_pages

# def ingest_pdf_folder_debug(folder: Path, tenant: str = "default", acl: str = "internal") -> tuple:
#     """
#     Debug version with verbose output to identify where chunks are lost
#     """
#     print(f"\n{'='*70}")
#     print(f"🔍 DEBUG: Starting PDF ingestion for folder: {folder}")
#     print(f"{'='*70}")
    
#     all_pages = []
#     pdf_files = list(folder.glob("*.pdf"))
    
#     print(f"📂 Found {len(pdf_files)} PDF file(s)")
    
#     for pdf in sorted(pdf_files):
#         print(f"\n📄 Processing: {pdf.name}")
#         print("-" * 70)
        
#         try:
#             # from back_indx.ingestion import load_and_clean_pdf
#             pages = load_and_clean_pdf(pdf, tenant=tenant, acl=acl)
            
#             print(f"   ✅ Loaded {len(pages)} pages")
            
#             # Show content stats
#             total_chars = sum(len(p.page_content) for p in pages)
#             print(f"   📊 Total characters: {total_chars}")
            
#             if pages:
#                 first_page_chars = len(pages[0].page_content)
#                 print(f"   📄 First page: {first_page_chars} chars")
#                 if first_page_chars > 0:
#                     print(f"   📝 Preview: {pages[0].page_content[:150]}...")
            
#             all_pages.extend(pages)
            
#         except Exception as e:
#             print(f"   ❌ Error processing {pdf.name}: {e}")
#             import traceback
#             traceback.print_exc()
    
#     print(f"\n{'='*70}")
#     print(f"📊 Total pages before chunking: {len(all_pages)}")
#     print(f"{'='*70}")
    
#     if not all_pages:
#         print("❌ No pages loaded from any PDF!")
#         return [], []
    
#     # Check total content
#     total_content = sum(len(p.page_content) for p in all_pages)
#     print(f"📏 Total content: {total_content} characters")
    
#     # Chunk
#     print(f"\n{'='*70}")
#     print(f"✂️  Starting chunking process...")
#     print(f"{'='*70}")
    
#     try:
#         # from back_indx.ingestion import semantic_then_window
#         chunks = semantic_then_window(all_pages)
#         print(f"✅ Created {len(chunks)} chunks")
        
#         if chunks:
#             avg_chunk_size = sum(len(c.page_content) for c in chunks) / len(chunks)
#             print(f"📊 Average chunk size: {avg_chunk_size:.0f} chars")
#             print(f"📝 First chunk preview: {chunks[0].page_content[:150]}...")
        
#     except Exception as e:
#         print(f"❌ Chunking failed: {e}")
#         import traceback
#         traceback.print_exc()
#         return [], all_pages
    
#     # Dedupe
#     print(f"\n{'='*70}")
#     print(f"🔄 Starting deduplication (near_thresh=0.90)...")
#     print(f"{'='*70}")
    
#     try:
#         # from back_indx.ingestion import Deduper
        
#         # LOWER THE THRESHOLD FOR TESTING
#         deduper = Deduper(near_thresh=0.95)  # Increased from 0.90
#         unique = []
#         duplicates_count = 0
        
#         for i, d in enumerate(chunks):
#             is_dup = deduper.is_duplicate(d.page_content)
#             if not is_dup:
#                 unique.append(d)
#             else:
#                 duplicates_count += 1
#                 if duplicates_count <= 3:  # Show first few duplicates
#                     print(f"   ⚠️  Chunk {i} marked as duplicate: {d.page_content[:100]}...")
        
#         print(f"✅ Deduplication complete:")
#         print(f"   - Original chunks: {len(chunks)}")
#         print(f"   - Duplicates removed: {duplicates_count}")
#         print(f"   - Unique chunks: {len(unique)}")
        
#         if duplicates_count == len(chunks):
#             print(f"\n❌ ALL CHUNKS MARKED AS DUPLICATES!")
#             print(f"   This is the problem. The deduplication is too aggressive.")
#             print(f"   Possible causes:")
#             print(f"   1. Document has very repetitive content")
#             print(f"   2. near_thresh=0.90 is too low")
#             print(f"   3. Short document causing false positives")
        
#         return unique, all_pages
        
#     except Exception as e:
#         print(f"❌ Deduplication failed: {e}")
#         import traceback
#         traceback.print_exc()
#         return chunks, all_pages


# # Test function
# if __name__ == "__main__":
#     from pathlib import Path
#     import sys
    
#     if len(sys.argv) < 2:
#         folder = Path("uploads/hairstylist")
#     else:
#         folder = Path(sys.argv[1])
    
#     print(f"Testing ingestion with: {folder}")
#     chunks, pages = ingest_pdf_folder_debug(folder, tenant="hairstylist", acl="internal")
    
#     print(f"\n{'='*70}")
#     print(f"FINAL RESULTS:")
#     print(f"{'='*70}")
#     print(f"Pages: {len(pages)}")
#     print(f"Chunks: {len(chunks)}")
    
#     if chunks:
#         print(f"\n✅ SUCCESS! {len(chunks)} chunks ready for embedding")
#     else:
#         print(f"\n❌ FAILED: No chunks produced")