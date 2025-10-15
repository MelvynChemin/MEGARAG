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

def learn_boilerplate_lines(pages: List[str], top_k: int = 3, bottom_k: int = 3, freq_threshold: float = 0.5) -> Set[str]:
    """
    Heuristic: grab top/bottom K lines of each page; if a line repeats in >= freq_threshold of pages, treat as boilerplate
    """
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
    return {ln for ln, c in counter.items() if c / max(1, total) >= freq_threshold}

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
    text = EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    text = PHONE_RE.sub("[REDACTED_PHONE]", text)
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
    base_meta = parse_filename_metadata(pdf_path)
    base_meta.update({"tenant": tenant, "acl": acl})

    # Load raw pages
    loader = PyPDFLoader(str(pdf_path))
    raw_docs = loader.load()  # one Document per page, metadata has {"source": path, "page": int}

    # Learn boilerplate lines across pages
    pages_text = [d.page_content or "" for d in raw_docs]
    boiler = learn_boilerplate_lines(pages_text)

    cleaned_docs = []
    for d in raw_docs:
        text = normalize_text(d.page_content or "")
        text = strip_boilerplate(text, boiler)
        text = redact_pii(text)

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
    deduper = Deduper(near_thresh=0.90)
    unique: List[Document] = []
    for d in chunks:
        if not deduper.is_duplicate(d.page_content):
            unique.append(d)

    return unique, all_pages
