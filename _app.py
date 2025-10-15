# app.py (fixed)

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi import Path as ApiPath                      # <-- FastAPI Path helper
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from pathlib import Path as FSPath                       # <-- filesystem Path
from typing import List
import shutil, re

app = FastAPI(title="Document Upload/Download API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Folders ---
BASE_DIR   = FSPath(__file__).parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def root():
    return FileResponse(STATIC_DIR / "index.html")

SAFE_SLUG = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$", re.I)
def normalize_tenant(raw: str) -> str:
    slug = re.sub(r"\s+", "-", raw.strip().lower())
    slug = re.sub(r"[^a-z0-9_-]", "-", slug)
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    if not slug or not SAFE_SLUG.match(slug):
        raise HTTPException(status_code=400, detail="Invalid company name")
    return slug

# Tenant-aware upload
@app.post("/upload/{tenant}")
async def upload_files_for_tenant(
    tenant: str = ApiPath(..., description="Company name"),   # <-- use ApiPath
    files: List[UploadFile] = File(...),
):
    tenant_slug = normalize_tenant(tenant)
    dest_dir = UPLOAD_DIR / tenant_slug
    dest_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for uf in files:
        if not uf.filename:
            continue
        dst = dest_dir / uf.filename
        with dst.open("wb") as out:
            shutil.copyfileobj(uf.file, out)
        saved.append(uf.filename)

    return JSONResponse({"tenant": tenant_slug, "count": len(saved)})

# (optional) legacy endpoints below… keep or remove as you like

@app.get("/files/{tenant}")
async def list_tenant_files(
    tenant: str = ApiPath(..., description="Company name")
):
    """
    List all files uploaded by a specific tenant.
    Returns file names, sizes, and modification times.
    """
    tenant_slug = normalize_tenant(tenant)
    tenant_dir = UPLOAD_DIR / tenant_slug
    
    # Check if tenant folder exists
    if not tenant_dir.exists():
        return JSONResponse({
            "tenant": tenant_slug,
            "files": [],
            "message": "No files uploaded yet"
        })
    
    files_info = []
    for file_path in tenant_dir.iterdir():
        if file_path.is_file():
            stat = file_path.stat()
            files_info.append({
                "filename": file_path.name,
                "size_bytes": stat.st_size,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "modified": stat.st_mtime,
            })
    
    # Sort by filename
    files_info.sort(key=lambda x: x["filename"])
    
    return JSONResponse({
        "tenant": tenant_slug,
        "count": len(files_info),
        "files": files_info
    })

@app.get("/download/{tenant}/{filename}")
async def download_file(
    tenant: str = ApiPath(..., description="Company name"),
    filename: str = ApiPath(..., description="File name to download")
):
    """
    Download a specific file from a tenant's folder.
    Returns the file with appropriate headers for download.
    """
    tenant_slug = normalize_tenant(tenant)
    
    # Sanitize filename to prevent directory traversal
    safe_filename = FSPath(filename).name  # Strips any path components
    file_path = UPLOAD_DIR / tenant_slug / safe_filename
    
    # Security checks
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"File '{safe_filename}' not found for tenant '{tenant_slug}'"
        )
    
    if not file_path.is_file():
        raise HTTPException(
            status_code=400,
            detail="Invalid file path"
        )
    
    # Ensure the resolved path is still within the tenant folder (prevent traversal)
    try:
        file_path.resolve().relative_to(UPLOAD_DIR.resolve())
    except ValueError:
        raise HTTPException(
            status_code=403,
            detail="Access denied"
        )
    
    return FileResponse(
        path=file_path,
        filename=safe_filename,
        media_type="application/octet-stream"
    )
