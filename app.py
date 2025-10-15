# app.py - Complete FastAPI Backend

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi import Path as ApiPath
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from pathlib import Path as FSPath
from typing import List
import shutil, re, subprocess, sys, os
from datetime import datetime

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

# Store RAG pipeline status (in production, use a database)
rag_status = {}

# --- Helper Functions ---
SAFE_SLUG = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$", re.I)

def normalize_tenant(raw: str) -> str:
    slug = re.sub(r"\s+", "-", raw.strip().lower())
    slug = re.sub(r"[^a-z0-9_-]", "-", slug)
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    if not slug or not SAFE_SLUG.match(slug):
        raise HTTPException(status_code=400, detail="Invalid company name")
    return slug

def run_rag_pipeline(tenant_slug: str):
    """
    Execute the indx_pipeline.py script for the given tenant.
    This runs in the background.
    """
    try:
        pipeline_script = BASE_DIR / "indx_pipeline.py"
        
        if not pipeline_script.exists():
            raise FileNotFoundError(f"Pipeline script not found: {pipeline_script}")
        venv_path = BASE_DIR / ".venv" / "bin" / "python"
        result = subprocess.run(
            [venv_path, str(pipeline_script), tenant_slug],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            rag_status[tenant_slug] = {
                "status": "completed",
                "completed_at": datetime.now().isoformat(),
                "output": result.stdout
            }
        else:
            rag_status[tenant_slug] = {
                "status": "failed",
                "error": result.stderr or "Unknown error",
                "failed_at": datetime.now().isoformat()
            }
            
    except subprocess.TimeoutExpired:
        rag_status[tenant_slug] = {
            "status": "failed",
            "error": "Pipeline execution timed out",
            "failed_at": datetime.now().isoformat()
        }
    except Exception as e:
        rag_status[tenant_slug] = {
            "status": "failed",
            "error": str(e),
            "failed_at": datetime.now().isoformat()
        }


# --- Routes ---

@app.get("/")
async def root():
    """Serve main upload page"""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/{tenant}")
async def tenant_dashboard(tenant: str = ApiPath(..., description="Company name")):
    """Serve tenant-specific dashboard"""
    normalize_tenant(tenant)  # Validate tenant name
    return FileResponse(STATIC_DIR / "dashboard.html")


@app.post("/upload/{tenant}")
async def upload_files_for_tenant(
    tenant: str = ApiPath(..., description="Company name"),
    files: List[UploadFile] = File(...),
):
    """Upload files for a specific tenant"""
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


@app.get("/files/{tenant}")
async def list_tenant_files(tenant: str = ApiPath(..., description="Company name")):
    """List all files for a specific tenant"""
    tenant_slug = normalize_tenant(tenant)
    tenant_dir = UPLOAD_DIR / tenant_slug
    
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
    """Download a specific file"""
    tenant_slug = normalize_tenant(tenant)
    safe_filename = FSPath(filename).name
    file_path = UPLOAD_DIR / tenant_slug / safe_filename
    
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"File '{safe_filename}' not found for tenant '{tenant_slug}'"
        )
    
    if not file_path.is_file():
        raise HTTPException(status_code=400, detail="Invalid file path")
    
    try:
        file_path.resolve().relative_to(UPLOAD_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return FileResponse(
        path=file_path,
        filename=safe_filename,
        media_type="application/octet-stream"
    )


@app.delete("/delete/{tenant}/{filename}")
async def delete_file(
    tenant: str = ApiPath(..., description="Company name"),
    filename: str = ApiPath(..., description="File name to delete")
):
    """Delete a specific file"""
    tenant_slug = normalize_tenant(tenant)
    safe_filename = FSPath(filename).name
    file_path = UPLOAD_DIR / tenant_slug / safe_filename
    
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"File '{safe_filename}' not found"
        )
    
    try:
        file_path.resolve().relative_to(UPLOAD_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")
    
    os.remove(file_path)
    
    return JSONResponse({
        "tenant": tenant_slug,
        "filename": safe_filename,
        "status": "deleted"
    })


@app.post("/rag/create/{tenant}")
async def create_rag_index(
    tenant: str = ApiPath(..., description="Company name"),
    background_tasks: BackgroundTasks = None
):
    """Trigger RAG pipeline for tenant's documents"""
    tenant_slug = normalize_tenant(tenant)
    tenant_dir = UPLOAD_DIR / tenant_slug
    
    if not tenant_dir.exists() or not any(tenant_dir.iterdir()):
        raise HTTPException(
            status_code=400,
            detail="No files found. Please upload documents first."
        )
    
    if tenant_slug in rag_status and rag_status[tenant_slug]["status"] == "processing":
        return JSONResponse({
            "message": "RAG index creation already in progress",
            "status": "processing"
        })
    
    rag_status[tenant_slug] = {
        "status": "processing",
        "started_at": datetime.now().isoformat(),
        "error": None
    }
    
    background_tasks.add_task(run_rag_pipeline, tenant_slug)
    
    return JSONResponse({
        "message": f"RAG index creation started for {tenant_slug}",
        "status": "processing"
    })


@app.get("/rag/status/{tenant}")
async def get_rag_status(tenant: str = ApiPath(..., description="Company name")):
    """Check RAG pipeline status"""
    tenant_slug = normalize_tenant(tenant)
    
    if tenant_slug not in rag_status:
        return JSONResponse({
            "status": "not_started",
            "message": "No RAG index creation has been initiated"
        })
    
    return JSONResponse(rag_status[tenant_slug])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)