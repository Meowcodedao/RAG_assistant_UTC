"""FastAPI endpoints for the Hybrid RAG Assistant."""
import json
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from app.config import settings
from app.schemas import RAGResponse
from app.services.rag import RAGService
from app.services.ingestion import IngestError, save_uploaded_pdf, run_ingest_script

app=FastAPI(title="Hybrid RAG Assistant API", version="1.0.0",
            description="Hỏi đáp trên tài liệu PDF với truy xuất hybrid FAISS/BM25.")
rag=RAGService()

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    retrieval_method: str = Field(default="hybrid", pattern="^(hybrid|semantic|keyword)$")
    top_k: int = Field(default=5, ge=1, le=20)

@app.get("/")
def root():
    return {"name":"Hybrid RAG Assistant","docs":"/docs","health":"/health"}

@app.get("/health")
def health():
    rag.reload_if_changed()
    return {"status":"ok","documents_dir":str(settings.PDF_DIR),"chunks":len(rag.chunks),
            "ready":bool(rag.chunks)}

@app.get("/documents")
def documents():
    p=settings.PROCESSED_DIR/"documents.json"
    if not p.exists(): return {"documents":[]}
    return {"documents":json.loads(p.read_text(encoding="utf-8"))}

@app.post("/ask", response_model=RAGResponse)
def ask(req: AskRequest):
    if not req.question.strip(): raise HTTPException(status_code=400,detail="Câu hỏi không được để trống.")
    result=rag.ask(req.question.strip(),req.retrieval_method,req.top_k)
    if not rag.chunks:
        result["answer"]="Chưa có dữ liệu đã xử lý. Hãy đặt PDF vào data/pdfs rồi chạy: python scripts/ingest.py"
    return result

@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    content=await file.read()
    try:
        target,is_new=await run_in_threadpool(save_uploaded_pdf,content,file.filename)
    except IngestError as exc:
        raise HTTPException(status_code=400,detail=str(exc))
    if not is_new:
        return {"status":"exists","filename":target.name,"chunks":len(rag.chunks),
                "log":"Tài liệu này đã có trong kho, không nạp lại."}
    try:
        # Chạy trong threadpool để không chặn event loop trong lúc ingest
        log=await run_in_threadpool(run_ingest_script)
    except Exception as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=500,detail=f"Không thể nạp PDF: {exc}")
    await run_in_threadpool(rag.reload)
    return {"status":"ok","filename":target.name,"chunks":len(rag.chunks),"log":log[-2000:]}
