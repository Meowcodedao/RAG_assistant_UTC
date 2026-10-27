"""Evaluate retrieval against a small manually curated benchmark (Recall@K, MRR)."""
import json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.config import settings
from app.services.rag import RAGService
from app.services.evaluation import load_queries, evaluate

def main():
    queries=load_queries()
    if not queries:
        raise SystemExit(f"Không thấy bộ câu hỏi đánh giá: {settings.EVALUATION_DIR/'queries.json'}")
    rag=RAGService()
    if not rag.retriever:
        raise SystemExit("Chưa có index. Hãy chạy python scripts/ingest.py trước.")
    out=evaluate(rag.retriever,rag.chunks,queries)
    out={"ground_truth_note":"ground_truth_mode='chunk' dùng relevant_chunk_ids; 'page' dùng mọi chunk trên trang ground truth.",
         "semantic_backend":rag.retriever.semantic_backend,**out}
    outpath=settings.REPORTS_DIR/"retrieval_evaluation.json"
    outpath.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(out["summary"],ensure_ascii=False,indent=2))
    print(f"\nĐã lưu báo cáo: {outpath}")

if __name__=="__main__": main()
