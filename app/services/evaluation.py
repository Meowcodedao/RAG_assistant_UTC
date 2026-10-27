"""
Đánh giá retrieval (Recall@K, MRR) và chạy thử nghiệm nhiều cấu hình.

Ground truth:
  - Mức chunk: relevant_chunk_ids trong queries.json (chỉ đúng với kho chunk đang dùng).
  - Mức trang: mọi chunk thuộc ground_truth_pages của ground_truth_source. Dùng khi câu hỏi
    chưa gán chunk, hoặc khi so sánh các cấu hình chunking khác nhau (ID chunk thay đổi).
"""

import json
from itertools import product
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from app.config import settings
from app.services.corpus import build_chunks
from app.services.retrieval import HybridRetriever

METHODS = ("hybrid", "semantic", "keyword")
KS = (1, 3, 5)


def queries_path():
    return settings.EVALUATION_DIR / "queries.json"


def load_queries() -> List[dict]:
    path = queries_path()
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def save_queries(queries: List[dict]) -> None:
    queries_path().write_text(json.dumps(queries, ensure_ascii=False, indent=2), encoding="utf-8")


def ground_truth_pages(q: dict) -> List[int]:
    pages = q.get("ground_truth_pages") or [q.get("ground_truth_page")]
    return sorted({int(p) for p in pages if p is not None})


def relevant_ids(q: dict, chunks: List[dict], chunk_level: bool = True) -> Tuple[set, str]:
    """Trả về (tập chunk_id đúng, "chunk" hoặc "page")."""
    if chunk_level and q.get("relevant_chunk_ids"):
        existing = {c["chunk_id"] for c in chunks}
        rel = set(q["relevant_chunk_ids"]) & existing
        if rel:
            return rel, "chunk"
    pages = set(ground_truth_pages(q))
    rel = {c["chunk_id"] for c in chunks
           if c.get("source") == q.get("ground_truth_source") and pages & set(chunk_pages(c))}
    return rel, "page"


def chunk_pages(chunk: dict) -> List[int]:
    """Các trang gốc của chunk (nhiều trang nếu slide đã được gộp khi chuẩn hóa)."""
    return chunk.get("pages") or [chunk.get("page")]


def _metrics(first_ranks: List[Optional[int]], ks: Sequence[int]) -> dict:
    n = max(1, len(first_ranks))
    out = {"queries": len(first_ranks),
           "MRR": round(sum(1 / r for r in first_ranks if r) / n, 4)}
    for k in ks:
        out[f"Recall@{k}"] = round(sum(1 for r in first_ranks if r and r <= k) / n, 4)
    return out


def evaluate(retriever: HybridRetriever, chunks: List[dict], queries: List[dict],
             methods: Sequence[str] = METHODS, ks: Sequence[int] = KS,
             chunk_level: bool = True) -> dict:
    """Chạy benchmark. Returns {"summary", "summary_by_type", "details"} theo từng phương pháp."""
    out = {"summary": {}, "summary_by_type": {}, "details": {}}
    for method in methods:
        rows = []
        for q in queries:
            results = retriever.search(q["question"], method, max(ks))
            relevant, mode = relevant_ids(q, chunks, chunk_level)
            ranks = [i + 1 for i, r in enumerate(results) if r["chunk"]["chunk_id"] in relevant]
            rows.append({
                "query_id": q["query_id"], "question": q["question"],
                "query_type": q.get("query_type", ""),
                "ground_truth": f"{q.get('ground_truth_source')} p.{','.join(map(str, ground_truth_pages(q)))}",
                "ground_truth_mode": mode,
                "hit": bool(ranks), "first_relevant_rank": min(ranks) if ranks else None,
                "top_sources": [f"{r['chunk']['source']} p.{r['chunk']['page']}" for r in results],
                "top_chunk_ids": [r["chunk"]["chunk_id"] for r in results],
                "top_relevant": [r["chunk"]["chunk_id"] in relevant for r in results],
            })
        out["details"][method] = rows
        out["summary"][method] = _metrics([r["first_relevant_rank"] for r in rows], ks)
        by_type = {}
        for qtype in sorted({r["query_type"] for r in rows}):
            by_type[qtype] = _metrics([r["first_relevant_rank"] for r in rows if r["query_type"] == qtype], ks)
        out["summary_by_type"][method] = by_type
    return out


def experiment_settings(embedding_model: str, rrf_k: int) -> SimpleNamespace:
    return SimpleNamespace(
        EMBEDDING_MODEL_NAME=embedding_model, RRF_K=rrf_k,
        FAISS_TOP_K=settings.FAISS_TOP_K, BM25_TOP_K=settings.BM25_TOP_K, FINAL_TOP_K=settings.FINAL_TOP_K,
    )


def run_experiments(documents: List[dict], queries: List[dict],
                    strategies: Sequence[str], chunk_sizes: Sequence[int], overlaps: Sequence[int],
                    embedding_models: Sequence[str], rrf_ks: Sequence[int],
                    methods: Sequence[str] = METHODS, slide_cleanups: Sequence[bool] = (True,),
                    progress: Optional[Callable[[float, str], None]] = None) -> List[dict]:
    """
    Chạy lưới cấu hình, dựng kho chunk trong bộ nhớ cho mỗi cấu hình (không ghi đè data/processed).
    Luôn dùng ground truth mức trang để các cấu hình chunking khác nhau so sánh được với nhau.
    """
    configs = [(clean, s, size, ov, model, rrf) for clean, s, size, ov, model, rrf
               in product(slide_cleanups, strategies, chunk_sizes, overlaps, embedding_models, rrf_ks) if ov < size]
    rows = []
    chunk_cache: Dict[tuple, List[dict]] = {}
    for i, (clean, strategy, size, overlap, model, rrf_k) in enumerate(configs):
        label = f"slide_cleanup={clean} | {strategy} size={size} overlap={overlap} | {model} | RRF_K={rrf_k}"
        if progress:
            progress(i / max(1, len(configs)), f"Cấu hình {i + 1}/{len(configs)}: {label}")
        key = (clean, strategy, size, overlap)
        if key not in chunk_cache:
            chunk_cache[key], _ = build_chunks(documents, strategy, size, overlap, clean, settings.MIN_PAGE_CHARS)
        chunks = chunk_cache[key]
        retriever = HybridRetriever(chunks, experiment_settings(model, rrf_k))
        result = evaluate(retriever, chunks, queries, methods, chunk_level=False)
        for method in methods:
            rows.append({
                "slide_cleanup": clean, "strategy": strategy, "chunk_size": size, "chunk_overlap": overlap,
                "embedding_model": model, "rrf_k": rrf_k, "method": method,
                "chunks": len(chunks), "semantic_backend": retriever.semantic_backend,
                **result["summary"][method],
            })
    if progress:
        progress(1.0, "Hoàn tất")
    return rows
