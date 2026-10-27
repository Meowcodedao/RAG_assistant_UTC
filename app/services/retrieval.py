"""Hybrid retrieval: semantic search (FAISS) + lexical BM25 + Reciprocal Rank Fusion."""
from __future__ import annotations
import json, re, unicodedata
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

TOKEN_RE = re.compile(r"[A-Za-zÀ-ỹĐđ0-9]+", re.UNICODE)

_ENCODERS: Dict[str, Any] = {}

def get_encoder(model_name: str):
    """Nạp SentenceTransformer một lần cho mỗi tên model (thử nghiệm dùng lại nhiều lần)."""
    if model_name not in _ENCODERS:
        from sentence_transformers import SentenceTransformer
        _ENCODERS[model_name] = SentenceTransformer(model_name)
    return _ENCODERS[model_name]

def tokenize(text: str) -> List[str]:
    # NFC để câu hỏi gõ dạng tổ hợp (NFD) vẫn khớp với văn bản PDF đã chuẩn hóa
    return [x.lower() for x in TOKEN_RE.findall(unicodedata.normalize("NFC", text or ""))]

class HybridRetriever:
    def __init__(self, chunks: List[dict], settings):
        self.chunks = chunks
        self.settings = settings
        self.encoder = None
        self.faiss = None
        self.matrix = None
        self.bm25 = None
        self.bm25_tokens = [tokenize(c.get("text", "")) for c in chunks]
        if not chunks:
            return
        try:
            from rank_bm25 import BM25Okapi
            self.bm25 = BM25Okapi(self.bm25_tokens)
        except Exception:
            self.bm25 = None
        try:
            import faiss
            self.encoder = get_encoder(settings.EMBEDDING_MODEL_NAME)
            vectors = self.encoder.encode(
                [c.get("text", "") for c in chunks],
                convert_to_numpy=True, normalize_embeddings=True,
                show_progress_bar=False
            ).astype("float32")
            self.matrix = vectors
            self.faiss = faiss.IndexFlatIP(vectors.shape[1])
            self.faiss.add(vectors)
        except Exception as exc:
            # TF-IDF is a lightweight fallback when model downloads/FAISS are unavailable.
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                self.vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None)
                self.matrix = self.vectorizer.fit_transform([c.get("text", "") for c in chunks])
                self.encoder = "tfidf"
            except Exception:
                self.vectorizer = None
            print(f"[Retriever] Semantic model unavailable; using fallback if available. Detail: {exc}")

    @property
    def semantic_backend(self) -> str:
        """Nhánh ngữ nghĩa đang thực sự được dùng (để giao diện báo khi phải fallback)."""
        if self.faiss is not None and self.encoder != "tfidf":
            return f"FAISS + {self.settings.EMBEDDING_MODEL_NAME}"
        if getattr(self, "vectorizer", None) is not None:
            return "TF-IDF (fallback)"
        return "lexical overlap (fallback)"

    def _semantic(self, query: str, k: int) -> List[tuple]:
        if not self.chunks:
            return []
        if self.faiss is not None and self.encoder != "tfidf":
            q = self.encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
            scores, ids = self.faiss.search(q, min(k, len(self.chunks)))
            return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]
        if getattr(self, "vectorizer", None) is not None:
            qv = self.vectorizer.transform([query])
            scores = (self.matrix @ qv.T).toarray().ravel()
            ids = np.argsort(scores)[::-1][:k]
            return [(int(i), float(scores[i])) for i in ids if scores[i] > 0]
        # Last-resort lexical overlap scoring
        qt = set(tokenize(query))
        scored = [(i, len(qt & set(t)) / max(1, len(qt)) ) for i,t in enumerate(self.bm25_tokens)]
        return [x for x in sorted(scored, key=lambda x:x[1], reverse=True)[:k] if x[1] > 0]

    def _lexical(self, query: str, k: int) -> List[tuple]:
        if not self.chunks:
            return []
        if self.bm25 is not None:
            scores = self.bm25.get_scores(tokenize(query))
        else:
            qt = set(tokenize(query))
            scores = np.array([len(qt & set(t)) / max(1, len(qt)) for t in self.bm25_tokens])
        ids = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in ids if scores[i] > 0]

    def search(self, query: str, method: str = "hybrid", top_k: int | None = None) -> List[dict]:
        if not query or not query.strip():
            return []
        final_k = top_k or self.settings.FINAL_TOP_K
        method = method.lower()
        # Số ứng viên mỗi nhánh không được nhỏ hơn số kết quả người dùng yêu cầu
        semantic = self._semantic(query, max(self.settings.FAISS_TOP_K, final_k)) if method != "keyword" else []
        lexical = self._lexical(query, max(self.settings.BM25_TOP_K, final_k)) if method != "semantic" else []
        if method == "semantic":
            ranked = [(i, 1.0/(rank+1)) for rank,(i,_) in enumerate(semantic)]
        elif method == "keyword":
            ranked = [(i, 1.0/(rank+1)) for rank,(i,_) in enumerate(lexical)]
        else:
            fused: Dict[int, dict] = {}
            for rank,(idx,score) in enumerate(semantic, 1):
                fused.setdefault(idx, {"rrf":0.0})["rrf"] += 1/(self.settings.RRF_K+rank)
                fused[idx]["faiss_rank"] = rank
                fused[idx]["faiss_score"] = score
            for rank,(idx,score) in enumerate(lexical, 1):
                fused.setdefault(idx, {"rrf":0.0})["rrf"] += 1/(self.settings.RRF_K+rank)
                fused[idx]["bm25_rank"] = rank
                fused[idx]["bm25_score"] = score
            ranked = sorted([(i,v["rrf"]) for i,v in fused.items()], key=lambda x:x[1], reverse=True)
        output=[]
        for final_rank,(idx,score) in enumerate(ranked[:final_k],1):
            meta={}
            if method == "hybrid":
                meta=fused.get(idx,{})
            elif method == "semantic":
                meta={"faiss_rank":final_rank, "faiss_score": next((s for i,s in semantic if i==idx),None)}
            else:
                meta={"bm25_rank":final_rank, "bm25_score": next((s for i,s in lexical if i==idx),None)}
            output.append({
                "chunk": self.chunks[idx],
                "faiss_rank":meta.get("faiss_rank"), "faiss_score":meta.get("faiss_score"),
                "bm25_rank":meta.get("bm25_rank"), "bm25_score":meta.get("bm25_score"),
                "rrf_score":meta.get("rrf"), "final_rank":final_rank
            })
        return output
