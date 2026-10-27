"""
Dựng kho chunk từ thư mục PDF: đọc → làm sạch → chunking.
Dùng chung cho scripts/ingest.py và tab Thử nghiệm trên giao diện
(thử nghiệm dựng kho trong bộ nhớ, không ghi đè data/processed).
"""

import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.services.chunking import chunk_documents
from app.services.pdf_loader import load_all_pdfs
from app.services.text_cleaner import clean_page_texts, normalize_slide_pages


def slugify(name: str) -> str:
    # Bỏ dấu tiếng Việt trước khi lọc ký tự: "Bài 5.pdf" -> "bai_5" (không phải "b_i_5")
    stem = unicodedata.normalize("NFKD", Path(name).stem.replace("đ", "d").replace("Đ", "D"))
    stem = "".join(ch for ch in stem if not unicodedata.combining(ch))
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", stem).strip("_").lower() or "document"


def unique_id(base: str, used: set) -> str:
    doc_id = base
    n = 2
    while doc_id in used:
        doc_id = f"{base}_{n}"
        n += 1
    used.add(doc_id)
    return doc_id


def _course_for(doc_id: str) -> str:
    if "bai5" in doc_id:
        return "Huấn luyện mạng nơ-ron"
    if "bai8" in doc_id or "recurrent" in doc_id:
        return "Mạng hồi quy (RNN)"
    return "Học sâu và ứng dụng"


def load_cleaned_documents(pdf_dir, errors: Optional[List[dict]] = None) -> List[dict]:
    """
    Đọc và làm sạch mọi PDF trong pdf_dir.

    Returns:
        List các dict {"meta": metadata tài liệu theo schema DocumentMetadata,
                       "pages": danh sách trang đã làm sạch}.
    """
    loaded = load_all_pdfs(str(pdf_dir), errors)
    documents = []
    used_ids = set()
    for filename, payload in loaded.items():
        metadata = payload["metadata"]
        doc_id = unique_id(slugify(filename), used_ids)
        pdf_title = (metadata.get("title") or "").strip()
        if not pdf_title or pdf_title.lower() in {"powerpoint presentation", "microsoft powerpoint"}:
            pdf_title = Path(filename).stem
        documents.append({
            "meta": {
                "document_id": doc_id, "document_title": pdf_title,
                "author": metadata.get("author", "Chưa rõ"), "year": None, "course": _course_for(doc_id),
                "source": filename, "source_url": None, "total_pages": metadata["total_pages"],
            },
            "pages": clean_page_texts(payload["pages"]),
        })
    return documents


def build_chunks(documents: List[dict], strategy: str, chunk_size: int, chunk_overlap: int,
                 slide_cleanup: bool = True, min_page_chars: int = 40) -> Tuple[List[dict], List[dict]]:
    """Chunk các tài liệu đã làm sạch. Returns (chunks, metadata tài liệu)."""
    all_chunks = []
    for doc in documents:
        meta = doc["meta"]
        pages = normalize_slide_pages(doc["pages"], min_page_chars) if slide_cleanup else doc["pages"]
        chunks = chunk_documents(
            pages, strategy=strategy, chunk_size=chunk_size,
            chunk_overlap=chunk_overlap, document_id=meta["document_id"], metadata=meta,
        )
        all_chunks.extend(chunks)
        print(f"[Ingest] {meta['source']}: {len(chunks)} chunks")
    return all_chunks, [doc["meta"] for doc in documents]


def page_texts(documents: List[dict]) -> Dict[Tuple[str, int], str]:
    """Mapping (tên file, số trang) → văn bản trang đã làm sạch."""
    return {(d["meta"]["source"], p["page_number"]): p["text"] for d in documents for p in d["pages"]}
