"""
Module đọc và trích xuất văn bản từ file PDF sử dụng PyMuPDF (fitz).
Đọc từng trang riêng biệt, giữ metadata vị trí (số trang).
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import pymupdf as fitz  # PyMuPDF


def extract_pdf_metadata(doc: fitz.Document, file_path: str) -> dict:
    """
    Trích xuất metadata cấp tài liệu từ file PDF.
    Các trường không xác định được sẽ để giá trị mặc định.
    """
    meta = doc.metadata or {}
    return {
        "title": meta.get("title", "").strip() or Path(file_path).stem,
        "author": meta.get("author", "").strip() or "Chưa rõ",
        "subject": meta.get("subject", "").strip() or "",
        "total_pages": len(doc),
        "source": os.path.basename(file_path),
    }


def load_pdf(pdf_path: str) -> Dict:
    """
    Đọc một file PDF và trích xuất văn bản theo từng trang.

    Args:
        pdf_path: Đường dẫn tuyệt đối hoặc tương đối tới file PDF.

    Returns:
        Dict với keys:
            - "metadata": dict chứa thông tin tài liệu
            - "pages": list[dict] mỗi phần tử có
                {"page_number": int, "text": str, "pdf_page_index": int}

    Raises:
        FileNotFoundError: Nếu file không tồn tại.
        ValueError: Nếu PDF không có văn bản trích xuất được.
    """
    pdf_path = str(pdf_path)
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"Không tìm thấy file PDF: {pdf_path}")

    start_time = time.time()
    print(f"[PDF Loader] Đang đọc: {os.path.basename(pdf_path)}")

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise ValueError(f"Không thể mở file PDF '{pdf_path}': {e}")

    pages = []
    empty_page_count = 0
    try:
        metadata = extract_pdf_metadata(doc, pdf_path)
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            text = page.get_text("text")  # Trích xuất văn bản thuần

            if not text or not text.strip():
                empty_page_count += 1
                text = ""

            pages.append({
                "page_number": page_idx + 1,  # 1-based, trang vật lý trong PDF
                "text": text,
                "pdf_page_index": page_idx,  # 0-based, dùng nội bộ
            })
    finally:
        doc.close()

    # Kiểm tra: nếu tất cả trang đều trống, cảnh báo
    total_text_chars = sum(len(p["text"].strip()) for p in pages)
    if total_text_chars == 0:
        raise ValueError(
            f"PDF '{os.path.basename(pdf_path)}' không có văn bản trích xuất được. "
            f"File có thể là bản scan (ảnh). Cần sử dụng OCR để xử lý."
        )

    elapsed = time.time() - start_time
    print(
        f"[PDF Loader] Hoàn thành: {metadata['source']} | "
        f"{metadata['total_pages']} trang | "
        f"{total_text_chars:,} ký tự | "
        f"{empty_page_count} trang trống | "
        f"{elapsed:.2f}s"
    )

    return {"metadata": metadata, "pages": pages}


def load_all_pdfs(pdf_dir: str, errors: Optional[List[dict]] = None) -> Dict[str, Dict]:
    """
    Đọc tất cả file PDF trong một thư mục.

    Args:
        pdf_dir: Đường dẫn thư mục chứa các file PDF.
        errors: Nếu truyền vào, các lỗi {"file", "error"} sẽ được thêm vào list này.

    Returns:
        Dict mapping tên file -> kết quả load_pdf().
    """
    pdf_dir = Path(pdf_dir)
    if not pdf_dir.is_dir():
        raise FileNotFoundError(f"Thư mục không tồn tại: {pdf_dir}")

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"[PDF Loader] Không tìm thấy file PDF nào trong: {pdf_dir}")
        return {}

    print(f"[PDF Loader] Tìm thấy {len(pdf_files)} file PDF trong {pdf_dir}")
    results = {}
    if errors is None:
        errors = []

    for pdf_file in pdf_files:
        try:
            result = load_pdf(str(pdf_file))
            results[pdf_file.name] = result
        except (ValueError, FileNotFoundError) as e:
            error_msg = f"[PDF Loader] LỖI khi đọc {pdf_file.name}: {e}"
            print(error_msg)
            errors.append({"file": pdf_file.name, "error": str(e)})

    print(
        f"[PDF Loader] Kết quả: {len(results)} thành công, "
        f"{len(errors)} thất bại trên tổng {len(pdf_files)} file."
    )
    return results
