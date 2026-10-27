"""
Lưu PDF người dùng tải lên và chạy lại pipeline ingest.
Dùng chung cho FastAPI và Gradio để hai giao diện xử lý giống nhau.
"""

import hashlib
import subprocess
import sys
import threading
from pathlib import Path
from typing import Tuple

from app.config import settings
from app.services.pdf_loader import load_pdf

# Chỉ cho phép một tiến trình ingest chạy tại một thời điểm trong process
_ingest_lock = threading.Lock()


class IngestError(Exception):
    """Lỗi khi lưu hoặc nạp PDF; message có thể hiển thị cho người dùng."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_uploaded_pdf(content: bytes, filename: str) -> Tuple[Path, bool]:
    """
    Kiểm tra và lưu PDF vào PDF_DIR.

    Returns:
        (đường dẫn, is_new). is_new=False nếu đã có tệp nội dung y hệt trong kho
        (không lưu thêm bản trùng).

    Raises:
        IngestError: Nếu tệp không phải PDF, quá lớn hoặc không có lớp văn bản.
    """
    safe_name = Path(filename or "").name
    if not safe_name or Path(safe_name).suffix.lower() != ".pdf":
        raise IngestError("Chỉ chấp nhận tệp PDF.")
    if not content.startswith(b"%PDF"):
        raise IngestError("Tệp tải lên không có định dạng PDF hợp lệ.")
    if len(content) > settings.MAX_UPLOAD_MB * 1024 * 1024:
        raise IngestError(f"Tệp vượt quá giới hạn {settings.MAX_UPLOAD_MB} MB.")

    digest = _sha256(content)
    for existing in settings.PDF_DIR.glob("*.pdf"):
        if existing.stat().st_size == len(content) and _sha256(existing.read_bytes()) == digest:
            return existing, False

    # Không ghi đè tệp khác nội dung nhưng trùng tên
    target = settings.PDF_DIR / safe_name
    n = 2
    while target.exists():
        target = settings.PDF_DIR / f"{Path(safe_name).stem}_{n}.pdf"
        n += 1

    target.write_bytes(content)
    try:
        # Kiểm tra trước để PDF scan/hỏng không lọt vào kho rồi bị bỏ qua âm thầm
        load_pdf(str(target))
    except (ValueError, FileNotFoundError) as exc:
        target.unlink(missing_ok=True)
        raise IngestError(str(exc)) from exc
    return target, True


def run_ingest_script() -> str:
    """
    Chạy scripts/ingest.py trên toàn bộ PDF_DIR.

    Returns:
        stdout của script.

    Raises:
        IngestError: Nếu script thất bại.
    """
    with _ingest_lock:
        proc = subprocess.run(
            [sys.executable, str(settings.BASE_DIR / "scripts" / "ingest.py")],
            cwd=str(settings.BASE_DIR), capture_output=True, text=True, timeout=600,
        )
    if proc.returncode != 0:
        raise IngestError(f"Nạp tài liệu thất bại:\n{proc.stderr[-1500:]}")
    return proc.stdout
