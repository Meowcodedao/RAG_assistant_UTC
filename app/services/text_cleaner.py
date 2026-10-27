"""
Module làm sạch văn bản trích xuất từ PDF.
Xử lý khoảng trắng dư thừa, dòng trống, từ bị ngắt dòng,
và chuẩn hóa Unicode cho tiếng Việt.
"""

import re
import unicodedata
from typing import List


def clean_text(text: str) -> str:
    """
    Làm sạch văn bản từ PDF.

    Xử lý:
    - Chuẩn hóa Unicode NFC (giữ nguyên dấu tiếng Việt)
    - Sửa từ bị ngắt dòng (vd: "ma-\\nchine" → "machine")
    - Loại bỏ khoảng trắng dư thừa trong mỗi dòng
    - Giảm dòng trống liên tiếp (≥3 dòng trống → 2)
    - Giữ cấu trúc đoạn văn (double newline)

    Args:
        text: Văn bản thô từ PDF.

    Returns:
        Văn bản đã làm sạch.
    """
    if not text:
        return ""

    # 1. Chuẩn hóa Unicode NFC - quan trọng cho tiếng Việt
    text = unicodedata.normalize("NFC", text)

    # 2. Sửa từ tiếng Anh bị ngắt dòng bằng dấu gạch ngang
    # Ví dụ: "ma-\nchine" → "machine". Chỉ nối khi hai bên đều là chữ thường ASCII
    # để không dính các cụm như "RNN-\nLSTM" hay "seq-\n2-seq"; tiếng Việt
    # không ngắt âm tiết bằng gạch nối nên không bị ảnh hưởng.
    text = re.sub(r"([a-z])-\n([a-z])", r"\1\2", text)

    # 3. Loại bỏ khoảng trắng dư thừa trong mỗi dòng (giữ newline)
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        # Nhiều space/tab liên tiếp → một space
        line = re.sub(r"[ \t]+", " ", line)
        # Bỏ khoảng trắng đầu/cuối dòng
        line = line.strip()
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # 4. Giảm dòng trống liên tiếp: ≥3 dòng trống → 2 dòng trống
    text = re.sub(r"\n{4,}", "\n\n\n", text)

    # 5. Loại bỏ khoảng trắng đầu và cuối toàn bộ văn bản
    text = text.strip()

    return text


def clean_page_texts(pages: List[dict]) -> List[dict]:
    """
    Làm sạch văn bản của tất cả các trang.

    Args:
        pages: Danh sách dict với key "text" chứa văn bản thô.

    Returns:
        Danh sách dict đã làm sạch, giữ nguyên các key khác.
    """
    cleaned_pages = []
    for page in pages:
        cleaned_page = page.copy()
        cleaned_page["text"] = clean_text(page.get("text", ""))
        cleaned_pages.append(cleaned_page)

    # Thống kê
    total_chars = sum(len(p["text"]) for p in cleaned_pages)
    non_empty = sum(1 for p in cleaned_pages if p["text"].strip())
    print(
        f"[Text Cleaner] Đã làm sạch {len(cleaned_pages)} trang | "
        f"{non_empty} trang có nội dung | "
        f"Tổng {total_chars:,} ký tự"
    )

    return cleaned_pages


# ---------- Chuẩn hóa slide bài giảng (PDF xuất từ PowerPoint) ----------

_PAGE_NUMBER_LINE = re.compile(r"^\d{1,3}$")


def _content_lines(text: str) -> List[str]:
    """Các dòng có nội dung, bỏ dòng chỉ chứa số trang slide."""
    return [l for l in text.split("\n") if l.strip() and not _PAGE_NUMBER_LINE.match(l.strip())]


def _joined(lines: List[str]) -> str:
    return " ".join(" ".join(lines).split())


def normalize_slide_pages(pages: List[dict], min_chars: int = 40) -> List[dict]:
    """
    Giảm nhiễu đặc thù của slide trước khi chunking:

    1. Trang hiệu ứng (animation build): nội dung trang là phần đầu của trang kế tiếp
       → bỏ, vì trang kế tiếp đã chứa đủ nội dung.
    2. Tiêu đề lặp lại: các dòng đầu trùng với slide liền trước (vd "Recurrent Neural
       Network là gì?" ở 9 slide liên tiếp) chỉ giữ ở slide đầu nhóm; các slide sau
       bỏ đi để không slide nào "giống câu hỏi" chỉ vì chung tiêu đề.
    3. Trang quá ngắn (< min_chars ký tự sau bước 2) được gộp vào slide cùng nhóm
       (hoặc slide kế tiếp nếu là slide tiêu đề mở đầu một phần mới).

    Returns:
        Danh sách trang {"page_number", "text", "header", "source_pages"}.
        header: tiêu đề nhóm slide; source_pages: các trang gốc đã gộp vào trang này.
    """
    items = []
    for p in pages:
        lines = _content_lines(p.get("text", ""))
        if lines:
            items.append({"page_number": p["page_number"], "lines": lines,
                          "source_pages": [p["page_number"]]})

    # 1. Bỏ trang hiệu ứng, chuyển số trang gốc sang trang đầy đủ phía sau
    deduped = []
    for i, item in enumerate(items):
        nxt = items[i + 1] if i + 1 < len(items) else None
        if nxt and _joined(nxt["lines"]).startswith(_joined(item["lines"])):
            nxt["source_pages"] = item["source_pages"] + nxt["source_pages"]
            continue
        deduped.append(item)

    # 2. Bỏ các dòng đầu trùng với slide liền trước; đánh số nhóm slide
    prev_lines, group, header = None, -1, ""
    for item in deduped:
        lines = item["lines"]
        k = 0
        if prev_lines is not None:
            while k < min(len(lines), len(prev_lines)) and lines[k] == prev_lines[k]:
                k += 1
        if k == 0:
            group += 1
            header = lines[0]
        item.update(body=lines[k:], group=group, header=header, starts_group=(k == 0))
        prev_lines = lines

    # 3. Gộp trang quá ngắn
    result, pending = [], None
    for i, item in enumerate(deduped):
        if pending:
            item["body"] = pending["body"] + item["body"]
            item["source_pages"] = pending["source_pages"] + item["source_pages"]
            pending = None
        if len(_joined(item["body"])) >= min_chars:
            result.append(item)
            continue
        nxt = deduped[i + 1] if i + 1 < len(deduped) else None
        if nxt and (nxt["group"] == item["group"] or item["starts_group"]):
            pending = item
        elif result and result[-1]["group"] == item["group"]:
            result[-1]["body"] += item["body"]
            result[-1]["source_pages"] += item["source_pages"]
        elif item["body"]:
            result.append(item)
    if pending and pending["body"]:
        result.append(pending)

    return [{"page_number": it["page_number"], "text": "\n".join(it["body"]),
             "header": it["header"], "source_pages": sorted(it["source_pages"])}
            for it in result]
