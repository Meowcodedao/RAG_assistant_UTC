"""
Module chunking - Chia văn bản thành các đoạn (chunk) với metadata.
Hỗ trợ 2 chiến lược bắt buộc:
  1. Fixed-size chunking: chia theo kích thước ký tự cố định, có overlap.
  2. Recursive chunking: chia theo ranh giới nội dung (đoạn → câu → ký tự).

ĐƠN VỊ ĐO: Tất cả kích thước chunk (chunk_size, chunk_overlap) đều tính bằng KÝ TỰ (characters).
"""

import re
from typing import Dict, List, Optional, Tuple


def detect_chapter_section(text: str, full_page_text: str = "") -> Tuple[str, str]:
    """
    Phát hiện chương và mục từ nội dung văn bản.

    Tìm các pattern phổ biến trong tài liệu tiếng Việt và tiếng Anh:
    - "Chương X", "CHƯƠNG X", "Chapter X"
    - "Phần X", "Bài X"
    - "X.Y Tiêu đề" (mục đánh số)

    Args:
        text: Nội dung chunk.
        full_page_text: Toàn bộ nội dung trang chứa chunk.

    Returns:
        Tuple (chapter, section). Giá trị "Chưa rõ" nếu không xác định.
    """
    chapter = "Chưa rõ"
    section = "Chưa rõ"

    # Ưu tiên tiêu đề nằm ngay trong chunk, sau đó mới tìm trong cả trang
    search_texts = [t for t in (text, full_page_text) if t]

    # Chương: chỉ nhận khi từ khóa đứng đầu dòng (tránh "phần 2 của ..." giữa câu).
    # Tiền tố và số được lấy từ CÙNG một match.
    chapter_labels = {
        "chương": "Chương", "chapter": "Chương",
        "phần": "Phần", "part": "Phần",
        "bài": "Bài", "lesson": "Bài",
    }
    chapter_re = re.compile(
        r"^\s*(Chương|CHƯƠNG|Chapter|CHAPTER|Phần|PHẦN|Part|PART|Bài|BÀI|Lesson)\s+(\d+)\b",
        re.MULTILINE,
    )
    for search_text in search_texts:
        match = chapter_re.search(search_text)
        if match:
            chapter = f"{chapter_labels[match.group(1).lower()]} {match.group(2)}"
            break

    # Mục: dòng bắt đầu bằng số mục (vd: "2.1", "2.1.3") rồi tới tiêu đề bắt đầu bằng chữ.
    # Không nhận số thập phân như "0.9 hoặc 0.99" hay "0.5 44".
    section_re = re.compile(
        r"^\s*([1-9]\d?(?:\.\d{1,2}){1,2})\.?\s*[\-:]?\s+([^\W\d_][^\n]*)$",
        re.MULTILINE,
    )
    for search_text in search_texts:
        match = section_re.search(search_text)
        if match:
            section_title = match.group(2).strip()[:60]  # Giới hạn độ dài
            section = f"{match.group(1)} {section_title}"
            break

    return chapter, section


def _build_page_text_map(pages: List[dict]) -> Dict[int, str]:
    """Tạo mapping page_number → text để tra cứu nhanh."""
    return {p["page_number"]: p["text"] for p in pages}


def fixed_size_chunking(
    pages: List[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    document_id: str = "",
    metadata: Optional[dict] = None,
) -> List[dict]:
    """
    Chiến lược 1: Chia văn bản theo kích thước cố định (tính bằng KÝ TỰ).

    Ghép toàn bộ văn bản với dấu hiệu phân trang, sau đó cắt thành các
    đoạn có độ dài chunk_size ký tự, với phần overlap giữa các đoạn kế tiếp.

    Args:
        pages: Danh sách trang đã làm sạch, mỗi phần tử có "page_number" và "text".
        chunk_size: Kích thước mỗi chunk (đơn vị: ký tự). Mặc định 500.
        chunk_overlap: Số ký tự chồng lấp giữa hai chunk liên tiếp. Mặc định 100.
        document_id: Mã tài liệu (vd: "doc01").
        metadata: Dict chứa thông tin tài liệu (document_title, author, year, course, source, source_url).

    Returns:
        Danh sách dict chunk theo schema ChunkItem.
    """
    if metadata is None:
        metadata = {}

    # Xây dựng văn bản liên tục với marker trang
    # Dùng marker để theo dõi chunk thuộc trang nào
    PAGE_MARKER = "\n<<<PAGE_{page_num}>>>\n"
    full_text = ""
    page_boundaries = []  # [(start_pos, page_number), ...]

    for page in pages:
        if not page["text"].strip():
            continue
        start_pos = len(full_text)
        marker = PAGE_MARKER.format(page_num=page["page_number"])
        full_text += marker + page["text"]
        page_boundaries.append((start_pos, page["page_number"]))

    if not full_text.strip():
        return []

    # Loại bỏ marker khỏi text nhưng giữ mapping vị trí
    # Tạo clean text và bảng ánh xạ vị trí
    clean_text = re.sub(r"\n<<<PAGE_\d+>>>\n", "\n", full_text)
    clean_text = clean_text.strip()

    # Tìm trang tương ứng cho mỗi vị trí ký tự
    def _get_page_for_position(pos: int) -> int:
        """Xác định trang chứa vị trí pos trong clean_text."""
        # Ước lượng: dựa trên tỷ lệ vị trí
        accumulated = 0
        for page in pages:
            page_len = len(page["text"].strip())
            if page_len == 0:
                continue
            if accumulated + page_len > pos:
                return page["page_number"]
            accumulated += page_len + 1  # +1 cho newline
        # Fallback: trang cuối cùng
        return pages[-1]["page_number"] if pages else 1

    # Chia chunk
    chunks = []
    chunk_idx = 0
    start = 0

    while start < len(clean_text):
        end = min(start + chunk_size, len(clean_text))
        chunk_text = clean_text[start:end].strip()

        if not chunk_text:
            start = end
            continue

        page_num = _get_page_for_position(start)
        page = next((p for p in pages if p["page_number"] == page_num), {})
        chapter, section = detect_chapter_section(chunk_text, page.get("text", ""))
        if page.get("header"):
            section = page["header"]

        chunk_id = f"{document_id}_p{page_num:03d}_c{chunk_idx:03d}"

        chunk = {
            "chunk_id": chunk_id,
            "text": chunk_text,
            "document_id": document_id,
            "document_title": metadata.get("document_title", metadata.get("title", "Chưa rõ")),
            "author": metadata.get("author", "Chưa rõ"),
            "year": metadata.get("year", None),
            "course": metadata.get("course", "Chưa rõ"),
            "source": metadata.get("source", ""),
            "page": page_num,
            "pages": page.get("source_pages", [page_num]),
            "chapter": chapter,
            "section": section,
            "source_url": metadata.get("source_url", None),
            "strategy": "fixed",
            "char_count": len(chunk_text),
        }
        chunks.append(chunk)
        chunk_idx += 1

        # Đã chạm cuối văn bản: dừng, tránh sinh chunk thừa nằm trọn trong chunk trước
        if end >= len(clean_text):
            break

        # Di chuyển: bước tiến = chunk_size - overlap
        step = chunk_size - chunk_overlap
        if step <= 0:
            step = chunk_size  # Tránh vòng lặp vô hạn
        start += step

    print(
        f"[Chunking - Fixed] Tạo {len(chunks)} chunks | "
        f"chunk_size={chunk_size} ký tự | overlap={chunk_overlap} ký tự"
    )
    return chunks


def _recursive_split(text: str, separators: List[str], chunk_size: int) -> List[str]:
    """
    Chia văn bản đệ quy theo danh sách separator từ lớn đến nhỏ.
    Khi một đoạn nhỏ hơn chunk_size, giữ nguyên.
    Khi lớn hơn, thử separator tiếp theo.
    """
    if not text.strip():
        return []

    if len(text) <= chunk_size:
        return [text]

    if not separators:
        # Hết separator: cắt cứng theo chunk_size
        parts = []
        for i in range(0, len(text), chunk_size):
            part = text[i:i + chunk_size].strip()
            if part:
                parts.append(part)
        return parts

    sep = separators[0]
    remaining_seps = separators[1:]

    # Chia theo separator hiện tại
    splits = text.split(sep)
    if len(splits) <= 1:
        # Separator không tồn tại trong text → thử separator tiếp theo
        return _recursive_split(text, remaining_seps, chunk_size)

    # Separator có ký tự nhìn thấy (vd ". ") → giữ lại dấu câu ở cuối mỗi phần,
    # chỉ dùng phần khoảng trắng để ghép, tránh mất dấu chấm ở cuối chunk.
    kept = sep.strip()
    if kept:
        splits = [s + kept for s in splits[:-1]] + [splits[-1]]
        sep = sep[sep.index(kept) + len(kept):] or " "

    # Ghép các phần nhỏ lại thành chunk
    result = []
    current = ""

    for part in splits:
        part = part.strip()
        if not part:
            continue

        candidate = (current + sep + part).strip() if current else part

        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                result.append(current)
            # Nếu part đơn lẻ vẫn lớn hơn chunk_size → chia đệ quy
            if len(part) > chunk_size:
                sub_parts = _recursive_split(part, remaining_seps, chunk_size)
                result.extend(sub_parts)
                current = ""
            else:
                current = part

    if current:
        result.append(current)

    return result


def _tail_on_word_boundary(text: str, max_chars: int) -> str:
    """Lấy tối đa max_chars ký tự cuối của text, không cắt giữa một từ."""
    if len(text) <= max_chars:
        return text
    tail = text[-max_chars:]
    if not text[-max_chars - 1].isspace():
        # Ký tự đầu của tail nằm giữa một từ → bỏ phần từ bị cắt dở
        space = tail.find(" ")
        tail = tail[space + 1:] if space != -1 else ""
    return tail.strip()


def recursive_chunking(
    pages: List[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    document_id: str = "",
    metadata: Optional[dict] = None,
) -> List[dict]:
    """
    Chiến lược 2: Chia văn bản đệ quy theo ranh giới nội dung.

    Thứ tự ưu tiên separator:
    1. "\\n\\n" — ranh giới đoạn văn
    2. "\\n"   — ranh giới dòng
    3. ". "   — ranh giới câu
    4. " "    — ranh giới từ
    5. cắt cứng theo số ký tự (khi hết separator)

    Sau khi chia, áp dụng overlap bằng cách thêm phần cuối chunk trước
    (cắt theo ranh giới từ) vào đầu chunk sau. Mỗi đoạn được chia với
    kích thước chunk_size - chunk_overlap nên chunk cuối cùng không vượt chunk_size.

    Args:
        pages: Danh sách trang đã làm sạch.
        chunk_size: Kích thước mục tiêu (đơn vị: ký tự). Mặc định 500.
        chunk_overlap: Số ký tự chồng lấp. Mặc định 100.
        document_id: Mã tài liệu.
        metadata: Dict chứa thông tin tài liệu.

    Returns:
        Danh sách dict chunk theo schema ChunkItem.
    """
    if metadata is None:
        metadata = {}

    separators = ["\n\n", "\n", ". ", " "]

    # Chừa chỗ cho phần overlap để chunk sau khi ghép không vượt chunk_size
    split_size = chunk_size - chunk_overlap - 1 if 0 < chunk_overlap < chunk_size - 1 else chunk_size

    all_chunks = []
    chunk_idx = 0

    for page in pages:
        page_text = page["text"].strip()
        if not page_text:
            continue

        page_num = page["page_number"]

        # Chia đệ quy
        raw_parts = _recursive_split(page_text, separators, split_size)

        # Áp dụng overlap (trong phạm vi một trang, để số trang trích dẫn luôn đúng)
        parts_with_overlap = []
        for i, part in enumerate(raw_parts):
            if i > 0 and split_size < chunk_size:
                overlap_text = _tail_on_word_boundary(raw_parts[i - 1], chunk_overlap)
                if overlap_text:
                    part = overlap_text + " " + part
            parts_with_overlap.append(part)

        for part in parts_with_overlap:
            part = part.strip()
            if not part:
                continue

            chapter, section = detect_chapter_section(part, page_text)
            if page.get("header"):
                section = page["header"]
            chunk_id = f"{document_id}_p{page_num:03d}_c{chunk_idx:03d}"

            chunk = {
                "chunk_id": chunk_id,
                "text": part,
                "document_id": document_id,
                "document_title": metadata.get("document_title", metadata.get("title", "Chưa rõ")),
                "author": metadata.get("author", "Chưa rõ"),
                "year": metadata.get("year", None),
                "course": metadata.get("course", "Chưa rõ"),
                "source": metadata.get("source", ""),
                "page": page_num,
                "pages": page.get("source_pages", [page_num]),
                "chapter": chapter,
                "section": section,
                "source_url": metadata.get("source_url", None),
                "strategy": "recursive",
                "char_count": len(part),
            }
            all_chunks.append(chunk)
            chunk_idx += 1

    print(
        f"[Chunking - Recursive] Tạo {len(all_chunks)} chunks | "
        f"chunk_size={chunk_size} ký tự | overlap={chunk_overlap} ký tự"
    )
    return all_chunks


def chunk_documents(
    pages: List[dict],
    strategy: str = "recursive",
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    document_id: str = "",
    metadata: Optional[dict] = None,
) -> List[dict]:
    """
    Hàm chính để chunk tài liệu với chiến lược được chọn.

    Args:
        pages: Danh sách trang đã làm sạch.
        strategy: "fixed" hoặc "recursive".
        chunk_size: Kích thước chunk (ký tự).
        chunk_overlap: Kích thước overlap (ký tự).
        document_id: Mã tài liệu.
        metadata: Thông tin metadata tài liệu.

    Returns:
        Danh sách chunk dạng dict.

    Raises:
        ValueError: Nếu strategy không hợp lệ.
    """
    strategy = strategy.lower().strip()

    if strategy == "fixed":
        return fixed_size_chunking(pages, chunk_size, chunk_overlap, document_id, metadata)
    elif strategy == "recursive":
        return recursive_chunking(pages, chunk_size, chunk_overlap, document_id, metadata)
    else:
        raise ValueError(
            f"Chiến lược chunking không hợp lệ: '{strategy}'. "
            f"Chỉ hỗ trợ: 'fixed', 'recursive'."
        )


def get_chunk_stats(chunks: List[dict]) -> dict:
    """
    Tính thống kê cho danh sách chunk.

    Returns:
        Dict với: total_chunks, avg_length, min_length, max_length, strategy.
    """
    if not chunks:
        return {
            "total_chunks": 0,
            "avg_length": 0,
            "min_length": 0,
            "max_length": 0,
            "strategy": "unknown",
        }

    lengths = [c["char_count"] for c in chunks]
    return {
        "total_chunks": len(chunks),
        "avg_length": round(sum(lengths) / len(lengths), 1),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "strategy": chunks[0].get("strategy", "unknown"),
    }
