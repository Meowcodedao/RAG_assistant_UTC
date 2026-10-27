from app.services.chunking import chunk_documents, get_chunk_stats, detect_chapter_section
from app.services.text_cleaner import clean_text

def test_clean_text_preserves_vietnamese():
    assert "tiếng Việt" in clean_text("tiếng   Việt\n")

def test_fixed_chunking_overlap():
    pages=[{"page_number":1,"text":"A"*2500}]
    chunks=chunk_documents(pages,"fixed",500,100,"doc",{"document_title":"Test","source":"x.pdf"})
    assert len(chunks) >= 6
    assert all(c["strategy"]=="fixed" for c in chunks)
    assert all(c["char_count"] <= 500 for c in chunks)

def test_recursive_chunking_and_stats():
    pages=[{"page_number":1,"text":("Đây là một câu. " * 100)}]
    chunks=chunk_documents(pages,"recursive",200,30,"doc",{"document_title":"Test","source":"x.pdf"})
    assert chunks
    assert all(c["strategy"]=="recursive" for c in chunks)
    assert get_chunk_stats(chunks)["total_chunks"]==len(chunks)

def test_fixed_chunking_has_no_redundant_tail():
    pages=[{"page_number":1,"text":"A"*2500}]
    chunks=chunk_documents(pages,"fixed",500,100,"doc",{})
    # Các chunk bắt đầu tại 0, 400, ..., 2000; chunk tại 2000 đã chạm cuối văn bản
    assert len(chunks)==6

def test_fixed_chunking_page_mapping():
    pages=[{"page_number":1,"text":"a"*300},{"page_number":2,"text":""},{"page_number":3,"text":"b"*300}]
    chunks=chunk_documents(pages,"fixed",200,0,"doc",{})
    assert [c["page"] for c in chunks]==[1,1,3,3]

def test_recursive_chunking_respects_size_and_keeps_periods():
    pages=[{"page_number":1,"text":("Đây là một câu khá dài. " * 60).strip()}]
    chunks=chunk_documents(pages,"recursive",200,40,"doc",{})
    assert all(c["char_count"] <= 200 for c in chunks)
    assert all(c["text"].endswith(".") for c in chunks)
    # Overlap không cắt giữa từ: chunk sau bắt đầu bằng một từ trọn vẹn của chunk trước
    first_word=chunks[1]["text"].split()[0]
    assert first_word in chunks[0]["text"].split()

def test_recursive_overlap_stays_within_page():
    pages=[{"page_number":1,"text":"x "*150},{"page_number":2,"text":"y "*150}]
    chunks=chunk_documents(pages,"recursive",100,20,"doc",{})
    assert all("y" not in c["text"] for c in chunks if c["page"]==1)
    assert all("x" not in c["text"] for c in chunks if c["page"]==2)

def test_detect_chapter_uses_matching_prefix():
    assert detect_chapter_section("", "Bài 5\nChương 3 Tối ưu")[0]=="Bài 5"
    assert detect_chapter_section("Chương 3 Tối ưu", "Bài 5\nChương 3 Tối ưu")[0]=="Chương 3"

def test_detect_section_ignores_decimal_numbers():
    assert detect_chapter_section("", "beta = 0.9 hoặc 0.99.")[1]=="Chưa rõ"
    assert detect_chapter_section("", "0.5 44")[1]=="Chưa rõ"
    assert detect_chapter_section("", "2.1 Phương pháp SGD\nnội dung")[1]=="2.1 Phương pháp SGD"

def test_clean_text_hyphen_handling():
    assert clean_text("ma-\nchine")=="machine"
    assert clean_text("RNN-\nLSTM")=="RNN-\nLSTM"

def test_normalize_slide_pages():
    from app.services.text_cleaner import normalize_slide_pages
    pages=[
        {"page_number":1,"text":"RNN là gì?\nDeep learning có 2 mô hình lớn: CNN cho ảnh, RNN cho dữ liệu chuỗi.\n1"},
        {"page_number":2,"text":"RNN là gì?\nPhân loại bài toán RNN:"},
        {"page_number":3,"text":"RNN là gì?\nPhân loại bài toán RNN:\nOne to one: một input và một output, ví dụ phân loại ảnh."},
        {"page_number":4,"text":"Mô hình LSTM\nCell state giúp tránh hiện tượng triệt tiêu gradient trong RNN."},
    ]
    out=normalize_slide_pages(pages,min_chars=20)
    by_page={p["page_number"]:p for p in out}
    # Trang hiệu ứng 2 nằm trọn trong trang 3 → gộp vào trang 3
    assert sorted(by_page)==[1,3,4] and by_page[3]["source_pages"]==[2,3]
    # Tiêu đề lặp chỉ giữ ở slide đầu nhóm, nhưng vẫn lưu làm header
    assert by_page[1]["text"].startswith("RNN là gì?")
    assert not by_page[3]["text"].startswith("RNN là gì?") and by_page[3]["header"]=="RNN là gì?"
    # Dòng chỉ chứa số trang bị bỏ
    assert not by_page[1]["text"].endswith("1")

def test_chunks_carry_source_pages_and_header():
    pages=[{"page_number":5,"text":"Dropout\nNgẫu nhiên tắt một số nơ-ron khi huấn luyện để chống overfitting.","header":"Dropout","source_pages":[4,5]}]
    for strategy in ("fixed","recursive"):
        c=chunk_documents(pages,strategy,500,50,"doc",{})[0]
        assert c["pages"]==[4,5] and c["section"]=="Dropout"
