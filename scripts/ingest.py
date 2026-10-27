"""Ingest PDFs from data/pdfs, clean text, chunk it and persist chunks.json."""
import json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.config import settings
from app.services.corpus import load_cleaned_documents, build_chunks
from app.services.chunking import get_chunk_stats

def main():
    settings.ensure_directories()
    errors=[]
    documents=load_cleaned_documents(settings.PDF_DIR,errors)
    for e in errors:
        print(f"[Ingest] LỖI {e['file']}: {e['error']}",file=sys.stderr)
    if not documents:
        # Không ghi đè chunks.json cũ bằng kho rỗng
        print("[Ingest] Không nạp được tài liệu nào; giữ nguyên dữ liệu đã xử lý trước đó.",file=sys.stderr)
        return 1
    all_chunks,docs=build_chunks(documents,settings.DEFAULT_CHUNK_STRATEGY,
                                 settings.CHUNK_SIZE,settings.CHUNK_OVERLAP,
                                 settings.SLIDE_CLEANUP,settings.MIN_PAGE_CHARS)
    (settings.PROCESSED_DIR/"chunks.json").write_text(json.dumps(all_chunks,ensure_ascii=False,indent=2),encoding="utf-8")
    (settings.PROCESSED_DIR/"documents.json").write_text(json.dumps(docs,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"\nHoàn tất: {len(docs)} tài liệu, {len(all_chunks)} chunks.")
    print("Thống kê:", get_chunk_stats(all_chunks))
    print("Đã lưu:", settings.PROCESSED_DIR/"chunks.json")
    if errors:
        print(f"[Ingest] Cảnh báo: {len(errors)} tệp không nạp được (xem stderr).",file=sys.stderr)
    return 0

if __name__=="__main__":
    sys.exit(main())
