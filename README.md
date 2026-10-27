# Hybrid RAG Assistant — Hỏi đáp trên tài liệu PDF

Dự án Python hỗ trợ hỏi đáp trên bài giảng PDF bằng quy trình Retrieval-Augmented Generation (RAG): trích xuất PDF, làm sạch, chuẩn hóa slide, chunking, tìm kiếm ngữ nghĩa (FAISS), BM25, hợp nhất thứ hạng RRF, sinh câu trả lời bằng LLM (Ollama / OpenAI / Gemini), API FastAPI, giao diện Gradio và công cụ đánh giá/thử nghiệm retrieval.

## Mục lục

1. [Yêu cầu hệ thống](#1-yêu-cầu-hệ-thống)
2. [Cài đặt trên Windows](#2-cài-đặt-trên-windows)
3. [Cài đặt trên Linux](#3-cài-đặt-trên-linux)
4. [Cài Ollama để chạy LLM cục bộ (tùy chọn)](#4-cài-ollama-để-chạy-llm-cục-bộ-tùy-chọn)
5. [Chạy dự án](#5-chạy-dự-án)
6. [Cấu hình LLM](#6-cấu-hình-llm)
7. [Đánh giá và thử nghiệm](#7-đánh-giá-và-thử-nghiệm)
8. [Cấu trúc dự án](#8-cấu-trúc-dự-án)
9. [Xử lý sự cố](#9-xử-lý-sự-cố)

---

## 1. Yêu cầu hệ thống

| Thành phần | Yêu cầu |
|---|---|
| Python | 3.10 – 3.14 (khuyên dùng **3.11 hoặc 3.12** trên Windows để có sẵn bản cài `faiss-cpu`) |
| RAM | Tối thiểu 4 GB; 8 GB nếu chạy thêm Ollama với model 3B |
| Ổ đĩa | ~3 GB cho thư viện (PyTorch) + ~500 MB cho embedding model + 1–2 GB cho mỗi model Ollama |
| Internet | Cần ở lần chạy đầu để tải thư viện và embedding model; sau đó chạy được offline (với Ollama hoặc chế độ mock) |
| GPU | Không bắt buộc. Hướng dẫn dưới đây cài PyTorch bản CPU để tiết kiệm dung lượng |

---

## 2. Cài đặt trên Windows

### Bước 1 — Cài Python

1. Tải Python 3.12 tại <https://www.python.org/downloads/windows/>.
2. Khi cài, **tick "Add python.exe to PATH"**.
3. Kiểm tra trong PowerShell:
   ```powershell
   py --version
   ```

### Bước 2 — Tạo môi trường ảo và cài thư viện

Mở **PowerShell** tại thư mục dự án `Chuyende`:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# PyTorch bản CPU (nhẹ hơn nhiều so với bản mặc định có CUDA)
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
Copy-Item .env.example .env
```

Nếu PowerShell báo *"running scripts is disabled on this system"* khi kích hoạt môi trường, chạy một lần:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Hoặc dùng trực tiếp Python trong môi trường ảo mà không cần kích hoạt: `.\.venv\Scripts\python.exe <lệnh>`.

### Bước 3 — Tạo dữ liệu và chạy

```powershell
python scripts/ingest.py
python -m app.frontend.gradio_app
```

Mở <http://127.0.0.1:7860> trên trình duyệt.

> Những lần sau chỉ cần: mở PowerShell tại thư mục dự án → `.\.venv\Scripts\Activate.ps1` → `python -m app.frontend.gradio_app`.

---

## 3. Cài đặt trên Linux

### Bước 1 — Cài Python và công cụ venv

| Bản phân phối | Lệnh |
|---|---|
| Ubuntu / Debian | `sudo apt update && sudo apt install python3 python3-venv python3-pip` |
| Fedora | `sudo dnf install python3 python3-pip` |
| Arch / Manjaro / Omarchy | `sudo pacman -S python` (có sẵn `venv`; **không** cần cài pip hệ thống) |

> Các bản Linux mới chặn `pip install` vào Python hệ thống (lỗi `externally-managed-environment`). Luôn cài thư viện **trong môi trường ảo** như dưới đây.

### Bước 2 — Tạo môi trường ảo và cài thư viện

```bash
cd Chuyende
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# PyTorch bản CPU (nhẹ hơn nhiều so với bản mặc định có CUDA)
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
cp .env.example .env
```

### Bước 3 — Tạo dữ liệu và chạy

```bash
python scripts/ingest.py
python -m app.frontend.gradio_app
```

Mở <http://127.0.0.1:7860> trên trình duyệt.

> Những lần sau chỉ cần: `cd Chuyende && source .venv/bin/activate && python -m app.frontend.gradio_app`.

---

## 4. Cài Ollama để chạy LLM cục bộ (tùy chọn)

Ollama chạy mô hình ngôn ngữ ngay trên máy: miễn phí, không cần API key, không gửi dữ liệu ra ngoài. Nếu bỏ qua bước này, hệ thống vẫn chạy ở chế độ `mock` (chỉ hiển thị đoạn bằng chứng, không tổng hợp câu trả lời).

### Windows

1. Tải và cài **OllamaSetup.exe** tại <https://ollama.com/download/windows>. Ollama tự chạy nền sau khi cài.
2. Mở PowerShell, tải model:
   ```powershell
   ollama pull qwen2.5:3b
   ```

### Linux

```bash
# Ubuntu / Debian / Fedora / các bản khác
curl -fsSL https://ollama.com/install.sh | sh

# Arch / Omarchy (thay cho lệnh trên)
sudo pacman -S ollama
sudo systemctl enable --now ollama

ollama pull qwen2.5:3b
```

### Kiểm tra

```bash
ollama list                              # phải thấy qwen2.5:3b
curl http://localhost:11434/api/version  # Ollama đang chạy
```

### Chọn model

| Model | Dung lượng | Ghi chú |
|---|---|---|
| `qwen2.5:1.5b` | ~1 GB | Nhanh, phù hợp máy yếu; câu trả lời đơn giản |
| `qwen2.5:3b` | ~1.9 GB | **Mặc định**, cân bằng tốc độ/chất lượng, tiếng Việt khá |
| `qwen2.5:7b` | ~4.7 GB | Chất lượng tốt hơn, cần ≥ 8 GB RAM |

Trên CPU, model 3B mất khoảng 20–40 giây cho mỗi câu trả lời.

### Bật Ollama trong dự án

Cách 1 — trên giao diện: tab **Hỏi đáp** → mở **Cấu hình LLM** → Provider `ollama` → chọn model trong danh sách (tự lấy từ `ollama list`).

Cách 2 — mặc định cho cả API và giao diện, sửa `.env`:

```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen2.5:3b
```

---

## 5. Chạy dự án

Các lệnh dưới đây chạy trong môi trường ảo đã kích hoạt (giống nhau trên Windows và Linux).

### 5.1. Nạp tài liệu (ingest)

Hai PDF mẫu đã có trong `data/pdfs/`. Có thể chép thêm PDF vào thư mục này rồi chạy:

```bash
python scripts/ingest.py
```

Kết quả: `data/processed/chunks.json` và `data/processed/documents.json`. Pipeline gồm:

1. **Trích xuất** văn bản từng trang (PyMuPDF).
2. **Làm sạch**: chuẩn hóa Unicode NFC, khoảng trắng, từ bị ngắt dòng.
3. **Chuẩn hóa slide** (`SLIDE_CLEANUP=true`): bỏ trang hiệu ứng (animation) đã nằm trọn trong trang sau, bỏ tiêu đề lặp lại ở các slide liên tiếp (tiêu đề được lưu vào trường `section`), gộp trang quá ngắn (`MIN_PAGE_CHARS`). Mỗi chunk lưu danh sách trang gốc trong trường `pages`.
4. **Chunking**: mặc định `recursive`, 500 ký tự (giới hạn cứng, kể cả overlap), overlap 100 ký tự trong cùng một trang.

Nếu không nạp được PDF nào, script trả mã lỗi khác 0 và giữ nguyên dữ liệu cũ. API và Gradio tự nạp lại khi `chunks.json` thay đổi.

### 5.2. Giao diện Gradio

```bash
python -m app.frontend.gradio_app
```

Mặc định tại <http://127.0.0.1:7860> (đổi bằng `UI_HOST`, `UI_PORT` trong `.env`).

| Tab | Chức năng |
|---|---|
| **Hỏi đáp** | Hỏi theo hybrid / semantic / keyword; chọn LLM (mock, Ollama, OpenAI, Gemini) cho phiên hiện tại |
| **Nạp tài liệu** | Tải PDF mới và tạo lại chỉ mục |
| **Duyệt tài liệu** | Xem văn bản từng trang và các chunk tương ứng; tìm chunk theo từ khóa |
| **Bộ câu hỏi** | Thêm/sửa/xóa câu hỏi đánh giá, tick chunk chứa đáp án |
| **Đánh giá** | Chạy benchmark cho cả 3 phương pháp, xem top-5 của từng câu |
| **Thử nghiệm** | So sánh nhiều cấu hình (chuẩn hóa slide, chunking, chunk size, overlap, embedding model, RRF_K) |

### 5.3. REST API

```bash
uvicorn app.api.main:app --host 127.0.0.1 --port 8000 --reload
```

Tài liệu API tương tác: <http://127.0.0.1:8000/docs>.

| Endpoint | Mô tả |
|---|---|
| `GET /health` | Trạng thái hệ thống |
| `GET /documents` | Metadata tài liệu đã ingest |
| `POST /ask` | Hỏi đáp (`hybrid`, `semantic`, `keyword`), dùng LLM cấu hình trong `.env` |
| `POST /ingest` | Tải PDF lên và xử lý lại kho. PDF trùng nội dung không nạp lại; PDF scan bị từ chối (400); giới hạn `MAX_UPLOAD_MB` |

Ví dụ:

```bash
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "Vanishing gradient ảnh hưởng RNN thế nào?", "retrieval_method": "hybrid", "top_k": 5}'
```

---

## 6. Cấu hình LLM

Mọi cấu hình nằm trong `.env` (tạo từ `.env.example`). **Không** đưa `.env` chứa API key lên Git.

| Provider | Cấu hình `.env` | Model mặc định |
|---|---|---|
| `mock` | `LLM_PROVIDER=mock` | — (chỉ hiển thị đoạn bằng chứng) |
| `ollama` | `LLM_PROVIDER=ollama`, `OLLAMA_MODEL=...`, `OLLAMA_BASE_URL=http://localhost:11434` | `qwen2.5:3b` |
| `openai` | `LLM_PROVIDER=openai`, `OPENAI_API_KEY=...` | `gpt-4o-mini` |
| `gemini` | `LLM_PROVIDER=gemini`, `GEMINI_API_KEY=...` | `gemini-2.5-flash` |

- `LLM_MODEL_NAME` để trống thì dùng model mặc định của provider; nếu điền, model phải thuộc đúng provider.
- `LLM_TIMEOUT` (giây, mặc định 180): thời gian chờ tối đa cho một câu trả lời.
- Khi gọi LLM lỗi (sai key, sai model, Ollama chưa chạy, mất mạng), hệ thống chuyển về chế độ offline và ghi rõ lý do ở cuối câu trả lời.

---

## 7. Đánh giá và thử nghiệm

### Đánh giá retrieval

```bash
python scripts/evaluate.py
```

Hoặc tab **Đánh giá** trên giao diện. Báo cáo tại `reports/retrieval_evaluation.json`: Recall@1/3/5 và MRR cho `hybrid`, `semantic`, `keyword`, kèm kết quả theo loại câu hỏi.

Ground truth:
- **Mức chunk**: `relevant_chunk_ids` gán ở tab **Bộ câu hỏi** (chỉ đúng với kho chunk lúc gán; nếu ingest lại với cấu hình khác, câu đó tự chuyển sang mức trang và giao diện cảnh báo).
- **Mức trang**: mọi chunk có trang gốc thuộc `ground_truth_pages`.

### Thử nghiệm nhiều cấu hình

Tab **Thử nghiệm** chạy lưới cấu hình, mỗi cấu hình dựng kho chunk trong bộ nhớ (không ghi đè dữ liệu hiện tại) và dùng ground truth mức trang để so sánh công bằng. Kết quả: `reports/experiments.json` và `reports/experiments.csv`. Để áp dụng cấu hình tốt nhất, sửa `.env` (`CHUNK_SIZE`, `CHUNK_OVERLAP`, `DEFAULT_CHUNK_STRATEGY`, `EMBEDDING_MODEL_NAME`, `RRF_K`, `SLIDE_CLEANUP`) rồi chạy lại `python scripts/ingest.py`.

> Model `bkai-foundation-models/vietnamese-bi-encoder` được huấn luyện trên văn bản đã tách từ tiếng Việt (vd "mạng_nơ_ron"); dự án chưa tách từ nên kết quả của model này có thể thấp hơn kỳ vọng.

### Kiểm thử

```bash
pytest -q
```

---

## 8. Cấu trúc dự án

```text
Chuyende/
├── app/
│   ├── api/main.py                  # REST API (FastAPI)
│   ├── config.py                    # Cấu hình từ .env
│   ├── frontend/gradio_app.py       # Giao diện web (Gradio)
│   ├── schemas/document.py          # Pydantic schemas
│   └── services/
│       ├── pdf_loader.py            # Trích xuất PDF theo trang
│       ├── text_cleaner.py          # Làm sạch tiếng Việt + chuẩn hóa slide
│       ├── chunking.py              # Fixed / Recursive chunking
│       ├── corpus.py                # Đọc + làm sạch + chunk toàn bộ PDF
│       ├── retrieval.py             # FAISS + BM25 + RRF
│       ├── rag.py                   # Pipeline hỏi đáp + gọi LLM
│       ├── ingestion.py             # Lưu PDF tải lên + chạy ingest
│       └── evaluation.py            # Recall@K, MRR, lưới thử nghiệm
├── data/
│   ├── pdfs/                        # Tài liệu PDF
│   ├── evaluation/queries.json      # Bộ câu hỏi đánh giá
│   ├── processed/                   # chunks.json, documents.json (sinh khi ingest)
│   └── indexes/
├── reports/                         # Báo cáo đánh giá / thử nghiệm
├── scripts/ingest.py
├── scripts/evaluate.py
├── tests/
├── .env.example
└── requirements.txt
```

---

## 9. Xử lý sự cố

| Lỗi | Nguyên nhân / cách xử lý |
|---|---|
| `pip: command not found` (Linux) | Không cần pip hệ thống. Tạo venv bằng `python3 -m venv .venv` rồi dùng `pip` bên trong venv |
| `externally-managed-environment` | Đang cài vào Python hệ thống. Kích hoạt venv trước (`source .venv/bin/activate`) |
| `running scripts is disabled` (Windows) | `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` |
| `py` không nhận lệnh (Windows) | Cài lại Python và tick "Add python.exe to PATH", hoặc dùng `python` thay cho `py` |
| Cài `faiss-cpu` thất bại | Phiên bản Python quá mới chưa có bản cài sẵn. Tạo venv bằng Python 3.11/3.12. Nếu thiếu FAISS, hệ thống tự dùng TF-IDF (kết quả kém hơn) |
| Câu trả lời có dòng "Cảnh báo: không gọi được LLM 'ollama'" | Ollama chưa chạy (`ollama serve` hoặc `systemctl start ollama`), hoặc chưa `ollama pull` model đã chọn |
| Danh sách model Ollama trống trên giao diện | Ollama chưa chạy lúc mở giao diện; bật Ollama rồi khởi động lại giao diện |
| Lần đầu chạy rất lâu / lỗi tải model | Đang tải embedding model (~500 MB) từ Hugging Face; cần Internet ở lần đầu. Đã tải rồi thì có thể đặt `HF_HUB_OFFLINE=1` để chạy offline |
| Tab Đánh giá báo "nhánh ngữ nghĩa: TF-IDF (fallback)" | Không tải được embedding model hoặc thiếu `sentence-transformers`/`faiss-cpu`; kiểm tra lại bước cài đặt |
| `Address already in use` / cổng 7860 bận | Đổi `UI_PORT` (hoặc `API_PORT`) trong `.env` |
| PDF tải lên bị từ chối "không có văn bản" | PDF dạng ảnh scan; cần OCR trước khi nạp |
