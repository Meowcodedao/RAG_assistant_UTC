"""
Module quản lý cấu hình tập trung cho toàn bộ hệ thống Hybrid RAG Assistant.
Đọc cấu hình từ file .env hoặc biến môi trường với giá trị mặc định chuẩn xác.
"""

import os
from pathlib import Path

# Cố gắng nạp python-dotenv nếu có, nếu không thì dùng biến môi trường thuần
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Định vị thư mục gốc dự án
BASE_DIR = Path(__file__).resolve().parent.parent

class Settings:
    BASE_DIR: Path = BASE_DIR

    # 1. API Keys & LLM Provider
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "mock").lower()  # "mock", "gemini", "openai", "ollama"
    # Để trống thì dùng model mặc định của provider (xem DEFAULT_LLM_MODELS)
    LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "").strip()
    DEFAULT_LLM_MODELS = {"openai": "gpt-4o-mini", "gemini": "gemini-2.5-flash",
                          "ollama": os.getenv("OLLAMA_MODEL", "qwen2.5:3b")}
    # Ollama chạy cục bộ, không cần API key
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    LLM_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", "180"))

    # 2. Embedding Model
    EMBEDDING_MODEL_NAME: str = os.getenv(
        "EMBEDDING_MODEL_NAME",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # 3. Chunking Settings
    DEFAULT_CHUNK_STRATEGY: str = os.getenv("DEFAULT_CHUNK_STRATEGY", "recursive")
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    # Chuẩn hóa slide: bỏ trang hiệu ứng, tiêu đề lặp lại, gộp trang quá ngắn
    SLIDE_CLEANUP: bool = os.getenv("SLIDE_CLEANUP", "true").lower() in ("1", "true", "yes")
    MIN_PAGE_CHARS: int = int(os.getenv("MIN_PAGE_CHARS", "40"))
    # Đơn vị đo kích thước chunk: "characters" (ký tự)
    CHUNK_UNIT: str = "characters"

    # 4. Retrieval & RRF Settings
    FAISS_TOP_K: int = int(os.getenv("FAISS_TOP_K", "8"))
    BM25_TOP_K: int = int(os.getenv("BM25_TOP_K", "8"))
    FINAL_TOP_K: int = int(os.getenv("FINAL_TOP_K", "5"))
    RRF_K: int = int(os.getenv("RRF_K", "60"))  # Hằng số làm mượt RRF theo đề bài

    # 5. Đường dẫn thư mục dữ liệu
    PDF_DIR: Path = BASE_DIR / os.getenv("PDF_DIR", "data/pdfs")
    PROCESSED_DIR: Path = BASE_DIR / os.getenv("PROCESSED_DIR", "data/processed")
    INDEX_DIR: Path = BASE_DIR / os.getenv("INDEX_DIR", "data/indexes")
    EVALUATION_DIR: Path = BASE_DIR / os.getenv("EVALUATION_DIR", "data/evaluation")
    REPORTS_DIR: Path = BASE_DIR / "reports"

    # 6. Server Settings
    API_HOST: str = os.getenv("API_HOST", "127.0.0.1")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    UI_HOST: str = os.getenv("UI_HOST", "127.0.0.1")
    UI_PORT: int = int(os.getenv("UI_PORT", "7860"))
    MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "50"))

    @classmethod
    def llm_model_for(cls, provider: str) -> str:
        """Model LLM cho provider; LLM_MODEL_NAME (nếu có) luôn được ưu tiên."""
        return cls.LLM_MODEL_NAME or cls.DEFAULT_LLM_MODELS.get(provider, "")

    @classmethod
    def ensure_directories(cls):
        """Đảm bảo các thư mục dữ liệu luôn tồn tại."""
        for path in [cls.PDF_DIR, cls.PROCESSED_DIR, cls.INDEX_DIR, cls.EVALUATION_DIR, cls.REPORTS_DIR]:
            path.mkdir(parents=True, exist_ok=True)

settings = Settings()
settings.ensure_directories()
