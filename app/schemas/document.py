"""
Các Pydantic schema định nghĩa cấu trúc dữ liệu chuẩn hóa cho toàn bộ hệ thống RAG:
Document, Chunk, Search Result, Citation, RAG Response và Evaluation.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class DocumentMetadata(BaseModel):
    """Metadata cấp tài liệu PDF."""
    document_id: str = Field(..., description="Mã định danh duy nhất của tài liệu (vd: doc01)")
    document_title: str = Field(..., description="Tên giáo trình / bài giảng")
    author: Optional[str] = Field(default="Chưa rõ", description="Tác giả hoặc cơ sở đào tạo")
    year: Optional[int] = Field(default=None, description="Năm xuất bản")
    course: Optional[str] = Field(default="Chưa rõ", description="Học phần tương ứng")
    source: str = Field(..., description="Tên file gốc (vd: csdl.pdf)")
    source_url: Optional[str] = Field(default=None, description="Đường link công bố nếu có")
    total_pages: int = Field(default=0, description="Tổng số trang của PDF")

class ChunkItem(BaseModel):
    """Schema chuẩn hóa cho từng đoạn văn bản (chunk)."""
    chunk_id: str = Field(..., description="Định danh duy nhất: <document_id>_p<trang 3 chữ số>_c<số thứ tự 3 chữ số>, vd: bai5_p005_c012")
    text: str = Field(..., description="Nội dung đã làm sạch của chunk")
    document_id: str = Field(..., description="Mã tài liệu")
    document_title: str = Field(..., description="Tên tài liệu")
    author: Optional[str] = Field(default="Chưa rõ", description="Tác giả")
    year: Optional[int] = Field(default=None, description="Năm xuất bản")
    course: Optional[str] = Field(default="Chưa rõ", description="Học phần")
    source: str = Field(..., description="Tên file PDF nguồn")
    page: int = Field(..., description="Số trang vật lý trong tài liệu")
    pages: List[int] = Field(default=[], description="Các trang gốc gộp vào chunk này (slide hiệu ứng, trang quá ngắn)")
    chapter: Optional[str] = Field(default="Chưa rõ", description="Tên hoặc số thứ tự chương")
    section: Optional[str] = Field(default="Chưa rõ", description="Tên hoặc số thứ tự mục")
    source_url: Optional[str] = Field(default=None, description="Nguồn tải trực tuyến")
    strategy: str = Field(..., description="Chiến lược chunking: 'fixed' hoặc 'recursive'")
    char_count: int = Field(default=0, description="Số lượng ký tự trong chunk")

class SearchResultItem(BaseModel):
    """Schema biểu diễn kết quả truy xuất từ FAISS, BM25 hoặc RRF."""
    chunk: ChunkItem
    faiss_rank: Optional[int] = Field(default=None, description="Thứ hạng trong kết quả FAISS (1-based)")
    faiss_score: Optional[float] = Field(default=None, description="Điểm tương đồng cosine (FAISS) hoặc TF-IDF khi dùng fallback")
    bm25_rank: Optional[int] = Field(default=None, description="Thứ hạng trong kết quả BM25 (1-based)")
    bm25_score: Optional[float] = Field(default=None, description="Điểm BM25")
    rrf_score: Optional[float] = Field(default=None, description="Điểm Reciprocal Rank Fusion tổng hợp (chỉ có ở chế độ hybrid)")
    final_rank: int = Field(..., description="Thứ hạng tổng kết sau truy xuất (1-based)")

class CitationItem(BaseModel):
    """Thông tin trích dẫn minh bạch phục vụ câu trả lời."""
    document_title: str
    course: Optional[str] = None
    page: int
    chapter: Optional[str] = None
    section: Optional[str] = None
    snippet: str = Field(..., description="Trích đoạn ngắn làm căn cứ trực tiếp")

class RAGResponse(BaseModel):
    """Schema câu trả lời sinh bởi RAG pipeline."""
    question: str
    answer: str
    citations: List[CitationItem] = []
    retrieved_chunks: List[SearchResultItem] = []
    retrieval_method: str = Field(..., description="'hybrid', 'semantic' hoặc 'keyword'")
    chunk_strategy: str = Field(..., description="'fixed' hoặc 'recursive'")
    latency_seconds: float = 0.0

class EvaluationQueryItem(BaseModel):
    """Schema bộ câu hỏi benchmark kiểm thử chất lượng."""
    query_id: str
    question: str
    query_type: str = Field(..., description="keyword | semantic | multi_hop")
    ground_truth_source: str
    ground_truth_page: int
    ground_truth_pages: List[int] = Field(default=[], description="Các trang cùng chứa đáp án (nếu nhiều hơn một)")
    ground_truth_section: Optional[str] = None
    relevant_chunk_ids: List[str] = []
    expected_answer: str
