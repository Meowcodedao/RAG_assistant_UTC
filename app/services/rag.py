"""End-to-end RAG answer generation with source-grounded offline mode."""
import json, threading, time
from typing import List, Optional
from app.config import settings
from app.services.retrieval import HybridRetriever

def _mock_answer(question: str, results: List[dict], note: Optional[str] = None) -> str:
    if not results:
        return "Mình chưa tìm thấy đoạn tài liệu phù hợp để trả lời câu hỏi này. Hãy thử diễn đạt lại hoặc nạp thêm tài liệu."
    # The offline mode intentionally does not invent an answer; it surfaces evidence snippets.
    best = results[:3]
    lines = ["Dựa trên nội dung tìm được trong tài liệu:"]
    for r in best:
        c = r["chunk"]
        snippet = " ".join(c.get("text", "").split())
        header = c.get("section")
        if header and header != "Chưa rõ" and not snippet.startswith(header):
            snippet = f"[{header}] {snippet}"
        if len(snippet) > 700:
            snippet = snippet[:697].rsplit(" ", 1)[0] + "..."
        lines.append(f"- {snippet}")
    lines.append("\nLưu ý: chế độ offline hiển thị bằng chứng truy xuất, không sử dụng mô hình sinh để tổng hợp.")
    if note:
        lines.append(f"Cảnh báo: {note}")
    return "\n".join(lines)

SYSTEM_PROMPT = ("Bạn là trợ lý học tập. Chỉ trả lời dựa trên CONTEXT được cung cấp. "
                 "Nếu thiếu bằng chứng, nói rõ chưa đủ thông tin. Trả lời bằng tiếng Việt và không bịa nguồn.")

def _generate_answer(question: str, results: List[dict], provider: Optional[str] = None,
                     model: Optional[str] = None, api_key: Optional[str] = None) -> str:
    """provider/model/api_key ghi đè cấu hình .env cho riêng lượt hỏi này (dùng từ giao diện)."""
    provider = (provider or settings.LLM_PROVIDER).lower()
    if provider == "mock" or not results:
        return _mock_answer(question, results)
    context = "\n\n".join(
        f"[Nguồn: {r['chunk'].get('document_title')} - trang {r['chunk'].get('page')} - mục: {r['chunk'].get('section')}]\n{r['chunk'].get('text','')}"
        for r in results
    )
    model_name = (model or "").strip() or settings.llm_model_for(provider)
    try:
        if provider == "openai":
            key = api_key or settings.OPENAI_API_KEY
            if not key:
                raise RuntimeError("chưa cấu hình OPENAI_API_KEY")
            from openai import OpenAI
            client = OpenAI(api_key=key, timeout=settings.LLM_TIMEOUT)
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role":"system","content":SYSTEM_PROMPT},
                    {"role":"user","content":f"Câu hỏi: {question}\n\nCONTEXT:\n{context}"}
                ],
                temperature=0.2
            )
            text = resp.choices[0].message.content
        elif provider == "gemini":
            key = api_key or settings.GEMINI_API_KEY
            if not key:
                raise RuntimeError("chưa cấu hình GEMINI_API_KEY")
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=key)
            response = client.models.generate_content(
                model=model_name,
                contents=f"Câu hỏi: {question}\n\nCONTEXT:\n{context}",
                config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0.2),
            )
            text = response.text
        elif provider == "ollama":
            # Ollama cung cấp API tương thích OpenAI tại <OLLAMA_BASE_URL>/v1
            from openai import OpenAI
            client = OpenAI(base_url=f"{settings.OLLAMA_BASE_URL}/v1", api_key="ollama",
                            timeout=settings.LLM_TIMEOUT)
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role":"system","content":SYSTEM_PROMPT},
                    {"role":"user","content":f"Câu hỏi: {question}\n\nCONTEXT:\n{context}"}
                ],
                temperature=0.2
            )
            text = resp.choices[0].message.content
        else:
            raise RuntimeError(f"LLM_PROVIDER không hợp lệ: '{provider}' (chỉ hỗ trợ mock, openai, gemini, ollama)")
        if not text:
            raise RuntimeError("mô hình trả về câu trả lời rỗng")
        return text
    except Exception as exc:
        print(f"[RAG] {provider} ({model_name}) failed, using offline mode: {exc}")
        return _mock_answer(question, results,
                            note=f"không gọi được LLM '{provider}' ({model_name}): {exc}. Đang hiển thị chế độ offline.")

class RAGService:
    def __init__(self):
        self.chunks=[]
        self.retriever=None
        self._chunks_mtime=None
        self._lock=threading.Lock()
        self.reload()

    def _chunks_path(self):
        return settings.PROCESSED_DIR / "chunks.json"

    def _current_mtime(self):
        path=self._chunks_path()
        return path.stat().st_mtime if path.exists() else None

    def reload(self):
        with self._lock:
            path=self._chunks_path()
            mtime=self._current_mtime()
            chunks=[]
            if mtime is not None:
                try:
                    chunks=json.loads(path.read_text(encoding="utf-8"))
                except Exception as exc:
                    print(f"[RAG] Cannot read chunks: {exc}")
            self.chunks=chunks
            self.retriever=HybridRetriever(chunks, settings) if chunks else None
            self._chunks_mtime=mtime

    def reload_if_changed(self):
        """Nạp lại khi chunks.json bị process khác (API/Gradio/CLI) cập nhật."""
        if self._current_mtime()!=self._chunks_mtime:
            self.reload()

    def ask(self, question: str, retrieval_method: str = "hybrid", top_k: int | None = None,
            llm: Optional[dict] = None):
        """llm: tùy chọn {"provider", "model", "api_key"} ghi đè .env cho lượt hỏi này."""
        start=time.perf_counter()
        self.reload_if_changed()
        results=self.retriever.search(question, retrieval_method, top_k) if self.retriever else []
        answer=_generate_answer(question, results, **(llm or {}))
        citations=[]
        seen=set()
        for r in results:
            c=r["chunk"]
            key=(c.get("document_title"), c.get("page"))
            if key in seen: continue
            seen.add(key)
            citations.append({
                "document_title":c.get("document_title","Chưa rõ"),
                "course":c.get("course"), "page":c.get("page",1),
                "chapter":c.get("chapter"), "section":c.get("section"),
                "snippet":" ".join(c.get("text","").split())[:350]
            })
        return {
            "question":question, "answer":answer, "citations":citations,
            "retrieved_chunks":results, "retrieval_method":retrieval_method,
            # Lấy theo dữ liệu đã ingest thực tế, không theo .env hiện tại
            "chunk_strategy":self.chunks[0].get("strategy","unknown") if self.chunks else settings.DEFAULT_CHUNK_STRATEGY,
            "latency_seconds":round(time.perf_counter()-start,4)
        }
