"""Gradio web UI. Run with: python -m app.frontend.gradio_app"""
import csv, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import gradio as gr
from app.services.rag import RAGService
from app.services.ingestion import IngestError, save_uploaded_pdf, run_ingest_script
from app.services.corpus import load_cleaned_documents, page_texts
from app.services.evaluation import (METHODS, KS, load_queries, save_queries, ground_truth_pages,
                                     chunk_pages, evaluate, run_experiments)
from app.config import settings

service=RAGService()
NEW_QUERY="__new__"
QUERY_TYPES=["keyword","semantic","multi_hop"]

# ---------- Dữ liệu dùng chung ----------
_docs_cache={"mtime":object(),"documents":[]}

def cleaned_documents():
    """Văn bản trang đã làm sạch; đọc lại PDF khi kho chunk thay đổi."""
    mtime=service._current_mtime()
    if _docs_cache["mtime"]!=mtime:
        _docs_cache["documents"]=load_cleaned_documents(settings.PDF_DIR)
        _docs_cache["mtime"]=mtime
    return _docs_cache["documents"]

def sources():
    service.reload_if_changed()
    return sorted({c["source"] for c in service.chunks})

def parse_ints(text):
    return sorted({int(x) for x in str(text).replace(";",",").split(",") if x.strip().isdigit()})

def short(text,n=160):
    t=" ".join((text or "").split())
    return t if len(t)<=n else t[:n-1]+"…"

# ---------- Hỏi đáp ----------
def ollama_models():
    """Danh sách model Ollama đã cài (rỗng nếu Ollama chưa chạy)."""
    import urllib.request
    try:
        with urllib.request.urlopen(f"{settings.OLLAMA_BASE_URL}/api/tags",timeout=2) as resp:
            return [m["name"] for m in json.load(resp).get("models",[])]
    except Exception:
        return []

def answer(question, method, top_k, provider, model, api_key):
    if not question or not question.strip():
        return "Vui lòng nhập câu hỏi.", "", ""
    llm={"provider":provider,"model":model,"api_key":(api_key or "").strip() or None}
    result=service.ask(question.strip(), method, int(top_k), llm=llm)
    citations=[]
    for c in result["citations"]:
        citations.append(f"- **{c['document_title']}** — trang {c['page']}\n\n  {c['snippet']}")
    sources_md="\n\n".join(citations) if citations else "Chưa tìm thấy nguồn phù hợp."
    retrieved=[]
    if service.retriever:
        retrieved.append(f"*Nhánh ngữ nghĩa: {service.retriever.semantic_backend} · thời gian: {result['latency_seconds']}s*")
    for r in result["retrieved_chunks"]:
        c=r["chunk"]
        if r.get("rrf_score") is not None: score=f"điểm RRF: {r['rrf_score']:.4f}"
        elif r.get("faiss_score") is not None: score=f"điểm ngữ nghĩa: {r['faiss_score']:.4f}"
        elif r.get("bm25_score") is not None: score=f"điểm BM25: {r['bm25_score']:.4f}"
        else: score="không có điểm"
        section=f" — *{c['section']}*" if c.get("section") and c["section"]!="Chưa rõ" else ""
        retrieved.append(f"**{r['final_rank']}. {c.get('document_title')} — trang {c.get('page')}**{section} · `{c.get('chunk_id')}` ({score})\n\n{c.get('text','')[:1200]}")
    return result["answer"], sources_md, "\n\n---\n\n".join(retrieved)

# ---------- Nạp tài liệu ----------
def ingest(file):
    if file is None: return "Chọn một tệp PDF trước."
    src=Path(file)
    try:
        target,is_new=save_uploaded_pdf(src.read_bytes(),src.name)
        if not is_new:
            return f"Tài liệu đã có trong kho ({target.name}), không nạp lại."
        try:
            log=run_ingest_script()
        except IngestError:
            target.unlink(missing_ok=True)
            raise
    except IngestError as exc:
        return f"Nạp thất bại:\n{exc}"
    service.reload()
    return log[-2500:] or f"Đã nạp thành công. Tổng số đoạn: {len(service.chunks)}"

# ---------- Duyệt tài liệu ----------
CHUNK_HEADERS=["chunk_id","trang","trang gốc","số ký tự","mục / tiêu đề slide","nội dung"]

def chunk_rows(chunks):
    return [[c["chunk_id"],c["page"],", ".join(map(str,chunk_pages(c))),c["char_count"],c.get("section"),short(c["text"],220)] for c in chunks]

def browse_page(source, page):
    if not source: return "Chọn tài liệu.", []
    page=int(page or 1)
    text=page_texts(cleaned_documents()).get((source,page))
    page_md=f"### {source} — trang {page}\n\n```text\n{text}\n```" if text else f"Trang {page} không có văn bản (hoặc không tồn tại)."
    return page_md, chunk_rows([c for c in service.chunks if c["source"]==source and page in chunk_pages(c)])

def search_chunks(keyword):
    kw=(keyword or "").strip().lower()
    if not kw: return []
    return chunk_rows([c for c in service.chunks if kw in c["text"].lower()][:100])

# ---------- Bộ câu hỏi ----------
QUERY_HEADERS=["query_id","loại","câu hỏi","ground truth","số chunk đã gán"]

def query_table():
    return [[q["query_id"],q.get("query_type"),q["question"],
             f"{q.get('ground_truth_source')} p.{','.join(map(str,ground_truth_pages(q)))}",
             len(q.get("relevant_chunk_ids",[]))] for q in load_queries()]

def query_choices():
    return [("➕ Câu hỏi mới",NEW_QUERY)]+[(f"{q['query_id']} — {short(q['question'],70)}",q["query_id"]) for q in load_queries()]

def next_query_id(queries):
    nums=[int(q["query_id"][1:]) for q in queries if q["query_id"][1:].isdigit()]
    return f"q{max(nums,default=0)+1:02d}"

def load_query(qid):
    queries=load_queries()
    q=next((x for x in queries if x["query_id"]==qid),None)
    if q is None:
        return (next_query_id(queries),"","semantic",gr.Dropdown(choices=sources(),value=None),"","",
                gr.CheckboxGroup(choices=[],value=[]),"Nhập câu hỏi mới rồi bấm **Tìm chunk ứng viên**.")
    choices,value=candidate_choices(q["question"],q.get("ground_truth_source"),ground_truth_pages(q),q.get("relevant_chunk_ids",[]))
    missing=set(q.get("relevant_chunk_ids",[]))-{c["chunk_id"] for c in service.chunks}
    note=f"⚠️ {len(missing)} chunk đã gán không còn trong kho hiện tại (do ingest lại với cấu hình khác): {', '.join(sorted(missing))}" if missing else ""
    return (q["query_id"],q["question"],q.get("query_type","semantic"),
            gr.Dropdown(choices=sources(),value=q.get("ground_truth_source")),
            ", ".join(map(str,ground_truth_pages(q))),q.get("expected_answer",""),
            gr.CheckboxGroup(choices=choices,value=value),note)

def candidate_choices(question, source, pages, selected):
    """Ứng viên = top 10 của cả 3 phương pháp + mọi chunk trên các trang ground truth."""
    ids=[]
    if service.retriever and question.strip():
        for m in METHODS:
            ids+=[r["chunk"]["chunk_id"] for r in service.retriever.search(question,m,10)]
    ids+=[c["chunk_id"] for c in service.chunks if c["source"]==source and set(pages)&set(chunk_pages(c))]
    ids+=list(selected)
    by_id={c["chunk_id"]:c for c in service.chunks}
    ordered=list(dict.fromkeys(i for i in ids if i in by_id))
    choices=[(f"{i} · {by_id[i]['source']} p.{', '.join(map(str,chunk_pages(by_id[i])))} · {short(by_id[i]['text'],140)}",i) for i in ordered]
    return choices,[i for i in selected if i in by_id]

def find_candidates(question, source, pages, selected):
    choices,value=candidate_choices(question or "",source,parse_ints(pages),selected or [])
    return gr.CheckboxGroup(choices=choices,value=value)

def save_query(qid, question, qtype, source, pages, expected, selected):
    qid=(qid or "").strip()
    if not qid or not (question or "").strip(): return "❌ Cần nhập mã và nội dung câu hỏi.",gr.Dropdown(),query_table()
    if not source: return "❌ Cần chọn tài liệu ground truth.",gr.Dropdown(),query_table()
    by_id={c["chunk_id"]:c for c in service.chunks}
    selected=list(selected or [])
    # Trang ground truth = trang nhập tay + trang của các chunk đã chọn (cùng tài liệu),
    # để tab Thử nghiệm (đánh giá mức trang) dùng đúng các nhãn vừa gán.
    page_set=set(parse_ints(pages))|{by_id[i]["page"] for i in selected if i in by_id and by_id[i]["source"]==source}
    if not page_set: return "❌ Cần nhập trang hoặc chọn ít nhất một chunk.",gr.Dropdown(),query_table()
    other=[i for i in selected if i in by_id and by_id[i]["source"]!=source]
    item={"query_id":qid,"question":question.strip(),"query_type":qtype,
          "ground_truth_source":source,"ground_truth_page":min(page_set),
          "ground_truth_pages":sorted(page_set),"ground_truth_section":None,
          "relevant_chunk_ids":selected,"expected_answer":(expected or "").strip()}
    queries=load_queries()
    for i,q in enumerate(queries):
        if q["query_id"]==qid:
            item["ground_truth_section"]=q.get("ground_truth_section")
            queries[i]=item; break
    else:
        queries.append(item)
    save_queries(queries)
    msg=f"✅ Đã lưu {qid} ({len(selected)} chunk, trang {', '.join(map(str,sorted(page_set)))})."
    if other: msg+=f"\n\n⚠️ {len(other)} chunk thuộc tài liệu khác chỉ được tính ở đánh giá mức chunk."
    return msg,gr.Dropdown(choices=query_choices(),value=qid),query_table()

def delete_query(qid):
    queries=load_queries()
    remaining=[q for q in queries if q["query_id"]!=qid]
    if len(remaining)==len(queries): return "Không tìm thấy câu hỏi để xóa.",gr.Dropdown(),query_table()
    save_queries(remaining)
    return f"🗑️ Đã xóa {qid}.",gr.Dropdown(choices=query_choices(),value=NEW_QUERY),query_table()

# ---------- Đánh giá ----------
METRIC_HEADERS=["MRR"]+[f"Recall@{k}" for k in KS]
DETAIL_HEADERS=["query_id","loại","câu hỏi","ground truth","GT theo","hạng đúng đầu tiên"]

def run_evaluation():
    service.reload_if_changed()
    queries=load_queries()
    if not service.retriever: return "❌ Chưa có kho chunk, hãy nạp tài liệu trước.",[],[],None,None,[]
    if not queries: return "❌ Bộ câu hỏi trống.",[],[],None,None,[]
    result=evaluate(service.retriever,service.chunks,queries)
    result["semantic_backend"]=service.retriever.semantic_backend
    path=settings.REPORTS_DIR/"retrieval_evaluation.json"
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    summary=[[m]+[result["summary"][m][h] for h in METRIC_HEADERS] for m in METHODS]
    by_type=[[m,t,v["queries"]]+[v[h] for h in METRIC_HEADERS]
             for m in METHODS for t,v in result["summary_by_type"][m].items()]
    modes={r["ground_truth_mode"] for r in result["details"]["hybrid"]}
    info=(f"Đã chạy {len(queries)} câu · nhánh ngữ nghĩa: **{service.retriever.semantic_backend}** · "
          f"ground truth: {', '.join(sorted(modes))} · đã lưu `{path.relative_to(settings.BASE_DIR)}`")
    return info,summary,by_type,result,str(path),detail_rows(result,"hybrid")

def detail_rows(result, method):
    if not result: return []
    return [[r["query_id"],r["query_type"],r["question"],r["ground_truth"],r["ground_truth_mode"],
             r["first_relevant_rank"] if r["first_relevant_rank"] else "✗ trượt"] for r in result["details"][method]]

def show_detail(result, method, evt: gr.SelectData):
    if not result: return ""
    row=result["details"][method][evt.index[0]]
    by_id={c["chunk_id"]:c for c in service.chunks}
    lines=[f"### {row['query_id']}: {row['question']}",f"Ground truth: **{row['ground_truth']}** (theo {row['ground_truth_mode']})",""]
    for i,(cid,ok) in enumerate(zip(row["top_chunk_ids"],row["top_relevant"]),1):
        c=by_id.get(cid,{})
        lines.append(f"{i}. {'✅' if ok else '❌'} `{cid}` — {c.get('source')} p.{c.get('page')}\n\n   {short(c.get('text',''),300)}")
    return "\n\n".join(lines)

# ---------- Thử nghiệm ----------
EXP_HEADERS=["slide_cleanup","strategy","chunk_size","chunk_overlap","embedding_model","rrf_k","method","chunks","semantic_backend"]+METRIC_HEADERS

def run_grid(strategies, sizes, overlaps, models, rrf_ks, methods, cleanups, progress=gr.Progress()):
    sizes,overlaps,rrf_ks=parse_ints(sizes),parse_ints(overlaps),parse_ints(rrf_ks)
    models=[m.strip() for m in (models or "").splitlines() if m.strip()]
    cleanups=[c=="Bật" for c in (cleanups or [])]
    if not (strategies and sizes and overlaps and models and rrf_ks and methods and cleanups):
        return "❌ Cần chọn ít nhất một giá trị cho mỗi tham số.",[],None,None
    queries=load_queries()
    if not queries: return "❌ Bộ câu hỏi trống.",[],None,None
    documents=cleaned_documents()
    if not documents: return "❌ Không có PDF nào trong data/pdfs.",[],None,None
    rows=run_experiments(documents,queries,strategies,sizes,overlaps,models,rrf_ks,methods,cleanups,
                         progress=lambda f,msg: progress(f,desc=msg))
    rows.sort(key=lambda r:(r["MRR"],r["Recall@5"]),reverse=True)
    json_path=settings.REPORTS_DIR/"experiments.json"
    csv_path=settings.REPORTS_DIR/"experiments.csv"
    json_path.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding="utf-8")
    with csv_path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=EXP_HEADERS+["queries"]); w.writeheader(); w.writerows(rows)
    best=[]
    for m in methods:
        top=next((r for r in rows if r["method"]==m),None)
        if top: best.append(f"- **{m}**: MRR {top['MRR']} · R@5 {top['Recall@5']} — chuẩn hóa slide {'bật' if top['slide_cleanup'] else 'tắt'}, {top['strategy']}, size {top['chunk_size']}, overlap {top['chunk_overlap']}, `{top['embedding_model']}`, RRF_K {top['rrf_k']}")
    fallback=[r for r in rows if "fallback" in r["semantic_backend"]]
    info=f"Đã chạy {len(rows)//max(1,len(methods))} cấu hình × {len(methods)} phương pháp trên {len(queries)} câu hỏi (ground truth mức trang).\n\n**Tốt nhất theo MRR:**\n"+"\n".join(best)
    if fallback: info+=f"\n\n⚠️ {len({r['embedding_model'] for r in fallback})} model không tải được, nhánh ngữ nghĩa đã dùng fallback — xem cột semantic_backend."
    return info,[[r[h] for h in EXP_HEADERS] for r in rows],str(json_path),str(csv_path)

# ---------- Giao diện ----------
with gr.Blocks(title="Hybrid RAG Assistant") as demo:
    gr.Markdown("# 📚 Hybrid RAG Assistant\nHỏi đáp trên bài giảng PDF, có hiển thị đoạn bằng chứng và số trang.")

    with gr.Tab("Hỏi đáp"):
        question=gr.Textbox(label="Câu hỏi",placeholder="Ví dụ: RNN gặp vấn đề gì với vanishing gradient?",lines=3)
        with gr.Row():
            method=gr.Dropdown(choices=[("Hybrid (FAISS + BM25 + RRF)","hybrid"),("Ngữ nghĩa","semantic"),("Từ khóa","keyword")],value="hybrid",label="Phương pháp truy xuất")
            top_k=gr.Slider(1,20,value=5,step=1,label="Số kết quả")
        with gr.Accordion("Cấu hình LLM (chỉ áp dụng cho phiên này, không lưu vào .env)",open=False):
            with gr.Row():
                provider=gr.Dropdown(choices=["mock","ollama","openai","gemini"],value=settings.LLM_PROVIDER,label="Provider")
                model=gr.Dropdown(choices=ollama_models(),value=settings.LLM_MODEL_NAME or None,allow_custom_value=True,
                                  label="Model",info="Để trống = mặc định (ollama: "+settings.DEFAULT_LLM_MODELS["ollama"]+", openai: gpt-4o-mini, gemini: gemini-2.5-flash)")
                api_key=gr.Textbox(label="API key",type="password",placeholder="Để trống = lấy từ .env (Ollama không cần)")
        submit=gr.Button("Tìm kiếm và trả lời",variant="primary")
        output=gr.Markdown(label="Trả lời")
        sources_out=gr.Markdown(label="Nguồn trích dẫn")
        with gr.Accordion("Các đoạn được truy xuất",open=False):
            retrieved=gr.Markdown()
        ask_inputs=[question,method,top_k,provider,model,api_key]
        submit.click(answer,ask_inputs,[output,sources_out,retrieved])
        question.submit(answer,ask_inputs,[output,sources_out,retrieved])

    with gr.Tab("Nạp tài liệu"):
        gr.Markdown("Tải PDF lên để thêm vào kho dữ liệu. Quá trình này trích xuất văn bản và tạo lại chỉ mục.")
        pdf=gr.File(label="Tệp PDF",file_types=[".pdf"],type="filepath")
        ingest_btn=gr.Button("Nạp PDF",variant="primary")
        ingest_output=gr.Textbox(label="Kết quả",lines=8)
        ingest_btn.click(ingest,[pdf],[ingest_output])

    with gr.Tab("Duyệt tài liệu"):
        gr.Markdown("Xem văn bản đã trích xuất của từng trang và các chunk tương ứng — dùng để kiểm tra ground truth hoặc vì sao một câu hỏi bị trượt.")
        with gr.Row():
            b_source=gr.Dropdown(choices=sources(),label="Tài liệu")
            b_page=gr.Number(value=1,precision=0,minimum=1,label="Trang")
            b_btn=gr.Button("Xem trang")
        b_chunks=gr.Dataframe(headers=CHUNK_HEADERS,label="Chunk trên trang này",interactive=False,wrap=True)
        b_text=gr.Markdown()
        with gr.Row():
            b_kw=gr.Textbox(label="Tìm chunk chứa từ khóa",placeholder="vd: dropout")
            b_kw_btn=gr.Button("Tìm")
        b_kw_out=gr.Dataframe(headers=CHUNK_HEADERS,label="Kết quả tìm (tối đa 100)",interactive=False,wrap=True)
        b_btn.click(browse_page,[b_source,b_page],[b_text,b_chunks])
        b_page.submit(browse_page,[b_source,b_page],[b_text,b_chunks])
        b_kw_btn.click(search_chunks,[b_kw],[b_kw_out])
        b_kw.submit(search_chunks,[b_kw],[b_kw_out])

    with gr.Tab("Bộ câu hỏi"):
        gr.Markdown("Thêm/sửa câu hỏi đánh giá và **đánh dấu chunk chứa đáp án**. Chunk đã gán dùng cho tab Đánh giá; "
                    "trang của các chunk đó được lưu vào `ground_truth_pages` để dùng cho tab Thử nghiệm.")
        q_table=gr.Dataframe(value=query_table(),headers=QUERY_HEADERS,label="Bộ câu hỏi hiện tại",interactive=False,wrap=True)
        q_select=gr.Dropdown(choices=query_choices(),value=NEW_QUERY,label="Chọn câu hỏi để sửa")
        with gr.Row():
            q_id=gr.Textbox(label="Mã",scale=1)
            q_type=gr.Dropdown(choices=QUERY_TYPES,value="semantic",label="Loại",scale=1)
            q_source=gr.Dropdown(choices=sources(),label="Tài liệu ground truth",scale=2)
            q_pages=gr.Textbox(label="Trang (vd: 42, 44)",scale=1)
        q_question=gr.Textbox(label="Câu hỏi",lines=2)
        q_expected=gr.Textbox(label="Đáp án mong đợi",lines=2)
        q_find=gr.Button("Tìm chunk ứng viên")
        q_chunks=gr.CheckboxGroup(choices=[],label="Chunk chứa đáp án (tick các chunk đúng)")
        with gr.Row():
            q_save=gr.Button("Lưu câu hỏi",variant="primary")
            q_delete=gr.Button("Xóa câu hỏi",variant="stop")
        q_status=gr.Markdown()
        q_select.change(load_query,[q_select],[q_id,q_question,q_type,q_source,q_pages,q_expected,q_chunks,q_status])
        q_find.click(find_candidates,[q_question,q_source,q_pages,q_chunks],[q_chunks])
        q_save.click(save_query,[q_id,q_question,q_type,q_source,q_pages,q_expected,q_chunks],[q_status,q_select,q_table])
        q_delete.click(delete_query,[q_id],[q_status,q_select,q_table])

    with gr.Tab("Đánh giá"):
        gr.Markdown("Chạy bộ câu hỏi trên kho chunk hiện tại cho cả 3 phương pháp. Bấm vào một dòng chi tiết để xem top-5 kết quả.")
        e_btn=gr.Button("Chạy đánh giá",variant="primary")
        e_info=gr.Markdown()
        e_summary=gr.Dataframe(headers=["phương pháp"]+METRIC_HEADERS,label="Tổng hợp",interactive=False)
        e_by_type=gr.Dataframe(headers=["phương pháp","loại câu hỏi","số câu"]+METRIC_HEADERS,label="Theo loại câu hỏi",interactive=False)
        e_method=gr.Radio(choices=list(METHODS),value="hybrid",label="Chi tiết theo phương pháp")
        e_details=gr.Dataframe(headers=DETAIL_HEADERS,label="Chi tiết từng câu",interactive=False,wrap=True)
        e_detail_md=gr.Markdown()
        e_file=gr.File(label="Báo cáo JSON")
        e_state=gr.State()
        e_btn.click(run_evaluation,[],[e_info,e_summary,e_by_type,e_state,e_file,e_details]).then(
            lambda:"",[],[e_detail_md])
        e_method.change(detail_rows,[e_state,e_method],[e_details])
        e_details.select(show_detail,[e_state,e_method],[e_detail_md])

    with gr.Tab("Thử nghiệm"):
        gr.Markdown("So sánh nhiều cấu hình. Mỗi cấu hình được dựng kho chunk **trong bộ nhớ** (không ghi đè dữ liệu hiện tại). "
                    "Ground truth tính theo **trang** để các cách chunking khác nhau so sánh công bằng. "
                    "Để dùng cấu hình tốt nhất cho Hỏi đáp: sửa `.env` rồi chạy lại `python scripts/ingest.py`.")
        with gr.Row():
            x_strat=gr.CheckboxGroup(choices=["fixed","recursive"],value=["fixed","recursive"],label="Chiến lược chunking")
            x_methods=gr.CheckboxGroup(choices=list(METHODS),value=list(METHODS),label="Phương pháp truy xuất")
            x_clean=gr.CheckboxGroup(choices=["Bật","Tắt"],value=["Bật","Tắt"],label="Chuẩn hóa slide (bỏ trang hiệu ứng, tiêu đề lặp)")
        with gr.Row():
            x_sizes=gr.Textbox(value="300, 500, 800",label="Chunk size (ký tự)")
            x_overlaps=gr.Textbox(value="50, 100",label="Overlap (ký tự)")
            x_rrf=gr.Textbox(value="60",label="RRF_K")
        x_models=gr.Textbox(value=f"{settings.EMBEDDING_MODEL_NAME}\nbkai-foundation-models/vietnamese-bi-encoder",
                            lines=3,label="Embedding model (mỗi dòng một model; lần đầu sẽ tải từ Internet)")
        x_btn=gr.Button("Chạy thử nghiệm",variant="primary")
        x_info=gr.Markdown()
        x_table=gr.Dataframe(headers=EXP_HEADERS,label="Kết quả (sắp xếp theo MRR)",interactive=False,wrap=True)
        with gr.Row():
            x_json=gr.File(label="experiments.json")
            x_csv=gr.File(label="experiments.csv")
        x_btn.click(run_grid,[x_strat,x_sizes,x_overlaps,x_models,x_rrf,x_methods,x_clean],[x_info,x_table,x_json,x_csv])

    gr.Markdown("**Lưu ý:** ở chế độ mock/offline, hệ thống hiển thị bằng chứng truy xuất thay vì tự tổng hợp câu trả lời.")

if __name__=="__main__":
    demo.launch(server_name=settings.UI_HOST,server_port=settings.UI_PORT,theme=gr.themes.Soft())
