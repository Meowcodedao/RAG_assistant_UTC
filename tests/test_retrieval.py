import sys
import unicodedata
from types import SimpleNamespace

import pytest

from app.services.retrieval import HybridRetriever

SETTINGS=SimpleNamespace(EMBEDDING_MODEL_NAME="none",FAISS_TOP_K=2,BM25_TOP_K=2,FINAL_TOP_K=5,RRF_K=60)
CHUNKS=[{"chunk_id":f"c{i}","text":f"mạng hồi quy RNN đoạn {i}"} for i in range(12)]

@pytest.fixture
def retriever(monkeypatch):
    # Không tải embedding model trong test: buộc dùng nhánh fallback
    monkeypatch.setitem(sys.modules,"sentence_transformers",None)
    return HybridRetriever(CHUNKS,SETTINGS)

@pytest.mark.parametrize("method",["hybrid","semantic","keyword"])
def test_top_k_larger_than_branch_k(retriever,method):
    assert len(retriever.search("mạng hồi quy",method,10))==10

def test_nfd_query_matches_nfc_text(retriever):
    query=unicodedata.normalize("NFD","hồi quy")
    assert retriever.search(query,"keyword",3)
