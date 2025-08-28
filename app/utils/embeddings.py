import os
from typing import List
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from tenacity import retry, wait_exponential_jitter, stop_after_attempt, retry_if_exception_type
from .cache import EmbeddingSQLiteCache

load_dotenv()

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")
ENABLE_EMBED_CACHE = os.getenv("ENABLE_EMBED_CACHE", "true").lower() == "true"
EMBED_CACHE_PATH = os.getenv("EMBED_CACHE_PATH", ".cache/embeddings.sqlite")

_embedder = None
_cache = EmbeddingSQLiteCache(EMBED_CACHE_PATH) if ENABLE_EMBED_CACHE else None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = OpenAIEmbeddings(model=EMBED_MODEL)
    return _embedder

@retry(wait=wait_exponential_jitter(initial=0.5, max=8.0),
       stop=stop_after_attempt(5),
       retry=retry_if_exception_type(Exception))
def _embed_documents_retry(emb, texts: List[str]):
    return emb.embed_documents(texts)

@retry(wait=wait_exponential_jitter(initial=0.5, max=8.0),
       stop=stop_after_attempt(5),
       retry=retry_if_exception_type(Exception))
def _embed_query_retry(emb, text: str):
    return emb.embed_query(text)

def embed_texts(texts: List[str]) -> List[List[float]]:
    emb = get_embedder()
    if _cache:
        to_compute, idx_map, out = [], [], [None]*len(texts)
        for i, t in enumerate(texts):
            c = _cache.get(EMBED_MODEL, t)
            if c is None:
                to_compute.append(t); idx_map.append(i)
            else:
                out[i] = c
        if to_compute:
            vecs = _embed_documents_retry(emb, to_compute)
            for j, v in enumerate(vecs):
                _cache.set(EMBED_MODEL, to_compute[j], v)
                out[idx_map[j]] = v
        return out
    return _embed_documents_retry(emb, texts)

def embed_query(text: str) -> List[float]:
    emb = get_embedder()
    if _cache:
        c = _cache.get(EMBED_MODEL, text)
        if c is not None: return c
        v = _embed_query_retry(emb, text)
        _cache.set(EMBED_MODEL, text, v)
        return v
    return _embed_query_retry(emb, text)
