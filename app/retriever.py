import os
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from app.utils.neo4j_client import run_query
from app.utils.embeddings import embed_query


load_dotenv()

TOP_K = int(os.getenv("TOP_K", "8"))
CANDIDATE_K = int(os.getenv("CANDIDATE_K", "48"))
TEXT_IDX = os.getenv("TEXT_VECTOR_INDEX_NAME", "text_embed")
TABLE_IDX = os.getenv("TABLE_VECTOR_INDEX_NAME", "table_embed")
IMAGE_IDX = os.getenv("IMAGECAP_VECTOR_INDEX_NAME", "imagecap_embed")
RERANK_STRATEGY = os.getenv("RERANK_STRATEGY", "mmr")
MMR_LAMBDA = float(os.getenv("MMR_LAMBDA", "0.5"))
ENABLE_TABLE = os.getenv("ENABLE_TABLE_MODALITY","false").lower()=="true"
ENABLE_IMAGE = os.getenv("ENABLE_IMAGE_MODALITY","false").lower()=="true"


def vector_search(index_name:str, qvec:List[float], k:int)->List[Dict[str,Any]]:
    cy = f"""
    WITH $qvec AS qvec
    CALL db.index.vector.queryNodes('{index_name}', $k, qvec)
    YIELD node, score
    RETURN node.id AS id, node.text AS text, node.doc_id AS doc_id, node.page_num AS page, score
    ORDER BY score DESC
    LIMIT $k
    """
    return run_query(cy, {"qvec": qvec, "k": k})

def mmr(cands:List[Dict[str,Any]], top_k:int, lam:float=0.5)->List[Dict[str,Any]]:
    selected = []; cand = cands.copy()
    while cand and len(selected)<top_k:
        if not selected:
            selected.append(cand.pop(0)); continue
        best_i, best_v = 0, -1e9
        for i, c in enumerate(cand):
            sim = c.get("score", 0.0)
            div = max([abs(c.get("score",0.0)-s.get("score",0.0)) for s in selected] + [0.0])
            val = lam*sim - (1-lam)*div
            if val>best_v: best_v, best_i = val, i
        selected.append(cand.pop(best_i))
    return selected

def rrf(*ranked_lists: List[Dict[str,Any]], k:int=60)->List[Dict[str,Any]]:
    scores = {}
    for lst in ranked_lists:
        for rank, item in enumerate(lst, start=1):
            scores.setdefault(item["id"], {"item":item, "score":0.0})
            scores[item["id"]]["score"] += 1.0 / (60.0 + rank)
    merged = [v["item"] | {"rrf": v["score"]} for v in scores.values()]
    merged.sort(key=lambda x: x.get("rrf",0), reverse=True)
    return merged[:k]

def retrieve(query:str)->Tuple[List[Dict[str,Any]], Dict[str,Any]]:
    qvec = embed_query(query)
    text_hits = vector_search(TEXT_IDX, qvec, CANDIDATE_K)
    lists = [text_hits]
    if ENABLE_TABLE:
        lists.append(vector_search(TABLE_IDX, qvec, CANDIDATE_K))
    if ENABLE_IMAGE:
        lists.append(vector_search(IMAGE_IDX, qvec, CANDIDATE_K))
    fused = rrf(*lists, k=max(CANDIDATE_K, TOP_K))
    reranked = mmr(fused, TOP_K, MMR_LAMBDA) if RERANK_STRATEGY == "mmr" else fused[:TOP_K]
    return reranked, {"candidates": len(fused), "strategy": RERANK_STRATEGY,
                      "modalities": ["text"] + (["table"] if ENABLE_TABLE else []) + (["image"] if ENABLE_IMAGE else [])}

