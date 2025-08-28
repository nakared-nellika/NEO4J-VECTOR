from fastapi import FastAPI, Response, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os, time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.cache import SQLiteCache
import langchain
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from app.utils.neo4j_client import get_driver 
from app.retriever import retrieve

load_dotenv()
app = FastAPI(title="GraphRAG Vector PRO (Repack)", version="1.2.1")

# LLM cache
if os.getenv("ENABLE_LLM_CACHE", "true").lower()=="true":
    langchain.llm_cache = SQLiteCache(database_path=os.getenv("LLM_CACHE_PATH", ".cache/llm_cache.sqlite"))

# Metrics
REQ_COUNT = Counter("api_requests_total", "Total API requests", ["endpoint","method","status"])
REQ_LAT   = Histogram("api_request_latency_seconds", "Request latency", ["endpoint","method"])
TOK_PROMPT= Counter("llm_prompt_tokens_total", "Total prompt tokens")
TOK_COMP  = Counter("llm_completion_tokens_total", "Total completion tokens")
TOK_TOTAL = Counter("llm_total_tokens_total", "Total tokens")

@app.get("/")
def home():
    return {
        "name": "GraphRAG Vector PRO",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.time()
    endpoint = request.url.path; method = request.method; status="500"
    try:
        resp = await call_next(request); status = str(resp.status_code); return resp
    finally:
        REQ_COUNT.labels(endpoint=endpoint, method=method, status=status).inc()
        REQ_LAT.labels(endpoint=endpoint, method=method).observe(time.time()-start)

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

class QARequest(BaseModel):
    question: str

@app.get("/health")
def health():
    drv = get_driver()
    try:
        with drv.session() as s: s.run("RETURN 1")
        return {"status":"ok"}
    except Exception as e:
        return {"status":"error","detail":str(e)}

@app.post("/graphrag/qa")
def qa(req: QARequest):
    hits, meta = retrieve(req.question)
    usage = {}
    try:
        llm = ChatOpenAI(model=os.getenv("CHAT_MODEL","gpt-4o-mini"))
        context = "\n\n".join([f"- {h['text']}" for h in hits])
        prompt = f"ตอบคำถามต่อไปนี้อย่างกระชับ อ้างอิงจากบริบท:\nคำถาม: {req.question}\n\nบริบท:\n{context}"
        msg = llm.invoke(prompt)
        answer = msg.content
        try:
            usage = (msg.response_metadata or {}).get("token_usage", {})
        except Exception:
            usage = {}
    except Exception as e:
        answer = f"(LLM ไม่พร้อม: {e})\n\nเอกสารที่เกี่ยวข้อง:\n" + "\n".join([f"- {h['text'][:200]}..." for h in hits])

    TOK_PROMPT.inc(float(usage.get("prompt_tokens", 0)))
    TOK_COMP.inc(float(usage.get("completion_tokens", 0)))
    TOK_TOTAL.inc(float(usage.get("total_tokens", 0)))

    return {"question": req.question, "answer": answer, "meta": meta, "top_k": len(hits), "hits": hits[:10]}

@app.post("/graphrag/qa_stream")
def qa_stream(req: QARequest):
    hits, meta = retrieve(req.question)
    context = "\n\n".join([f"- {h['text']}" for h in hits])
    system_prompt = (
        "ตอบให้กระชับ อ้างอิงจากบริบทด้านล่าง ถ้าไม่แน่ใจให้บอกว่าไม่ทราบ\n"
        f"บริบท:\n{context}\n"
    )
    llm = ChatOpenAI(model=os.getenv("CHAT_MODEL","gpt-4o-mini"))

    def gen():
        yield "เริ่มสตรีมคำตอบ...\n"
        try:
            for chunk in llm.stream(system_prompt + f"\nคำถาม: {req.question}"):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            yield f"\n[สตรีมผิดพลาด: {e}]"
        yield "\n\n[จบ]"

    return StreamingResponse(gen(), media_type="text/plain; charset=utf-8")
