# GraphRAG – Neo4j Vector PRO (RRF + Retry + Streaming + Token Metrics)

## Run (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env  # แก้ OPENAI_API_KEY + Neo4j
python app/check_connection.py
python app/setup_schema.py
python app/ingest_docs.py
python app/ingest_pdf_multi.py
python app/ingest_semistructured.py
python -m uvicorn app.app:app --port 8000
# เปิด /docs, /metrics
```
