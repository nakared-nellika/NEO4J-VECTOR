import os, glob, hashlib
from datetime import datetime, timezone
from dotenv import load_dotenv
from utils.neo4j_client import write_tx
from utils.embeddings import embed_texts
from PyPDF2 import PdfReader

load_dotenv()
DOCS_DIR = os.getenv("DOCS_DIR", "./docs/samples")

def extract_text_from_pdf(path):
    reader = PdfReader(path)
    return [(i+1, (p.extract_text() or "")) for i, p in enumerate(reader.pages)]

def upsert_blocks(tx, doc_id, source, rows):
    tx.run("""
    MERGE (d:Document {id:$doc_id})
      ON CREATE SET d.source=$source, d.created_at=$now
      ON MATCH SET d.source=$source
    WITH d
    UNWIND $rows AS r
    MERGE (b:Block {id:r.id})
      ON CREATE SET b.created_at=$now
    SET b += r.props
    MERGE (d)-[:HAS_BLOCK]->(b)
    """, {"doc_id": doc_id, "source": source, "now": datetime.now(timezone.utc).isoformat(), "rows": rows})

def main():
    pdfs = glob.glob(os.path.join(DOCS_DIR, "*.pdf"))
    for path in pdfs:
        pages = extract_text_from_pdf(path)
        doc_id = f"doc::{hashlib.md5(path.encode()).hexdigest()}"
        rows = []
        for page_num, t in pages:
            if not t.strip(): continue
            v = embed_texts([t])[0]
            rows.append({"id": f"{doc_id}::page::{page_num}::block::1",
                         "props": {"type":"TEXT","text":t,"embedding":v,"doc_id":doc_id,"page_num":page_num,"source":path,"lang":"th"}})
        if rows:
            write_tx(upsert_blocks, doc_id, path, rows)
        print(f"✓ Ingested PDF: {os.path.basename(path)} ({len(rows)} blocks)")

if __name__=='__main__': main()
