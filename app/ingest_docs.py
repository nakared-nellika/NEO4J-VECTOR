import os, glob, hashlib
from datetime import datetime, timezone
from dotenv import load_dotenv
from utils.neo4j_client import write_tx
from utils.embeddings import embed_texts

load_dotenv()
DOCS_DIR = os.getenv("DOCS_DIR", "./docs/samples")

def chunks(text, max_len=800, overlap=120):
    words = text.split()
    out = []; i=0
    while i < len(words):
        out.append(" ".join(words[i:i+max_len]))
        i += (max_len - overlap)
    return [c for c in out if c.strip()]

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
    WITH b,r
    CALL {
      WITH b, r
      REMOVE b:TextBlock:TableSummary:ImageCaption
      SET b:Block
      FOREACH (x IN CASE WHEN r.props.type='TEXT' THEN [1] ELSE [] END | SET b:TextBlock)
      FOREACH (x IN CASE WHEN r.props.type='TABLE_SUMMARY' THEN [1] ELSE [] END | SET b:TableSummary)
      FOREACH (x IN CASE WHEN r.props.type='IMAGE_CAPTION' THEN [1] ELSE [] END | SET b:ImageCaption)
    }
    """, {"doc_id": doc_id, "source": source, "now": datetime.now(timezone.utc).isoformat(), "rows": rows})

def main():
    txts = glob.glob(os.path.join(DOCS_DIR, "*.txt"))
    for path in txts:
        text = open(path, "r", encoding="utf-8", errors="ignore").read()
        chs = chunks(text)
        vecs = embed_texts(chs)
        doc_id = f"doc::{hashlib.md5(path.encode()).hexdigest()}"
        rows = []
        for i, (c, v) in enumerate(zip(chs, vecs), start=1):
            rows.append({"id": f"{doc_id}::page::0::block::{i}",
                         "props": {"type":"TEXT","text":c,"embedding":v,"doc_id":doc_id,"page_num":0,"source":path,"lang":"th"}})
        write_tx(upsert_blocks, doc_id, path, rows)
        print(f"✓ Ingested TXT: {os.path.basename(path)} ({len(rows)} blocks)")

if __name__=="__main__": main()
