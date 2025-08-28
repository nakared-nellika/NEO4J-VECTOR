import os, glob, json, hashlib
from datetime import datetime, timezone
import pandas as pd
from dotenv import load_dotenv
from utils.neo4j_client import write_tx
from utils.embeddings import embed_texts

load_dotenv()
DOCS_DIR = os.getenv("DOCS_DIR", "./docs/samples")

def summarize_row(row: dict) -> str:
    return ", ".join([f"{k}={v}" for k,v in row.items()])

def upsert_dataset_records(tx, dataset_name, rows):
    tx.run("""
    MERGE (ds:Dataset {name:$name})
      ON CREATE SET ds.created_at=$now
    WITH ds
    UNWIND $rows AS r
    MERGE (rec:Record {rid:r.rid})
      ON CREATE SET rec.created_at=$now
    SET rec += r.props
    MERGE (ds)-[:HAS_RECORD]->(rec)
    """, {"name": dataset_name, "now": datetime.now(timezone.utc).isoformat(), "rows": rows})

def main():
    # CSV
    csvs = glob.glob(os.path.join(DOCS_DIR, "*.csv"))
    for path in csvs:
        df = pd.read_csv(path)
        name = os.path.basename(path)
        rows, texts = [], []
        for i, rec in df.iterrows():
            d = rec.fillna("").to_dict()
            s = summarize_row(d)
            texts.append(s)
            rows.append({"rid": f"{name}::row::{i+1}", "props": {"data": d, "summary": s, "source": path}})
        vecs = embed_texts(texts)
        for j, v in enumerate(vecs):
            rows[j]["props"]["embedding"] = v
        write_tx(upsert_dataset_records, name, rows)
        print(f"✓ Ingested CSV: {name} ({len(rows)} rows)")

    # JSON
    jsons = glob.glob(os.path.join(DOCS_DIR, "*.json"))
    for path in jsons:
        data = json.load(open(path, "r", encoding="utf-8"))
        if isinstance(data, dict): data = [data]
        name = os.path.basename(path)
        rows, texts = [], []
        for i, d in enumerate(data, start=1):
            s = summarize_row(d)
            texts.append(s)
            rows.append({"rid": f"{name}::row::{i}", "props": {"data": d, "summary": s, "source": path}})
        vecs = embed_texts(texts)
        for j, v in enumerate(vecs):
            rows[j]["props"]["embedding"] = v
        write_tx(upsert_dataset_records, name, rows)
        print(f"✓ Ingested JSON: {name} ({len(rows)} rows)")

if __name__=='__main__': main()
