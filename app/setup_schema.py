import os
from dotenv import load_dotenv
from utils.neo4j_client import run_query

load_dotenv()
VECTOR_DIM = int(os.getenv("VECTOR_DIM", "3072"))
TEXT_IDX = os.getenv("TEXT_VECTOR_INDEX_NAME", "text_embed")
TABLE_IDX = os.getenv("TABLE_VECTOR_INDEX_NAME", "table_embed")
IMAGE_IDX = os.getenv("IMAGECAP_VECTOR_INDEX_NAME", "imagecap_embed")

DDL = f'''
CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE;
CREATE CONSTRAINT block_id IF NOT EXISTS FOR (b:Block)   REQUIRE b.id IS UNIQUE;
CREATE INDEX IF NOT EXISTS FOR (b:Block) ON (b.type);
CREATE INDEX IF NOT EXISTS FOR (b:Block) ON (b.doc_id);
CREATE INDEX IF NOT EXISTS FOR (b:Block) ON (b.page_num);
CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.source);
CREATE INDEX IF NOT EXISTS FOR (b:Block) ON (b.created_at);

CREATE VECTOR INDEX {TEXT_IDX} IF NOT EXISTS
FOR (b:TextBlock) ON (b.embedding)
OPTIONS {{ indexConfig: {{ `vector.dimensions`: {VECTOR_DIM}, `vector.similarity_function`: 'cosine' }} }};

CREATE VECTOR INDEX {TABLE_IDX} IF NOT EXISTS
FOR (b:TableSummary) ON (b.embedding)
OPTIONS {{ indexConfig: {{ `vector.dimensions`: {VECTOR_DIM}, `vector.similarity_function`: 'cosine' }} }};

CREATE VECTOR INDEX {IMAGE_IDX} IF NOT EXISTS
FOR (b:ImageCaption) ON (b.embedding)
OPTIONS {{ indexConfig: {{ `vector.dimensions`: {VECTOR_DIM}, `vector.similarity_function`: 'cosine' }} }};
'''

def main():
    for stmt in [x.strip() for x in DDL.strip().split(';') if x.strip()]:
        run_query(stmt)
    print("✓ Schema & Vector Index ready")
if __name__=="__main__": main()
