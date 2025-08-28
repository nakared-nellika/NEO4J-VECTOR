import os, sqlite3, hashlib, json
from pathlib import Path

def _ensure_parent(path:str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

class EmbeddingSQLiteCache:
    def __init__(self, db_path:str):
        self.db_path = db_path
        _ensure_parent(db_path)
        self._init()

    def _init(self):
        con = sqlite3.connect(self.db_path)
        con.execute("""CREATE TABLE IF NOT EXISTS embed_cache (key TEXT PRIMARY KEY, vec BLOB)""")
        con.commit(); con.close()

    def _key(self, model:str, text:str):
        return hashlib.sha256((model+'||'+text).encode('utf-8')).hexdigest()

    def get(self, model:str, text:str):
        con = sqlite3.connect(self.db_path)
        cur = con.execute("SELECT vec FROM embed_cache WHERE key=?", (self._key(model, text),))
        row = cur.fetchone(); con.close()
        if row:
            return json.loads(row[0].decode('utf-8'))
        return None

    def set(self, model:str, text:str, vec):
        con = sqlite3.connect(self.db_path)
        con.execute("INSERT OR REPLACE INTO embed_cache(key, vec) VALUES(?,?)",
                    (self._key(model, text), json.dumps(vec).encode('utf-8')))
        con.commit(); con.close()
