from utils.neo4j_client import run_query
def main():
    r = run_query("RETURN 1 AS ok")
    assert r and r[0]["ok"]==1, "Neo4j connection failed"
    print("✓ Connected to Neo4j")
if __name__=="__main__": main()
