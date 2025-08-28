import os
from neo4j import GraphDatabase
from dotenv import load_dotenv
from tenacity import retry, wait_exponential_jitter, stop_after_attempt, retry_if_exception
from neo4j.exceptions import ServiceUnavailable, TransientError, SessionExpired

load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "test1234")

_driver = None
def get_driver():
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    return _driver

RETRYABLE = (ServiceUnavailable, TransientError, SessionExpired)
def _is_retryable(e: Exception) -> bool:
    return isinstance(e, RETRYABLE)

@retry(wait=wait_exponential_jitter(initial=0.5, max=8.0),
       stop=stop_after_attempt(5),
       retry=retry_if_exception(_is_retryable))
def run_query(cypher: str, params: dict | None = None):
    drv = get_driver()
    with drv.session() as s:
        return s.run(cypher, params or {}).data()

@retry(wait=wait_exponential_jitter(initial=0.5, max=8.0),
       stop=stop_after_attempt(5),
       retry=retry_if_exception(_is_retryable))
def write_tx(func, *args, **kwargs):
    drv = get_driver()
    with drv.session() as s:
        return s.execute_write(func, *args, **kwargs)
