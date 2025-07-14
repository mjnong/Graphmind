from functools import lru_cache
from src.app.db.session import SessionLocal
from src.app.db.services.files import FileService
from src.app.services.neo4j_driver import get_driver, Neo4jDriver

@lru_cache()
def get_file_service() -> FileService:
    return FileService(SessionLocal)

@lru_cache(maxsize=1)
def get_neo4j_driver() -> Neo4jDriver:
    return get_driver()
