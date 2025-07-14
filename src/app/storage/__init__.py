from .local import LocalStore
from .s3 import S3Store
from .azure import AzureStore
from .base import BaseStore
from functools import lru_cache
from src.app.configs.config import get_config
import os


@lru_cache
def get_store() -> BaseStore:
    backend = get_config().storage_backend
    if backend == "s3":
        return S3Store(
            bucket=os.environ["S3_BUCKET"],
            region=os.getenv("AWS_REGION"),
            endpoint_url=os.getenv("S3_ENDPOINT"),  # leave empty for AWS
            prefix=os.getenv("S3_PREFIX", ""),
            public=bool(int(os.getenv("S3_PUBLIC", "0"))),
        )
    elif backend == "local":
        return LocalStore(
            base_path=get_config().local_path,
            base_url=get_config().local_base_url,
        )
    elif backend == "azure":
        try:
            return AzureStore(
                account_name=os.environ["AZURE_STORAGE_ACCOUNT"],
                account_key=os.environ["AZURE_STORAGE_KEY"], 
                container_name=os.getenv("AZURE_CONTAINER_NAME", "uploads"),
                prefix=os.getenv("AZURE_PREFIX", ""),
                public=bool(int(os.getenv("AZURE_PUBLIC", "0"))),
            )
        except KeyError as e:
            raise ValueError(f"Azure storage backend requires environment variable: {e}")
        except ImportError as e:
            raise ValueError(f"Azure storage backend not available: {e}") 
    else:
        raise ValueError(f"Unknown storage backend: {backend}")
