# app/storage/azure.py
from pathlib import Path
from typing import BinaryIO
from datetime import datetime, timedelta

try:
    from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
except ImportError:
    # Azure SDK not installed
    BlobServiceClient = None
    generate_blob_sas = None
    BlobSasPermissions = None

from .base import BaseStore

class AzureStore(BaseStore):
    """
    Wraps Azure Blob Storage service.
    Requires Azure Storage Account credentials via environment variables
    or explicit constructor parameters.
    """

    def __init__(
        self,
        account_name: str,
        account_key: str,
        container_name: str,
        public: bool = False,
        prefix: str = "",
    ):
        if BlobServiceClient is None:
            raise ImportError(
                "Azure Blob Storage dependencies not installed. "
                "Install with: pip install azure-storage-blob"
            )
            
        self.account_name = account_name
        self.account_key = account_key
        self.container_name = container_name
        self.prefix = prefix.strip("/")
        self.public = public  # if True: return raw https URL instead of SAS URL
        
        # Initialize the blob service client
        account_url = f"https://{account_name}.blob.core.windows.net"
        self.blob_service_client = BlobServiceClient(
            account_url=account_url,
            credential=account_key
        )
        
        # Ensure container exists
        self._ensure_container_exists()

    # ---------- helpers ---------- #
    def _ensure_container_exists(self) -> None:
        """Create container if it doesn't exist"""
        try:
            container_client = self.blob_service_client.get_container_client(self.container_name)
            container_client.get_container_properties()
        except Exception:
            # Container doesn't exist, create it
            self.blob_service_client.create_container(self.container_name)

    def _blob_name(self, object_key: str) -> str:
        """Convert object_key to blob name with optional prefix"""
        return f"{self.prefix}/{object_key}" if self.prefix else object_key

    # ---------- API ---------- #
    def put(self, file: BinaryIO, object_key: str) -> None:
        """Upload file to Azure Blob Storage"""
        blob_name = self._blob_name(object_key)
        blob_client = self.blob_service_client.get_blob_client(
            container=self.container_name,
            blob=blob_name
        )
        blob_client.upload_blob(file, overwrite=True)

    def get(self, object_key: str, dest: Path) -> Path:
        """Download blob to local destination"""
        blob_name = self._blob_name(object_key)
        blob_client = self.blob_service_client.get_blob_client(
            container=self.container_name,
            blob=blob_name
        )
        
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        
        with open(dest, "wb") as download_file:
            download_stream = blob_client.download_blob()
            download_file.write(download_stream.readall())
        
        return dest

    def delete(self, object_key: str) -> None:
        """Delete blob from Azure Storage"""
        blob_name = self._blob_name(object_key)
        blob_client = self.blob_service_client.get_blob_client(
            container=self.container_name,
            blob=blob_name
        )
        try:
            blob_client.delete_blob()
        except Exception:
            # Blob might not exist, ignore the error
            pass

    def url(self, object_key: str, expires: int = 3600) -> str:
        """Generate a publicly accessible URL (SAS URL or public URL)"""
        blob_name = self._blob_name(object_key)
        
        if self.public:
            # Return direct URL if container/blob is public
            return f"https://{self.account_name}.blob.core.windows.net/{self.container_name}/{blob_name}"
        
        # Generate SAS URL with read permissions
        if generate_blob_sas is None or BlobSasPermissions is None:
            raise ImportError("Azure SDK components not available for SAS generation")
            
        expiry_time = datetime.utcnow() + timedelta(seconds=expires)
        
        sas_token = generate_blob_sas(
            account_name=self.account_name,
            account_key=self.account_key,
            container_name=self.container_name,
            blob_name=blob_name,
            permission=BlobSasPermissions(read=True),
            expiry=expiry_time
        )
        
        return f"https://{self.account_name}.blob.core.windows.net/{self.container_name}/{blob_name}?{sas_token}"
