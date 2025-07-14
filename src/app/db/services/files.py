from contextlib import contextmanager
from typing import Generator, Optional

from sqlalchemy.orm import Session

from src.app.db.crud.files import (
    count_uploads,
    count_uploads_by_status,
    create_upload,
    delete_upload,
    get_recent_uploads,
    get_upload,
    get_upload_by_object_key,
    get_uploads,
    get_uploads_by_status,
    get_uploads_by_status_paginated,
    get_uploads_paginated,
    search_uploads_by_filename,
    update_upload,
    update_upload_progress,
    update_upload_status,
)
from src.app.db.schemas.file_metadata import (
    PaginatedResponse,
    PaginationParams,
    UploadCreate,
    UploadRead,
    UploadUpdate,
)


class FileService:
    """Service class for managing file uploads in the application.
    
    This class provides methods to create, update, delete, and retrieve upload records.
    It handles the business logic for file operations and manages database transactions.
    """

    def __init__(self, session_factory):
        """Initialize the FileService with a session factory."""
        self._session_factory = session_factory

    def _upload_to_upload_response(self, upload) -> UploadRead:
        """Convert an upload object to an UploadRead response."""
        return UploadRead(
            id=upload.id,
            filename=upload.filename,
            mime_type=upload.mime_type,
            object_key=upload.object_key,
            status=upload.status,
            progress=upload.progress,
            created_at=upload.created_at,
            updated_at=upload.updated_at,
        )

    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """Context manager to get a database session with proper cleanup."""
        db = self._session_factory()
        try:
            yield db
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()

    def create_upload(self, upload_data: UploadCreate) -> UploadRead:
        """Create a new upload record."""
        with self.session_scope() as db:
            upload = create_upload(db, upload_data)
            if not upload:
                raise ValueError("Failed to create upload")
            return UploadRead.model_validate(self._upload_to_upload_response(upload))

    def get_upload_by_id(self, upload_id: int) -> Optional[UploadRead]:
        """Get an upload by its ID."""
        with self.session_scope() as db:
            upload = get_upload(db, upload_id)
            if upload:
                return UploadRead.model_validate(self._upload_to_upload_response(upload))
            return None

    def get_upload_by_object_key(self, object_key: str) -> Optional[UploadRead]:
        """Get an upload by its object key."""
        with self.session_scope() as db:
            upload = get_upload_by_object_key(db, object_key)
            if upload:
                return UploadRead.model_validate(self._upload_to_upload_response(upload))
            return None

    def get_uploads(self, skip: int = 0, limit: int = 100) -> list[UploadRead]:
        """Get multiple uploads with pagination."""
        with self.session_scope() as db:
            uploads = get_uploads(db, skip, limit)
            return [
                UploadRead.model_validate(self._upload_to_upload_response(upload))
                for upload in uploads
            ]

    def get_uploads_by_status(self, status: str, skip: int = 0, limit: int = 100) -> list[UploadRead]:
        """Get uploads filtered by status."""
        with self.session_scope() as db:
            uploads = get_uploads_by_status(db, status, skip, limit)
            return [
                UploadRead.model_validate(self._upload_to_upload_response(upload))
                for upload in uploads
            ]

    def get_uploads_paginated(self, pagination: PaginationParams) -> PaginatedResponse[UploadRead]:
        """Get paginated uploads."""
        with self.session_scope() as db:
            uploads, total_count = get_uploads_paginated(db, pagination)
            upload_reads = [
                UploadRead.model_validate(self._upload_to_upload_response(upload))
                for upload in uploads
            ]
            return PaginatedResponse.create(
                items=upload_reads,
                total_items=total_count,
                pagination_params=pagination,
            )

    def get_uploads_by_status_paginated(
        self, status: str, pagination: PaginationParams
    ) -> PaginatedResponse[UploadRead]:
        """Get paginated uploads filtered by status."""
        with self.session_scope() as db:
            uploads, total_count = get_uploads_by_status_paginated(db, status, pagination)
            upload_reads = [
                UploadRead.model_validate(self._upload_to_upload_response(upload))
                for upload in uploads
            ]
            return PaginatedResponse.create(
                items=upload_reads,
                total_items=total_count,
                pagination_params=pagination,
            )

    def update_upload(self, upload_id: int, upload_data: UploadUpdate) -> Optional[UploadRead]:
        """Update an upload record."""
        with self.session_scope() as db:
            upload = update_upload(db, upload_id, upload_data)
            if upload:
                return UploadRead.model_validate(self._upload_to_upload_response(upload))
            return None

    def update_upload_status(self, upload_id: int, status: str) -> Optional[UploadRead]:
        """Update the status of an upload."""
        with self.session_scope() as db:
            upload = update_upload_status(db, upload_id, status)
            if upload:
                return UploadRead.model_validate(self._upload_to_upload_response(upload))
            return None

    def update_upload_progress(self, upload_id: int, progress: int) -> Optional[UploadRead]:
        """Update the progress of an upload."""
        with self.session_scope() as db:
            upload = update_upload_progress(db, upload_id, progress)
            if upload:
                return UploadRead.model_validate(self._upload_to_upload_response(upload))
            return None

    def delete_upload(self, upload_id: int) -> bool:
        """Delete an upload record."""
        with self.session_scope() as db:
            return delete_upload(db, upload_id)

    def get_upload_count(self) -> int:
        """Get the total count of uploads."""
        with self.session_scope() as db:
            return count_uploads(db)

    def get_upload_count_by_status(self, status: str) -> int:
        """Get the count of uploads by status."""
        with self.session_scope() as db:
            return count_uploads_by_status(db, status)

    def get_recent_uploads(self, limit: int = 10) -> list[UploadRead]:
        """Get the most recently created uploads."""
        with self.session_scope() as db:
            uploads = get_recent_uploads(db, limit)
            return [
                UploadRead.model_validate(self._upload_to_upload_response(upload))
                for upload in uploads
            ]

    def search_uploads_by_filename(
        self, filename_pattern: str, skip: int = 0, limit: int = 100
    ) -> list[UploadRead]:
        """Search uploads by filename pattern."""
        with self.session_scope() as db:
            uploads = search_uploads_by_filename(db, filename_pattern, skip, limit)
            return [
                UploadRead.model_validate(self._upload_to_upload_response(upload))
                for upload in uploads
            ]

    # Convenience methods for common operations
    def mark_upload_as_processing(self, upload_id: int) -> Optional[UploadRead]:
        """Mark an upload as being processed."""
        return self.update_upload_status(upload_id, "processing")

    def mark_upload_as_complete(self, upload_id: int) -> Optional[UploadRead]:
        """Mark an upload as complete."""
        result = self.update_upload_status(upload_id, "complete")
        if result:
            # Set progress to 100% when marking as complete
            result = self.update_upload_progress(upload_id, 100)
        return result

    def mark_upload_as_failed(self, upload_id: int) -> Optional[UploadRead]:
        """Mark an upload as failed."""
        return self.update_upload_status(upload_id, "failed")

    def increment_upload_progress(self, upload_id: int, increment: int = 10) -> Optional[UploadRead]:
        """Increment upload progress by a specified amount."""
        upload = self.get_upload_by_id(upload_id)
        if upload:
            new_progress = min(100, upload.progress + increment)
            return self.update_upload_progress(upload_id, new_progress)
        return None
