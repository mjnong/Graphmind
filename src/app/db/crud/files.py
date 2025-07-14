from typing import Optional

from sqlalchemy.orm import Session

from src.app.db.models.file_metadata import Upload
from src.app.db.schemas.file_metadata import (
    PaginationParams,
    UploadCreate,
    UploadUpdate,
)


def create_upload(db: Session, upload_data: UploadCreate) -> Upload:
    """Create a new upload record."""
    upload = Upload(**upload_data.model_dump())
    db.add(upload)
    db.commit()
    db.refresh(upload)
    return upload


def get_upload(db: Session, upload_id: int) -> Optional[Upload]:
    """Get an upload by ID."""
    return db.query(Upload).filter(Upload.id == upload_id).first()


def get_upload_by_object_key(db: Session, object_key: str) -> Optional[Upload]:
    """Get an upload by object key."""
    return db.query(Upload).filter(Upload.object_key == object_key).first()


def get_uploads(db: Session, skip: int = 0, limit: int = 100) -> list[Upload]:
    """Get multiple uploads with pagination."""
    return db.query(Upload).offset(skip).limit(limit).all()


def get_uploads_by_status(db: Session, status: str, skip: int = 0, limit: int = 100) -> list[Upload]:
    """Get uploads by status."""
    return db.query(Upload).filter(Upload.status == status).offset(skip).limit(limit).all()


def get_uploads_paginated(db: Session, pagination: PaginationParams) -> tuple[list[Upload], int]:
    """Get paginated uploads with total count."""
    query = db.query(Upload)
    
    # Get total count
    total_count = query.count()
    
    # Get paginated results
    uploads = query.offset(pagination.skip).limit(pagination.limit).all()
    
    return uploads, total_count


def get_uploads_by_status_paginated(
    db: Session, status: str, pagination: PaginationParams
) -> tuple[list[Upload], int]:
    """Get paginated uploads by status with total count."""
    query = db.query(Upload).filter(Upload.status == status)
    
    # Get total count
    total_count = query.count()
    
    # Get paginated results
    uploads = query.offset(pagination.skip).limit(pagination.limit).all()
    
    return uploads, total_count


def update_upload(db: Session, upload_id: int, upload_data: UploadUpdate) -> Optional[Upload]:
    """Update an upload record."""
    upload = db.query(Upload).filter(Upload.id == upload_id).first()
    if not upload:
        return None
    
    # Update only provided fields
    update_data = upload_data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(upload, field, value)
    
    db.commit()
    db.refresh(upload)
    return upload


def update_upload_status(db: Session, upload_id: int, status: str) -> Optional[Upload]:
    """Update upload status."""
    upload = db.query(Upload).filter(Upload.id == upload_id).first()
    if not upload:
        return None
    
    upload.status = status
    db.commit()
    db.refresh(upload)
    return upload


def update_upload_progress(db: Session, upload_id: int, progress: int) -> Optional[Upload]:
    """Update upload progress."""
    upload = db.query(Upload).filter(Upload.id == upload_id).first()
    if not upload:
        return None
    
    upload.progress = max(0, min(100, progress))  # Ensure progress is between 0-100
    db.commit()
    db.refresh(upload)
    return upload


def delete_upload(db: Session, upload_id: int) -> bool:
    """Delete an upload record."""
    upload = db.query(Upload).filter(Upload.id == upload_id).first()
    if not upload:
        return False
    
    db.delete(upload)
    db.commit()
    return True


def count_uploads(db: Session) -> int:
    """Count total uploads."""
    return db.query(Upload).count()


def count_uploads_by_status(db: Session, status: str) -> int:
    """Count uploads by status."""
    return db.query(Upload).filter(Upload.status == status).count()


def get_recent_uploads(db: Session, limit: int = 10) -> list[Upload]:
    """Get most recently created uploads."""
    return (
        db.query(Upload)
        .order_by(Upload.created_at.desc())
        .limit(limit)
        .all()
    )


def search_uploads_by_filename(db: Session, filename_pattern: str, skip: int = 0, limit: int = 100) -> list[Upload]:
    """Search uploads by filename pattern."""
    return (
        db.query(Upload)
        .filter(Upload.filename.ilike(f"%{filename_pattern}%"))
        .offset(skip)
        .limit(limit)
        .all()
    )
