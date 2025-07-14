from datetime import datetime
from typing import Generic, List, Optional, TypeVar

from pydantic import BaseModel, ConfigDict


class UploadBase(BaseModel):
    """Base upload schema with common fields."""
    filename: str
    mime_type: str
    object_key: str
    status: str = "uploaded"
    progress: int = 0


class UploadCreate(UploadBase):
    """Schema for creating a new upload."""
    pass


class UploadUpdate(BaseModel):
    """Schema for updating an upload."""
    filename: Optional[str] = None
    mime_type: Optional[str] = None
    object_key: Optional[str] = None
    status: Optional[str] = None
    progress: Optional[int] = None


class UploadRead(UploadBase):
    """Schema for reading upload data."""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    created_at: datetime
    updated_at: datetime


class UploadInDB(UploadRead):
    """Schema for upload data as stored in database."""
    pass


# Pagination schemas
class PaginationParams(BaseModel):
    """Pagination parameters."""
    page: int = 1
    page_size: int = 20
    
    @property
    def skip(self) -> int:
        """Calculate the number of items to skip."""
        return (self.page - 1) * self.page_size
    
    @property
    def limit(self) -> int:
        """Get the limit for the query."""
        return self.page_size


T = TypeVar('T')

class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated response."""
    items: List[T]
    total_items: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_previous: bool
    
    @classmethod
    def create(cls, items: List[T], total_items: int, pagination_params: PaginationParams):
        """Create a paginated response."""
        total_pages = (total_items + pagination_params.page_size - 1) // pagination_params.page_size
        
        return cls(
            items=items,
            total_items=total_items,
            page=pagination_params.page,
            page_size=pagination_params.page_size,
            total_pages=total_pages,
            has_next=pagination_params.page < total_pages,
            has_previous=pagination_params.page > 1,
        )
