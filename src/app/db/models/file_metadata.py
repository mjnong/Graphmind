from datetime import datetime

from sqlalchemy import BigInteger, Integer, Text, func, DateTime
from sqlalchemy.orm import Mapped, mapped_column

from src.app.db.base import Base


class Upload(Base):
    """Upload model representing the uploads table."""
    
    __tablename__ = "uploads"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    mime_type: Mapped[str] = mapped_column(Text, nullable=False)
    object_key: Mapped[str] = mapped_column(Text, nullable=False)  # e.g. "uploads/2025/07/..."
    status: Mapped[str] = mapped_column(Text, nullable=False, default="uploaded")
    progress: Mapped[int] = mapped_column(Integer, nullable=False, default=0)  # 0-100
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    def __repr__(self) -> str:
        return f"<Upload(id={self.id}, filename={self.filename}, status={self.status})>"
