"""File validation utilities for secure file uploads."""

import os
from typing import List, Dict, Any, Optional
from fastapi import UploadFile

from src.app.configs.config import get_config


class ValidationError(Exception):
    """Custom exception for file validation errors."""
    pass


class FileValidator:
    """Validates uploaded files for security and compliance."""
    
    # Allowed MIME types with their magic number signatures
    ALLOWED_TYPES = {
        # Images
        "image/jpeg": [
            b"\xff\xd8\xff\xe0",  # JPEG JFIF
            b"\xff\xd8\xff\xe1",  # JPEG EXIF
            b"\xff\xd8\xff\xdb",  # JPEG
        ],
        "image/png": [
            b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a",  # PNG
        ],
        # Documents
        "application/pdf": [
            b"\x25\x50\x44\x46",  # %PDF
        ],
        # Text files
        "text/plain": [
            # Text files don't have reliable magic numbers, so we'll use content inspection
        ],
    }
    
    # File extensions for additional validation
    ALLOWED_EXTENSIONS = {
        ".jpg", ".jpeg", ".png", ".pdf", ".txt"
    }
    
    def __init__(self):
        self.config = get_config()
        self.max_size_bytes = self.config.max_file_size_mb * 1024 * 1024
        
    async def validate_file(self, file: UploadFile) -> Dict[str, Any]:
        """
        Comprehensive file validation including magic number checking.
        
        Args:
            file: The uploaded file to validate
            
        Returns:
            Dict containing validation results and file metadata
            
        Raises:
            ValidationError: If validation fails
        """
        validation_result = {
            "valid": False,
            "filename": file.filename,
            "content_type": file.content_type,
            "size": 0,
            "detected_type": None,
            "errors": []
        }
        
        try:
            # 1. Basic checks
            if not file.filename:
                raise ValidationError("Filename is required")
                
            # 2. File extension validation
            file_ext = os.path.splitext(file.filename)[1].lower()
            if file_ext not in self.ALLOWED_EXTENSIONS:
                raise ValidationError(
                    f"File extension '{file_ext}' not allowed. "
                    f"Allowed extensions: {', '.join(self.ALLOWED_EXTENSIONS)}"
                )
            
            # 3. Read file content for validation
            content = await file.read()
            await file.seek(0)  # Reset file pointer
            
            validation_result["size"] = len(content)
            
            # 4. File size validation
            if len(content) == 0:
                raise ValidationError("File is empty")
                
            if len(content) > self.max_size_bytes:
                raise ValidationError(
                    f"File size ({len(content) / 1024 / 1024:.1f} MB) exceeds "
                    f"maximum allowed size ({self.config.max_file_size_mb} MB)"
                )
            
            # 5. Magic number validation
            detected_type = self._detect_file_type(content)
            validation_result["detected_type"] = detected_type
            
            if not detected_type:
                raise ValidationError("Could not determine file type from content")
            
            # 6. Content-type vs magic number consistency
            if file.content_type and file.content_type != detected_type:
                # Allow some flexibility for text files
                if not (file.content_type == "text/plain" and detected_type == "text/plain"):
                    raise ValidationError(
                        f"Content-Type header '{file.content_type}' does not match "
                        f"detected file type '{detected_type}'"
                    )
            
            # 7. Additional content validation
            self._validate_file_content(content, detected_type)
            
            validation_result["valid"] = True
            return validation_result
            
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"Validation failed: {str(e)}")
    
    def _detect_file_type(self, content: bytes) -> Optional[str]:
        """
        Detect file type using magic numbers.
        
        Args:
            content: File content as bytes
            
        Returns:
            Detected MIME type or None if not recognized
        """
        # Check magic numbers
        for mime_type, signatures in self.ALLOWED_TYPES.items():
            if mime_type == "text/plain":
                # Special handling for text files
                if self._is_text_file(content):
                    return "text/plain"
                continue
                
            for signature in signatures:
                if content.startswith(signature):
                    return mime_type
        
        # Fallback: try python-magic if available
        try:
            import magic
            mime = magic.from_buffer(content, mime=True)
            if mime in self.ALLOWED_TYPES:
                return mime
        except (ImportError, Exception):
            # python-magic not available or error in detection, continue with basic detection
            pass
        
        return None
    
    def _is_text_file(self, content: bytes) -> bool:
        """
        Check if content appears to be a text file.
        
        Args:
            content: File content as bytes
            
        Returns:
            True if content appears to be text
        """
        try:
            # Try to decode as UTF-8
            content.decode('utf-8')
            
            # Check for common text characteristics
            text_chars = sum(1 for byte in content[:1000] if 32 <= byte <= 126 or byte in [9, 10, 13])
            total_chars = min(len(content), 1000)
            
            if total_chars == 0:
                return False
                
            # If more than 85% of characters are printable, consider it text
            return (text_chars / total_chars) > 0.85
            
        except UnicodeDecodeError:
            return False
    
    def _validate_file_content(self, content: bytes, file_type: str) -> None:
        """
        Additional content validation based on file type.
        
        Args:
            content: File content as bytes
            file_type: Detected MIME type
            
        Raises:
            ValidationError: If content validation fails
        """
        if file_type == "application/pdf":
            self._validate_pdf_content(content)
        elif file_type.startswith("image/"):
            self._validate_image_content(content, file_type)
        elif file_type == "text/plain":
            self._validate_text_content(content)
    
    def _validate_pdf_content(self, content: bytes) -> None:
        """Validate PDF file content."""
        # Check for PDF version
        if not content.startswith(b"%PDF-"):
            raise ValidationError("Invalid PDF file structure")
        
        # Check for PDF end marker
        if b"%%EOF" not in content[-1000:]:
            raise ValidationError("PDF file appears to be truncated or corrupted")
    
    def _validate_image_content(self, content: bytes, file_type: str) -> None:
        """Validate image file content."""
        # Basic structure validation already done by magic numbers
        # Additional checks could be added here using PIL/Pillow if needed
        pass
    
    def _validate_text_content(self, content: bytes) -> None:
        """Validate text file content."""
        try:
            # Decode to validate UTF-8 encoding
            content.decode('utf-8')
            
            # Check for suspicious content patterns
            suspicious_patterns = [
                b"<script", b"javascript:", b"eval(", b"exec("
            ]
            
            content_lower = content.lower()
            for pattern in suspicious_patterns:
                if pattern in content_lower:
                    raise ValidationError(
                        f"Text file contains potentially unsafe content: {pattern.decode()}"
                    )
                    
        except UnicodeDecodeError:
            raise ValidationError("Text file contains invalid UTF-8 encoding")
    
    def validate_multiple_files(self, files: List[UploadFile]) -> Dict[str, Any]:
        """
        Validate multiple files at once.
        
        Args:
            files: List of uploaded files to validate
            
        Returns:
            Dict containing validation results for all files
            
        Raises:
            ValidationError: If any validation fails
        """
        if len(files) > self.config.max_files_per_upload:
            raise ValidationError(
                f"Too many files. Maximum {self.config.max_files_per_upload} files allowed per upload"
            )
        
        total_size = 0
        results = []
        
        for i, file in enumerate(files):
            try:
                # Note: We can't use async here, so we'll need to handle this at the endpoint level
                file_size = 0
                if hasattr(file, 'size') and file.size:
                    file_size = file.size
                
                total_size += file_size
                
                # Check total size limit (conservative estimate)
                if total_size > self.max_size_bytes * len(files):
                    raise ValidationError(
                        f"Combined file size too large. Maximum total size: "
                        f"{self.config.max_file_size_mb * self.config.max_files_per_upload} MB"
                    )
                
                results.append({
                    "index": i,
                    "filename": file.filename,
                    "preliminary_check": "passed"
                })
                
            except Exception as e:
                raise ValidationError(f"File {i} ({file.filename}): {str(e)}")
        
        return {
            "total_files": len(files),
            "estimated_total_size": total_size,
            "preliminary_results": results
        }


# Global validator instance
_validator_instance = None

def get_file_validator() -> FileValidator:
    """Get the global file validator instance."""
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = FileValidator()
    return _validator_instance
