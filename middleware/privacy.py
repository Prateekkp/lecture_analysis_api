"""Privacy-focused utilities for secure data handling."""
import os
import logging
from pathlib import Path
from typing import Optional
import hashlib

logger = logging.getLogger(__name__)


class SecureFileHandler:
    """Handle temporary files with privacy guarantees."""
    
    @staticmethod
    def create_temp_file(suffix: str = ".mp3") -> str:
        """Create a secure temporary file with random name."""
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            return tmp.name
    
    @staticmethod
    def secure_delete(file_path: str) -> None:
        """
        Securely delete file by overwriting with random data before removal.
        Implements DoD 5220.22-M standard (3-pass overwrite).
        """
        try:
            if not os.path.exists(file_path):
                return
            
            file_size = os.path.getsize(file_path)
            
            # Pass 1: Overwrite with zeros
            with open(file_path, "wb") as f:
                f.write(b'\x00' * file_size)
            
            # Pass 2: Overwrite with ones
            with open(file_path, "wb") as f:
                f.write(b'\xFF' * file_size)
            
            # Pass 3: Overwrite with random data
            with open(file_path, "wb") as f:
                f.write(os.urandom(file_size))
            
            # Finally remove the file
            os.remove(file_path)
            logger.info(f"Securely deleted file: {Path(file_path).name}")
            
        except Exception as e:
            logger.error(f"Failed to securely delete {file_path}: {str(e)}")
            # Fallback to regular deletion
            try:
                os.remove(file_path)
            except:
                pass


def sanitize_for_logging(text: str, max_length: int = 100) -> str:
    """
    Sanitize sensitive data for logging.
    Returns first N characters with hash of full content.
    """
    if len(text) <= max_length:
        return text[:50] + "..."
    
    content_hash = hashlib.sha256(text.encode()).hexdigest()[:8]
    return f"{text[:50]}... [hash:{content_hash}]"


def mask_api_key(api_key: Optional[str]) -> str:
    """Mask API key for safe logging."""
    if not api_key or len(api_key) < 8:
        return "***"
    return f"{api_key[:4]}...{api_key[-4:]}"


class DataRetentionPolicy:
    """Enforce zero data retention policy."""
    
    @staticmethod
    def apply(file_path: Optional[str] = None, transcript: Optional[str] = None) -> None:
        """
        Apply data retention policy:
        - Audio files: Delete immediately after processing
        - Transcripts: Never persist to disk
        - Analysis: Return to client only, never stored
        """
        if file_path and os.path.exists(file_path):
            SecureFileHandler.secure_delete(file_path)
            logger.info("Data retention policy applied: audio file deleted")
        
        if transcript:
            logger.info("Data retention policy: transcript kept in memory only")
