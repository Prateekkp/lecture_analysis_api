"""Input validation middleware for safe inputs guarantee."""
from fastapi import UploadFile, HTTPException
from typing import Set

# Safe input constants
MAX_FILE_SIZE_MB = 25
ALLOWED_AUDIO_TYPES: Set[str] = {
    "audio/mpeg",
    "audio/mp3",
    "audio/wav",
    "audio/m4a",
    "audio/x-m4a",
    "audio/mp4"
}
ALLOWED_EXTENSIONS: Set[str] = {".mp3", ".wav", ".m4a", ".mp4"}


async def validate_audio_file(file: UploadFile) -> None:
    """
    Validate uploaded audio file for:
    - File size limits
    - Content type
    - File extension
    
    Raises HTTPException if validation fails.
    """
    # Check content type
    if file.content_type not in ALLOWED_AUDIO_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Must be audio file."
        )
    
    # Check file extension
    if file.filename:
        ext = "." + file.filename.split(".")[-1].lower() if "." in file.filename else ""
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file extension: {ext}. Allowed: {ALLOWED_EXTENSIONS}"
            )
    
    # Check file size by reading in chunks
    file_size = 0
    max_size_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    
    # Read file to validate size and reset pointer
    content = await file.read()
    file_size = len(content)
    
    if file_size > max_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {file_size / 1024 / 1024:.2f}MB. Max: {MAX_FILE_SIZE_MB}MB"
        )
    
    if file_size == 0:
        raise HTTPException(
            status_code=400,
            detail="File is empty"
        )
    
    # Reset file pointer for downstream processing
    await file.seek(0)


def validate_api_keys(openai_key: str | None, nvidia_key: str | None) -> None:
    """Validate that required API keys are present."""
    if not openai_key or len(openai_key.strip()) == 0:
        raise ValueError("OPENAI_API_KEY is missing or empty in environment")
    
    if not nvidia_key or len(nvidia_key.strip()) == 0:
        raise ValueError("NVIDIA_API_KEY is missing or empty in environment")
