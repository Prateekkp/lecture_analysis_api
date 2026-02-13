from fastapi import APIRouter, UploadFile, File, HTTPException
import tempfile
import os
import time
import uuid
import requests

from services.transcription import transcribe_audio
from services.nvidia_writer import generate_charter_compliant_output, generate_pedagogical_score
from middleware.validators import validate_audio_file
from middleware.privacy import SecureFileHandler, DataRetentionPolicy, sanitize_for_logging
from middleware.observability import StructuredLogger, metrics_tracker, track_operation
from middleware.failure_containment import ExternalAPIError
from models.responses import SuccessResponse, ErrorResponse, AnalysisResponse
from models.requests import AudioUrlRequest

router = APIRouter()
logger = StructuredLogger(__name__)


async def _process_audio_and_generate_response(request_id: str, audio_content: bytes, syllabus: str, start_time: float):
    temp_path = None
    try:
        # Create temporary file with privacy considerations
        temp_path = SecureFileHandler.create_temp_file(suffix=".mp3")
        
        # Write content
        with open(temp_path, "wb") as f:
            f.write(audio_content)
        
        logger.info("File saved temporarily", request_id=request_id, size_bytes=len(audio_content))
        
        # GUARANTEE 3: Failure Containment - Track transcription operation
        with track_operation("transcription", request_id):
            transcript = transcribe_audio(temp_path)
        
        logger.info(
            "Transcription completed",
            request_id=request_id,
            transcript_preview=sanitize_for_logging(transcript)
        )
        
        # GUARANTEE 3: Failure Containment - Track analysis operation
        with track_operation("analysis", request_id):
            analysis = generate_charter_compliant_output(transcript, syllabus)
        
        logger.info(
            "Analysis completed",
            request_id=request_id,
            analysis_preview=sanitize_for_logging(analysis)
        )
        
        # GUARANTEE 3: Failure Containment - Track scoring operation
        with track_operation("scoring", request_id):
            score_data = generate_pedagogical_score(analysis, transcript, syllabus)
        
        logger.info(
            "Scoring completed",
            request_id=request_id,
            score=score_data.get("score")
        )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # GUARANTEE 2: Predictable Outputs - Standardized response
        response_data = AnalysisResponse(
            analysis=analysis,
            pedagogical_score=score_data.get("score", 50),
            score_reasoning=score_data.get("reasoning", "Score generated"),
            processing_time_seconds=round(processing_time, 2)
        )
        
        return response_data

    finally:
        # GUARANTEE 4: Privacy by Design - Always clean up
        if temp_path:
            DataRetentionPolicy.apply(file_path=temp_path)


@router.post("/audio-to-document", response_model=SuccessResponse)
async def audio_to_document(audio: UploadFile = File(...), syllabus: str = ""):
    """
    Process audio file and return pedagogical analysis.
    
    Five Guarantees:
    1. Safe Inputs: File validation (type, size, content)
    2. Predictable Outputs: Standardized response schema
    3. Failure Containment: Try-catch with retries and circuit breakers
    4. Privacy by Design: Secure file deletion, no data persistence
    5. Operational Discipline: Structured logging and metrics
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()
    success = False
    
    try:
        logger.info("Received analysis request via file upload", request_id=request_id, filename=audio.filename)
        
        # GUARANTEE 1: Safe Inputs - Validate file
        await validate_audio_file(audio)
        logger.info("File validation passed", request_id=request_id)
        
        content = await audio.read()
        
        response_data = await _process_audio_and_generate_response(request_id, content, syllabus, start_time)
        
        success = True
        return SuccessResponse(request_id=request_id, data=response_data.dict())
    
    except ValueError as e:
        logger.error("Validation error", request_id=request_id, error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    
    except ExternalAPIError as e:
        logger.error("External API error", request_id=request_id, error=str(e))
        raise HTTPException(status_code=502, detail=f"External service error: {str(e)}")
    
    except Exception as e:
        logger.error("Unexpected error", request_id=request_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")
    
    finally:
        # GUARANTEE 5: Operational Discipline - Record metrics
        processing_time = time.time() - start_time
        metrics_tracker.record_request(processing_time, success, request_id)
        
        logger.info(
            "Request completed",
            request_id=request_id,
            success=success,
            total_time=f"{processing_time:.2f}s"
        )


import re

# ... (other imports)

def _get_download_url(url: str) -> str:
    """
    Convert a Google Drive sharing URL to a direct download link.
    Returns the original URL if it's not a Google Drive link.
    """
    match = re.search(r"/file/d/([^/]+)/", url)
    if "drive.google.com" in url and match:
        file_id = match.group(1)
        logger.info(f"Google Drive file ID detected: {file_id}")
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    return url

@router.post("/audio-url-to-document", response_model=SuccessResponse)
async def audio_url_to_document(request: AudioUrlRequest):
    """
    Process audio from a URL and return pedagogical analysis.
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()
    success = False

    try:
        logger.info("Received analysis request via URL", request_id=request_id, url=str(request.audio_url))

        download_url = _get_download_url(str(request.audio_url))

        # Download audio from URL
        try:
            # For Google Drive, you might need to handle redirects and large files
            with requests.get(download_url, stream=True, timeout=60) as response:
                response.raise_for_status()
                
                # Basic validation
                content_length = response.headers.get('content-length')
                if content_length and int(content_length) > 100 * 1024 * 1024: # 100MB limit
                     raise ValueError("File size exceeds 100MB limit.")

                # Stream the download to avoid high memory usage for large files
                audio_content = b""
                for chunk in response.iter_content(chunk_size=8192):
                    audio_content += chunk

            logger.info("File downloaded successfully", request_id=request_id, size_bytes=len(audio_content))

        except requests.exceptions.RequestException as e:
            logger.error("Failed to download file", request_id=request_id, error=str(e))
            raise HTTPException(status_code=400, detail=f"Failed to download audio from URL: {e}")

        response_data = await _process_audio_and_generate_response(request_id, audio_content, request.syllabus, start_time)
        
        success = True
        return SuccessResponse(request_id=request_id, data=response_data.dict())

    except ValueError as e:
        logger.error("Validation error", request_id=request_id, error=str(e))
        raise HTTPException(status_code=400, detail=str(e))

    except ExternalAPIError as e:
        logger.error("External API error", request_id=request_id, error=str(e))
        raise HTTPException(status_code=502, detail=f"External service error: {str(e)}")

    except Exception as e:
        logger.error("Unexpected error", request_id=request_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

    finally:
        # GUARANTEE 5: Operational Discipline - Record metrics
        processing_time = time.time() - start_time
        metrics_tracker.record_request(processing_time, success, request_id)
        
        logger.info(
            "Request completed",
            request_id=request_id,
            success=success,
            total_time=f"{processing_time:.2f}s"
        )
