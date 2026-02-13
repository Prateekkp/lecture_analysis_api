from fastapi import APIRouter, UploadFile, File, HTTPException
import tempfile
import os
import time
import uuid

from services.transcription import transcribe_audio
from services.nvidia_writer import generate_charter_compliant_output, generate_pedagogical_score
from middleware.validators import validate_audio_file
from middleware.privacy import SecureFileHandler, DataRetentionPolicy, sanitize_for_logging
from middleware.observability import StructuredLogger, metrics_tracker, track_operation
from middleware.failure_containment import ExternalAPIError
from models.responses import SuccessResponse, ErrorResponse, AnalysisResponse

router = APIRouter()
logger = StructuredLogger(__name__)


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
    temp_path = None
    success = False
    
    try:
        logger.info("Received analysis request", request_id=request_id, filename=audio.filename)
        
        # GUARANTEE 1: Safe Inputs - Validate file
        await validate_audio_file(audio)
        logger.info("File validation passed", request_id=request_id)
        
        # Create temporary file with privacy considerations
        temp_path = SecureFileHandler.create_temp_file(suffix=".mp3")
        
        # Write uploaded content
        content = await audio.read()
        with open(temp_path, "wb") as f:
            f.write(content)
        
        logger.info("File saved temporarily", request_id=request_id, size_bytes=len(content))
        
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
        
        success = True
        return SuccessResponse(request_id=request_id, data=response_data.dict())
    
    except ValueError as e:
        # Input validation errors
        logger.error("Validation error", request_id=request_id, error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    
    except ExternalAPIError as e:
        # External API failures
        logger.error("External API error", request_id=request_id, error=str(e))
        raise HTTPException(status_code=502, detail=f"External service error: {str(e)}")
    
    except Exception as e:
        # Unexpected errors
        logger.error("Unexpected error", request_id=request_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")
    
    finally:
        # GUARANTEE 4: Privacy by Design - Always clean up
        if temp_path:
            DataRetentionPolicy.apply(file_path=temp_path)
        
        # GUARANTEE 5: Operational Discipline - Record metrics
        processing_time = time.time() - start_time
        metrics_tracker.record_request(processing_time, success, request_id)
        
        logger.info(
            "Request completed",
            request_id=request_id,
            success=success,
            total_time=f"{processing_time:.2f}s"
        )
