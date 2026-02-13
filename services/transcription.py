from openai import OpenAI
from config import OPENAI_API_KEY
from middleware.failure_containment import retry_with_backoff, ExternalAPIError, openai_circuit_breaker
from middleware.observability import StructuredLogger
import requests

client = OpenAI(api_key=OPENAI_API_KEY, timeout=30.0)
logger = StructuredLogger(__name__)


@retry_with_backoff(max_retries=3, exceptions=(Exception,))
def transcribe_audio(file_path: str) -> str:
    """
    Transcribe audio file using OpenAI Whisper API.
    
    Guarantees:
    - Retry logic with exponential backoff (Failure Containment)
    - Circuit breaker protection (Failure Containment)
    - Timeout protection (Safe Inputs)
    - Structured logging (Operational Discipline)
    """
    try:
        logger.info(f"Starting transcription", file=file_path)
        
        def _transcribe():
            with open(file_path, "rb") as audio:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio
                )
            return transcript.text
        
        # Use circuit breaker
        result = openai_circuit_breaker.call(_transcribe)
        
        logger.info(f"Transcription completed", length=len(result))
        return result
        
    except Exception as e:
        logger.error(f"Transcription failed", error=str(e))
        raise ExternalAPIError(f"OpenAI transcription failed: {str(e)}") from e
