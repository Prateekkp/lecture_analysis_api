"""Operational discipline: logging, monitoring, and observability."""
import logging
import sys
import time
from typing import Optional
from datetime import datetime
from contextlib import contextmanager


class StructuredLogger:
    """Structured logging for operational visibility."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self._configure_logging()
    
    def _configure_logging(self):
        """Configure logging format and handlers."""
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def info(self, message: str, **kwargs):
        """Log info with structured context."""
        extra_info = " | ".join(f"{k}={v}" for k, v in kwargs.items())
        self.logger.info(f"{message} | {extra_info}" if extra_info else message)
    
    def error(self, message: str, **kwargs):
        """Log error with structured context."""
        extra_info = " | ".join(f"{k}={v}" for k, v in kwargs.items())
        self.logger.error(f"{message} | {extra_info}" if extra_info else message)
    
    def warning(self, message: str, **kwargs):
        """Log warning with structured context."""
        extra_info = " | ".join(f"{k}={v}" for k, v in kwargs.items())
        self.logger.warning(f"{message} | {extra_info}" if extra_info else message)


class RequestMetrics:
    """Track request metrics for monitoring."""
    
    def __init__(self):
        self.total_requests = 0
        self.total_errors = 0
        self.total_processing_time = 0.0
        self.logger = StructuredLogger("metrics")
    
    def record_request(self, processing_time: float, success: bool, request_id: str):
        """Record request metrics."""
        self.total_requests += 1
        self.total_processing_time += processing_time
        
        if not success:
            self.total_errors += 1
        
        self.logger.info(
            "Request completed",
            request_id=request_id,
            processing_time=f"{processing_time:.2f}s",
            success=success
        )
    
    def get_metrics(self) -> dict:
        """Get current metrics snapshot."""
        avg_time = (
            self.total_processing_time / self.total_requests
            if self.total_requests > 0
            else 0.0
        )
        
        error_rate = (
            (self.total_errors / self.total_requests) * 100
            if self.total_requests > 0
            else 0.0
        )
        
        return {
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "error_rate_percent": round(error_rate, 2),
            "average_processing_time_seconds": round(avg_time, 2)
        }


# Global metrics tracker
metrics_tracker = RequestMetrics()


@contextmanager
def track_operation(operation_name: str, request_id: str):
    """
    Context manager to track operation timing and success.
    
    Usage:
        with track_operation("transcription", request_id):
            result = transcribe_audio(file)
    """
    logger = StructuredLogger("operations")
    start_time = time.time()
    
    logger.info(
        f"Starting {operation_name}",
        request_id=request_id,
        timestamp=datetime.utcnow().isoformat()
    )
    
    try:
        yield
        duration = time.time() - start_time
        logger.info(
            f"Completed {operation_name}",
            request_id=request_id,
            duration=f"{duration:.2f}s",
            status="success"
        )
    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            f"Failed {operation_name}",
            request_id=request_id,
            duration=f"{duration:.2f}s",
            error=str(e),
            status="failed"
        )
        raise
