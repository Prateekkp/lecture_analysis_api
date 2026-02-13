from fastapi import FastAPI
from api.routes import router
from middleware.observability import metrics_tracker

app = FastAPI(
    title="Audio to English Document API",
    description="Lecture analysis API with five core guarantees: Safe Inputs, Predictable Outputs, Failure Containment, Privacy by Design, and Operational Discipline",
    version="1.0.0"
)

app.include_router(router)


@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring.
    
    Returns:
    - status: Service health status
    - metrics: Operational metrics
    """
    return {
        "status": "healthy",
        "service": "lecture_analysis_api",
        "metrics": metrics_tracker.get_metrics()
    }


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Audio to English Document API",
        "version": "1.0.0",
        "guarantees": [
            "Safe Inputs: File validation (type, size, content)",
            "Predictable Outputs: Standardized response schemas",
            "Failure Containment: Retry logic, circuit breakers, timeouts",
            "Privacy by Design: Secure deletion, zero data retention",
            "Operational Discipline: Structured logging and metrics"
        ],
        "endpoints": {
            "/health": "Health check with metrics",
            "/audio-to-document": "POST audio file for pedagogical analysis"
        }
    }
