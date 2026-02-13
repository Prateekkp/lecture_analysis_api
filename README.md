# Multilingual Audio to Document API

**API for Automated Pedagogical Analysis**

A FastAPI-based service that transcribes audio lectures (supporting English, Hindi, and Hinglish) and performs deep pedagogical analysis with context-aware scoring. Built with five core reliability guarantees for enterprise environments.

---

## Overview

The Lecture Analysis API automatically evaluates teaching methodology by analyzing:
- Question frequency and quality
- Student wait time and engagement patterns  
- Participation balance between instructor and students
- Cultural contextualization through code-switching

The system provides **human-readable analysis** and **accuracy-based scoring** (0-100) that accounts for curriculum alignment when a syllabus is provided.

**Key Features:**
- Automated audio transcription (OpenAI Whisper)
- Supports multilingual audio: English, Hindi, Hinglish
- Intelligent pedagogical analysis (NVIDIA API)
- Context-aware scoring with syllabus alignment
- Enterprise-grade reliability with five guarantees
- Secure data handling (zero retention policy)
- Complete request tracing and monitoring
- Production-ready deployment

---

## Architecture & Workflow

### System Flow


### 1. Safe Inputs
- File type validation (audio formats only)
- File size limits (max 100MB)
- Content validation (non-empty files)
- API key validation on startup

### 2. Predictable Outputs
- Standardized response schemas with `SuccessResponse` and `ErrorResponse`
- Request IDs for traceability
- Timestamps on all responses
- Consistent error handling

### 3. Failure Containment
- Retry logic with exponential backoff (3 attempts)
- Circuit breaker pattern for external APIs
- Timeout protection (30s per API call for `audio-to-document`, 60s for `audio-url-to-document`)
- Graceful error handling

### 4. Privacy by Design
- Secure file deletion (DoD 5220.22-M 3-pass overwrite)
- Zero data retention (no persistence to disk)
- API key masking in logs
- Temporary file cleanup guaranteed

### 5. Operational Discipline
- Structured logging with context
- Request tracking and metrics
- Health check endpoint with metrics
- Operation timing and monitoring

## API Endpoints

### `POST /audio-to-document`
Upload audio file for pedagogical analysis.

**Request:**
- Content-Type: `multipart/form-data`
- Field: `audio` (MP3, WAV, M4A, MP4) - **Required**
- Field: `syllabus` (text) - **Optional** - Curriculum/syllabus context for alignment scoring
- Max size: 100MB

**Response:**
```json
{
  "status": "success",
  "request_id": "uuid",
  "timestamp": "2026-02-13T...",
  "data": {
    "analysis": "The instructor maintained good engagement throughout the session, asking approximately 8 questions over the 30-minute period. After posing questions, there were noticeable pauses averaging 4 seconds, which gave students time to think. The class dynamic showed a healthy balance with about 65% instructor speaking time and 35% student participation. Several instances of Hindi-English code-switching were observed, creating a culturally comfortable learning environment.",
    "pedagogical_score": 78,
    "score_reasoning": "Strong question frequency and optimal wait time. Good student participation at 35%. Code-switching shows cultural engagement. Covered 4 out of 5 syllabus topics with appropriate depth.",
    "processing_time_seconds": 12.34
  }
}
```

### `POST /audio-url-to-document`
Process audio from a publicly accessible URL for pedagogical analysis. Supports Google Drive direct download links.

**Request:**
- Content-Type: `application/json`
- Body:
```json
{
  "audio_url": "https://example.com/audio.mp3",
  "syllabus": "Topics: Introduction to Machine Learning, Supervised Learning, Neural Networks, Deep Learning Fundamentals, Model Evaluation"
}
```
- `audio_url` (string, `HttpUrl`) - **Required** - URL of the audio file (e.g., AWS S3, Google Drive direct link). Max file size for download: 100MB.
- `syllabus` (text) - **Optional** - Curriculum/syllabus context for alignment scoring

**Response:**
(Same as `/audio-to-document` endpoint response)

### `GET /health`
Health check and metrics.

**Response:**
```json
{
  "status": "healthy",
  "service": "lecture_analysis_api",
  "metrics": {
    "total_requests": 42,
    "total_errors": 2,
    "error_rate_percent": 4.76,
    "average_processing_time_seconds": 11.5
  }
}
```

### `GET /`
API information and available endpoints.

## Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Configure environment variables:**
Create `.env` file:
```env
OPENAI_API_KEY=sk-...
NVIDIA_API_KEY=nvapi-...
```

3. **Run the server:**
```bash
uvicorn main:app --reload
```

4. **Test with curl:**
```bash
# Basic request with file upload
curl -X POST "http://localhost:8000/audio-to-document" \
  -F "audio=@lecture.mp3"

# With syllabus for context-aware scoring via file upload
curl -X POST "http://localhost:8000/audio-to-document" \
  -F "audio=@lecture.mp3" \
  -F "syllabus=Topics: Introduction to Machine Learning, Supervised Learning, Neural Networks, Deep Learning Fundamentals, Model Evaluation"

# Basic request with audio URL
curl -X POST "http://localhost:8000/audio-url-to-document" \
  -H "Content-Type: application/json" \
  -d '{
        "audio_url": "https://drive.google.com/file/d/YOUR_GOOGLE_DRIVE_FILE_ID/view?usp=sharing",
        "syllabus": "Topics: Introduction to Machine Learning, Supervised Learning, Neural Networks, Deep Learning Fundamentals, Model Evaluation"
      }'
```

## Architecture

```
lecture_analysis_api/
├── main.py                 # FastAPI app with health checks
├── config.py              # Configuration with API key validation
├── api/
│   └── routes.py          # API endpoints with five guarantees
├── services/
│   ├── transcription.py   # OpenAI Whisper integration
│   └── nvidia_writer.py   # NVIDIA LLM analysis
├── middleware/
│   ├── validators.py      # Input validation (Safe Inputs)
│   ├── failure_containment.py  # Retry, circuit breakers
│   ├── privacy.py         # Secure deletion, data retention
│   └── observability.py   # Logging and metrics
└── models/
    ├── requests.py        # Request schemas (for URL-based input)
    └── responses.py       # Response schemas (Predictable Outputs)
```

## Pedagogical Analysis

The analysis provides a **human-readable, conversational observation report** covering:

1. **Question Frequency & Quality**: How often questions are asked and their distribution
2. **Wait Time & Pacing**: Pause duration after questions and overall session pacing
3. **Student Engagement**: Balance between instructor and student participation
4. **Cultural Contextualization**: Hindi/English/Hinglish code-switching for cultural adaptation

The analysis is written in natural, flowing paragraphs that educators can easily understand and act upon - not robotic bullet points.

### Pedagogical Quality Score (0-100)

Context-aware scoring based on:

**Teaching Methodology (80 points):**
- **Question frequency (0-35 pts)**: Optimal is 2-4 questions per 10 minutes
- **Wait time (0-20 pts)**: >3 seconds silence after questions is optimal  
- **Student participation (0-20 pts)**: 30-50% student voice is optimal
- **Cultural engagement (0-5 pts)**: Code-switching shows cultural adaptation

**Curriculum Alignment (0-20 pts)** - *Only when syllabus is provided:*
- Content coverage against syllabus topics
- Depth and accuracy of topic treatment
- Alignment with learning objectives

**Baseline:** Content delivery alone earns 30-40 points. Monologue lectures typically score 30-50.

The score includes detailed, actionable reasoning (2-3 sentences) explaining the assessment.

## Security Features

- **No data persistence**: Audio deleted after processing
- **Secure deletion**: 3-pass overwrite (DoD standard)
- **API key protection**: Validated on startup, masked in logs
- **Timeout protection**: All external calls have 30s timeout (60s for URL downloads)
- **Circuit breakers**: Prevent cascading failures

## Monitoring

All requests tracked with:
- Request ID (UUID)
- Timestamp
- Processing time
- Success/failure status
- Structured logs

Access metrics via `/health` endpoint.

## Error Handling

HTTP codes:
- `200`: Success
- `400`: Bad request (invalid file, download error, URL error)
- `413`: File too large
- `502`: External API failure
- `500`: Internal server error

All errors return standardized `ErrorResponse` with request ID and details.
