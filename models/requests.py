from pydantic import BaseModel, HttpUrl

class AudioUrlRequest(BaseModel):
    audio_url: HttpUrl
    syllabus: str = ""
