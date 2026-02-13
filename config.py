import os
from dotenv import load_dotenv
from middleware.validators import validate_api_keys

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

# Validate API keys on startup (Safe Inputs Guarantee)
validate_api_keys(OPENAI_API_KEY, NVIDIA_API_KEY)
