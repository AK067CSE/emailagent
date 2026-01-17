import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    LIVEKIT_URL = os.getenv("LIVEKIT_URL")
    LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
    
    # Validate that required keys exist
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found in environment variables")
    if not LIVEKIT_URL:
        raise ValueError("LIVEKIT_URL not found in environment variables")
    if not LIVEKIT_API_KEY:
        raise ValueError("LIVEKIT_API_KEY not found in environment variables")
    
    # Model Configuration - Specific models for different tasks
    EMAIL_REPLY_MODEL = "qwen/qwen-32b"  # For email reply system
    EMAIL_ORGANIZER_MODEL = "llama-3.1-8b-instant"  # For email organizer
    VOICE_MODEL = "whisper-large-v3"  # For voice/audio processing
    DEFAULT_MODEL = "llama-3.1-8b-instant"
    
    # Model Settings
    TEMPERATURE = 0.1
    MAX_TOKENS = 2048
    
    # Data Processing
    MAX_RETRIES = 3
    INITIAL_RETRY_DELAY = 1
    
    # Categories for email classification
    EMAIL_CATEGORIES = [
        "Work/Professional",
        "Personal", 
        "Marketing/Newsletter",
        "Notifications/System",
        "Billing/Financial",
        "Support/Customer Service",
        "HR/Administrative",
        "Security/Alert",
        "Social/Community",
        "Other"
    ]
    
    # Priority levels
    PRIORITY_LEVELS = ["High", "Medium", "Low"]
    
    # Action types
    ACTION_TYPES = [
        "Reply Immediately",
        "Schedule Meeting", 
        "Archive",
        "Delete",
        "Flag for Follow-up",
        "Forward",
        "Review Later",
        "No Action Needed"
    ]
