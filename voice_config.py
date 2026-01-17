"""
Voice Agent Configuration
Environment variables and API keys needed for the voice agent system
"""

import os
from typing import Dict, Any

def get_required_env_vars() -> Dict[str, str]:
    """Get all required environment variables with descriptions"""
    return {
        # Groq API Key (for LLM and STT/TTS)
        "GROQ_API_KEY": {
            "description": "Groq API key for Llama3 model and speech services",
            "required": True,
            "example": "gsk_...",
            "how_to_get": "Get from https://console.groq.com/"
        },
        
        # LiveKit API Key (for voice streaming)
        "LIVEKIT_API_KEY": {
            "description": "LiveKit API key for real-time voice streaming",
            "required": True,
            "example": "livekit_...",
            "how_to_get": "Get from https://dashboard.livekit.io/"
        },
        
        # LiveKit API Secret
        "LIVEKIT_API_SECRET": {
            "description": "LiveKit API secret for voice streaming",
            "required": True,
            "example": "livekit_secret_...",
            "how_to_get": "Get from https://dashboard.livekit.io/"
        },
        
        # Optional: Database connection (for email integration)
        "DATABASE_URL": {
            "description": "Database URL for email storage and retrieval",
            "required": False,
            "example": "postgresql://user:password@localhost/emaildb"
        },
        
        # Optional: Redis connection (for caching)
        "REDIS_URL": {
            "description": "Redis URL for session caching",
            "required": False,
            "example": "redis://localhost:6379"
        }
    }

def check_env_vars() -> Dict[str, Any]:
    """Check which environment variables are set"""
    required_vars = get_required_env_vars()
    status = {}
    
    for var_name, var_info in required_vars.items():
        value = os.getenv(var_name)
        status[var_name] = {
            "set": bool(value),
            "value": value if value else None,
            "description": var_info["description"],
            "required": var_info["required"],
            "example": var_info["example"]
        }
    
    return status

def print_setup_instructions():
    """Print setup instructions for the user"""
    status = check_env_vars()
    
    print("🔧 Voice Agent Environment Setup")
    print("=" * 50)
    
    missing_required = []
    for var_name, var_status in status.items():
        if var_status["required"] and not var_status["set"]:
            missing_required.append(var_name)
    
    if missing_required:
        print("❌ Missing Required Environment Variables:")
        for var in missing_required:
            print(f"   • {var}")
        print(f"\n📝 To set these variables, run:")
        print(f"   set {missing_required[0]}=your_api_key_here")
        print(f"   set {missing_required[1]}=your_secret_here")
        if len(missing_required) > 2:
            print(f"   set {missing_required[2]}=your_value_here")
        print(f"\n💡 Or create a .env file with:")
        for var in missing_required:
            print(f"   {var}=your_value")
        print(f"\n🚀 Then restart your application")
    else:
        print("✅ All required environment variables are set!")
        print("\n🎤 Voice Agent is ready to use!")
    
    print("\n" + "=" * 50)
    print("📋 Optional Environment Variables:")
    for var_name, var_status in status.items():
        if not var_status["required"]:
            status_icon = "✅" if var_status["set"] else "❌"
            print(f"   {status_icon} {var_name}: {'Set' if var_status['set'] else 'Not Set'}")
    
    print("\n🔗 Quick Setup Commands:")
    print("   For Windows (PowerShell):")
    print("   $env:GROQ_API_KEY=your_key_here")
    print("   $env:LIVEKIT_API_KEY=your_key_here")
    print("   $env:LIVEKIT_API_SECRET=your_secret_here")
    print("\n   For Linux/Mac:")
    print("   export GROQ_API_KEY=your_key_here")
    print("   export LIVEKIT_API_KEY=your_key_here")
    print("   export LIVEKIT_API_SECRET=your_secret_here")

if __name__ == "__main__":
    print_setup_instructions()
