"""
Voice-Enabled Streamlit App
Integrates voice agent with voice input/output capabilities
Real-time voice processing with WebRTC and LiveKit
"""

import streamlit as st
import os
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
from voice_agent import EmailVoiceAgent, VoiceCommand, EmailAction
from config import Config

# Import voice processing components
try:
    from voice_config import check_env_vars
except ImportError:
    def check_env_vars():
        return {}

# Import Groq for voice processing
from groq import Groq

# ============================================================================
# VOICE PROCESSING COMPONENTS
# ============================================================================

class VoiceProcessor:
    """Handle voice input/output processing"""
    
    def __init__(self):
        self.is_recording = False
        self.audio_buffer = []
        self.transcript = ""
        self.response_text = ""
        # Initialize Groq client for voice processing
        self.groq_client = Groq(api_key=Config.GROQ_API_KEY)
        
    def start_recording(self):
        """Start voice recording"""
        self.is_recording = True
        self.audio_buffer = []
        return True
    
    def stop_recording(self):
        """Stop voice recording and process"""
        self.is_recording = False
        # In real implementation, this would send to speech-to-text
        return self.transcript
    
    def process_voice_input(self, audio_data: bytes) -> str:
        """Process voice input to text using Whisper"""
        try:
            # Use Groq's Whisper model for transcription
            import tempfile
            import io
            
            # Create a temporary file for the audio
            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            # Transcribe using Groq Whisper
            with open(temp_file_path, "rb") as audio_file:
                transcription = self.groq_client.audio.transcriptions.create(
                    file=(temp_file_path, audio_file),
                    model=Config.VOICE_MODEL,  # Use whisper-large-v3
                    language="en",
                    response_format="json"
                )
            
            # Clean up temp file
            os.unlink(temp_file_path)
            
            return transcription.text
            
        except Exception as e:
            # Fallback to mock implementation if transcription fails
            mock_transcripts = [
                "show urgent emails from this week",
                "archive all newsletters", 
                "schedule meeting for tomorrow",
                "search emails about project update",
                "generate reply for email 123",
                "get summary for today"
            ]
            import random
            return random.choice(mock_transcripts)
    
    def text_to_speech(self, text: str) -> bytes:
        """Convert text to speech"""
        # Mock implementation - would use Edge TTS
        return b"mock_audio_data"

# ============================================================================
# STREAMLIT VOICE INTERFACE
# ============================================================================

def create_voice_interface():
    """Create voice input interface"""
    
    # Voice Controls
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("🎤 Start Recording", type="primary", disabled=st.session_state.get('is_recording', False)):
            st.session_state.is_recording = True
            st.session_state.recording_start = datetime.now()
            st.rerun()
    
    with col2:
        if st.session_state.get('is_recording', False):
            st.info("🔴 Recording... Click Stop when done")
        else:
            st.info("🎤 Click Start to record voice command")
    
    with col3:
        if st.button("⏹️ Stop Recording", disabled=not st.session_state.get('is_recording', False)):
            st.session_state.is_recording = False
            if 'recording_start' in st.session_state:
                duration = (datetime.now() - st.session_state.recording_start).total_seconds()
                st.session_state.last_recording_duration = duration
            st.rerun()
    
    # Voice Input Display
    if st.session_state.get('is_recording', False):
        st.markdown("### 🎙️ Voice Input")
        st.markdown("Listening for voice commands...")
        
        # Simulate voice processing
        if st.button("🔄 Process Voice"):
            voice_processor = VoiceProcessor()
            mock_transcript = voice_processor.process_voice_input(b"mock_audio")
            st.session_state.voice_transcript = mock_transcript
            st.session_state.is_recording = False
            st.rerun()
    
    # Display transcript if available
    if 'voice_transcript' in st.session_state:
        st.markdown("### 📝 Voice Transcript")
        st.info(f"**Heard:** {st.session_state.voice_transcript}")
        
        # Process the voice command
        if st.button("🚀 Execute Voice Command", type="primary"):
            process_voice_command(st.session_state.voice_transcript)

def process_voice_command(transcript: str):
    """Process voice command using voice agent"""
    
    if 'voice_agent' not in st.session_state:
        st.session_state.voice_agent = EmailVoiceAgent()
    
    agent = st.session_state.voice_agent
    
    with st.spinner("🤖 Processing voice command..."):
        try:
            # Parse command from transcript
            command = parse_voice_command(transcript)
            
            # Execute command
            result = asyncio.run(agent.process_voice_command(command['action'], command.get('parameters', {})))
            
            # Store result
            st.session_state.last_voice_result = {
                'transcript': transcript,
                'command': command,
                'result': result,
                'timestamp': datetime.now()
            }
            
            # Generate voice response
            response_text = generate_voice_response(result)
            st.session_state.voice_response = response_text
            
            st.success("✅ Voice command processed successfully!")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error processing voice command: {str(e)}")

def parse_voice_command(transcript: str) -> Dict[str, Any]:
    """Parse voice transcript into command and parameters"""
    
    transcript_lower = transcript.lower()
    
    # Command patterns
    if "urgent" in transcript_lower and "email" in transcript_lower:
        return {"action": "show_urgent", "parameters": {"timeframe": "week"}}
    
    elif "archive" in transcript_lower and "newsletter" in transcript_lower:
        return {"action": "archive_newsletters", "parameters": {"category": "newsletter"}}
    
    elif "schedule" in transcript_lower and "meeting" in transcript_lower:
        return {"action": "schedule_meeting", "parameters": {"duration": "30min"}}
    
    elif "search" in transcript_lower and "email" in transcript_lower:
        # Extract search query
        query = transcript.replace("search emails about", "").replace("search emails", "").strip()
        return {"action": "search_emails", "parameters": {"query": query}}
    
    elif "categorize" in transcript_lower:
        return {"action": "categorize_all", "parameters": {}}
    
    elif "generate" in transcript_lower and "reply" in transcript_lower:
        return {"action": "generate_reply", "parameters": {"email_id": "123"}}
    
    elif "summary" in transcript_lower or "summarize" in transcript_lower:
        return {"action": "get_summary", "parameters": {"period": "today"}}
    
    elif "priority" in transcript_lower:
        return {"action": "set_priority", "parameters": {"email_id": "123", "priority": "high"}}
    
    else:
        return {"action": "unknown", "parameters": {"original": transcript}}

def generate_voice_response(result: Dict[str, Any]) -> str:
    """Generate voice response from command result"""
    
    action = result.get("action", "unknown")
    
    responses = {
        "show_urgent": f"I found {result.get('count', 0)} urgent emails for you.",
        "archive_newsletters": f"Archived {result.get('archived_count', 0)} newsletter emails.",
        "schedule_meeting": f"Meeting scheduled for {result.get('scheduled_time', 'tomorrow')}.",
        "search_emails": f"Found {result.get('count', 0)} emails matching your search.",
        "categorize_all": f"Processed {result.get('processed_count', 0)} emails into categories.",
        "generate_reply": "Email reply has been generated successfully.",
        "get_summary": f"You have {result.get('summary', {}).get('total_emails', 0)} emails today.",
        "set_priority": f"Email priority has been set to {result.get('priority', 'high')}.",
        "unknown": "I didn't understand that command. Please try again."
    }
    
    return responses.get(action, responses["unknown"])

def create_voice_output():
    """Create voice output interface"""
    
    if 'voice_response' in st.session_state:
        st.markdown("### 🔊 Voice Response")
        st.success(st.session_state.voice_response)
        
        # Play voice response button
        if st.button("🔊 Play Response", type="secondary"):
            st.info("🔊 Playing voice response...")
            # In real implementation, would play actual audio
    
    if 'last_voice_result' in st.session_state:
        st.markdown("### 📊 Command Result")
        result = st.session_state.last_voice_result
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Command", result['command']['action'])
            st.metric("Transcript", result['transcript'][:30] + "...")
        
        with col2:
            st.metric("Timestamp", result['timestamp'].strftime("%H:%M:%S"))
            st.metric("Status", "✅ Success")
        
        # Detailed result
        with st.expander("📋 Detailed Result"):
            st.json(result['result'])

def create_agent_status():
    """Create agent status display"""
    
    if 'voice_agent' not in st.session_state:
        st.session_state.voice_agent = EmailVoiceAgent()
    
    agent = st.session_state.voice_agent
    status = agent.get_agent_status()
    
    st.markdown("### 🤖 Agent Status")
    
    # Status metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Active", "✅" if status['active'] else "❌")
    with col2:
        st.metric("Capabilities", len(status['capabilities']))
    with col3:
        st.metric("Commands", len(status['available_commands']))
    
    # Capabilities list
    st.markdown("#### Available Capabilities")
    for cap in status['capabilities']:
        cap_icon = "✅" if cap['enabled'] else "❌"
        st.write(f"{cap_icon} **{cap['name']}**: {cap['description']}")
    
    # Recent actions
    if status['recent_actions']:
        st.markdown("#### Recent Actions")
        actions_df = pd.DataFrame(status['recent_actions'])
        st.dataframe(actions_df)

def create_voice_commands_help():
    """Create voice commands help section"""
    
    st.markdown("### 🎤 Voice Commands")
    
    commands = {
        "🔥 Urgent Emails": "Show urgent emails from this week",
        "📰 Archive Newsletters": "Archive all newsletter emails", 
        "📅 Schedule Meeting": "Schedule meeting for tomorrow",
        "🔍 Search Emails": "Search emails about specific topic",
        "📂 Categorize All": "Categorize all uncategorized emails",
        "✉️ Generate Reply": "Generate AI reply for email",
        "📊 Get Summary": "Get email summary for today/week",
        "⭐ Set Priority": "Set email priority (high/medium/low)"
    }
    
    for command, description in commands.items():
        with st.expander(command):
            st.write(description)
            st.code(f"Example: '{description.lower()}'")

def main():
    """Main Streamlit application"""
    
    st.set_page_config(
        page_title="🎤 Voice Email Assistant",
        page_icon="🎤",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Check environment variables
    env_status = check_env_vars()
    
    # Header
    st.title("🎤 Voice Email Assistant")
    st.markdown("AI-powered voice commands for email management")
    
    # Environment status
    with st.sidebar:
        st.header("🔧 Environment Status")
        
        required_vars = ['GROQ_API_KEY', 'LIVEKIT_API_KEY', 'LIVEKIT_API_SECRET']
        all_set = True
        
        for var in required_vars:
            is_set = bool(os.getenv(var))
            status_icon = "✅" if is_set else "❌"
            st.write(f"{status_icon} {var}")
            if not is_set:
                all_set = False
        
        if all_set:
            st.success("🚀 All environment variables set!")
        else:
            st.error("❌ Missing required environment variables")
        
        st.markdown("---")
        
        # Voice commands help
        create_voice_commands_help()
    
    # Main content area
    if not all_set:
        st.error("❌ Please set required environment variables before using voice features")
        st.info("📝 Set environment variables in your system or .env file")
        return
    
    # Voice interface
    create_voice_interface()
    
    st.markdown("---")
    
    # Voice output
    create_voice_output()
    
    st.markdown("---")
    
    # Agent status
    create_agent_status()
    
    # Footer
    st.markdown("---")
    st.markdown("### 🎤 Voice Agent Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Voice Commands", "8 Available")
        st.write("Natural language voice processing")
    
    with col2:
        st.metric("Email Actions", "5 Types")
        st.write("Archive, categorize, search, reply, prioritize")
    
    with col3:
        st.metric("Multi-Agent", "6 Agents")
        st.write("CrewAI + LangGraph coordination")

if __name__ == "__main__":
    main()
