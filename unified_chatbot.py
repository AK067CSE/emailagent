"""
Unified Email Chatbot - Single Streamlit Interface
All agents (RAG, CrewAI, Organizer, Voice) accessible via one chat interface
Fully automated - no tab switching required.
"""

import streamlit as st
import time
from datetime import datetime
from orchestrator import EmailOrchestrator
from email_rag import EmailRAGSystem
from email_reply_agents import EmailReplyAgents, EmailReplyTasks, ReplyWorkflow
from agents import EmailOrchestrator as EmailOrganizerSystem
from config import Config
import asyncio
import io
import wave
import tempfile
import os
import threading
from collections import deque

# Try to import voice agent, fallback if not available
try:
    from voice_agent import EmailVoiceAgent
    VOICE_AGENT_AVAILABLE = True
except ImportError:
    VOICE_AGENT_AVAILABLE = False
    EmailVoiceAgent = None

# Try to import streamlit-webrtc for real microphone capture
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode
    import av
    import numpy as np
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

_AUDIO_BUFFERS_LOCK = threading.Lock()
_AUDIO_BUFFERS: dict[str, deque] = {}


def _get_audio_buffer(key: str) -> deque:
    with _AUDIO_BUFFERS_LOCK:
        if key not in _AUDIO_BUFFERS:
            _AUDIO_BUFFERS[key] = deque(maxlen=2000)
        return _AUDIO_BUFFERS[key]

# Try to import LiveKit agents (optional)
LIVEKIT_IMPORT_ERROR = None
try:
    from livekit import agents as livekit_agents
    LIVEKIT_AVAILABLE = True
except Exception as e:
    LIVEKIT_AVAILABLE = False
    LIVEKIT_IMPORT_ERROR = str(e)

# Page configuration
st.set_page_config(
    page_title="🤖 Unified Email Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful colorful chatbot interface
st.markdown("""
<style>
.chat-container {
    max-width: 800px;
    margin: 0 auto;
    padding: 1rem;
}

.chat-message {
    padding: 1rem;
    border-radius: 1rem;
    margin-bottom: 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    animation: fadeIn 0.3s ease-in;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.user-message {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    align-items: flex-end;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
}

.assistant-message {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    color: white;
    align-items: flex-start;
    box-shadow: 0 4px 15px rgba(240, 147, 251, 0.3);
}

.message-content {
    background: rgba(255, 255, 255, 0.95);
    color: #1f2937;
    padding: 1.2rem;
    border-radius: 0.75rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    max-width: 100%;
    word-wrap: break-word;
    font-weight: 600;
    backdrop-filter: blur(10px);
}

.command-indicator {
    background: linear-gradient(135deg, #ffd89b 0%, #19547b 100%);
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 2rem;
    font-size: 0.8rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    display: inline-block;
    box-shadow: 0 2px 8px rgba(25, 84, 123, 0.3);
}

.status-indicator {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1rem 1.5rem;
    border-radius: 1rem;
    margin: 1rem 0;
    text-align: center;
    font-weight: 700;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
}

.quick-actions {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 0.75rem;
    margin: 1rem 0;
}

.action-button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1rem;
    border-radius: 0.75rem;
    border: none;
    font-weight: 700;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
}

.action-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
}

.stSpinner > div {
    border-top-color: #667eea !important;
}

.chat-input-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 0.5rem;
    border-radius: 1rem;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
}

/* Bold all text content */
.message-content strong, .message-content b {
    color: #667eea;
    font-weight: 800;
}

/* Streamlit input styling */
.stTextInput > div > div > input {
    background: rgba(255, 255, 255, 0.9);
    border-radius: 0.5rem;
    border: 2px solid #667eea;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    if 'orchestrator' not in st.session_state:
        with st.spinner("🚀 Initializing All Agent Systems..."):
            st.session_state.orchestrator = EmailOrchestrator()
    if 'voice_agent' not in st.session_state and VOICE_AGENT_AVAILABLE:
        st.session_state.voice_agent = EmailVoiceAgent()
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'system_status' not in st.session_state:
        st.session_state.system_status = "ready"
    if 'voice_input' not in st.session_state:
        st.session_state.voice_input = ""
    if 'is_recording' not in st.session_state:
        st.session_state.is_recording = False

init_session_state()

# One-time capabilities suggestion
if 'capabilities_shown' not in st.session_state:
    capabilities_msg = (
        "I can help you manage your inbox using multiple agents:\n\n"
        "RAG (Search/Chat/Draft/List)\n"
        "- search pricing plans\n"
        "- chat What are the main customer concerns?\n"
        "- draft email to client@example.com about follow-up meeting\n"
        "- list all\n"
        "- list from John\n"
        "- list thread thread_001\n"
        "- emails invoice\n\n"
        "Email Organizer (Categorize/Priority/Actions)\n"
        "- organize all\n"
        "- categorize all emails\n"
        "- filter high priority\n\n"
        "CrewAI Reply (Categorize + Research + Write)\n"
        "- reply to \"I need pricing details\"\n\n"
        "Voice (if available)\n"
        "- Use the sidebar voice controls and speak a command like: \"show urgent emails\"\n\n"
        "Type `help` anytime for the full list."
    )
    st.session_state.chat_history.append({
        "timestamp": datetime.now(),
        "role": "assistant",
        "content": capabilities_msg
    })
    st.session_state.capabilities_shown = True

# Header
st.title("🤖 Unified Email Assistant")
st.markdown("**All agents working together automatically** - RAG + CrewAI + Organizer + Voice")

# System Status
status_col1, status_col2, status_col3, status_col4 = st.columns(4)

with status_col1:
    if st.session_state.orchestrator.rag_system:
        st.success("✅ RAG System")
    else:
        st.error("❌ RAG System")

with status_col2:
    if st.session_state.orchestrator.reply_agents:
        st.success("✅ CrewAI Reply")
    else:
        st.error("❌ CrewAI Reply")

with status_col3:
    if st.session_state.orchestrator.email_organizer:
        st.success("✅ Email Organizer")
    else:
        st.error("❌ Email Organizer")

with status_col4:
    if VOICE_AGENT_AVAILABLE and hasattr(st.session_state, 'voice_agent') and st.session_state.voice_agent:
        st.success("✅ Voice Agent")
    else:
        st.error("❌ Voice Agent (Import Failed)")

# Quick Actions
st.markdown("### 🚀 Quick Actions")
with st.container():
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔍 Search Emails", use_container_width=True, key="quick_search"):
            st.session_state.quick_command = "search pricing plans"
    
    with col2:
        if st.button("💬 Chat with Emails", use_container_width=True, key="quick_chat"):
            st.session_state.quick_command = "chat What are the main customer concerns?"
    
    with col3:
        if st.button("📝 Draft Email", use_container_width=True, key="quick_draft"):
            st.session_state.quick_command = "draft email to client@example.com about follow-up meeting"
    
    with col4:
        if st.button("✉️ Generate Reply", use_container_width=True, key="quick_reply"):
            st.session_state.quick_command = 'reply to "I am interested in your enterprise pricing"'

# Chat Interface
st.markdown("### 💬 Chat Interface")
st.markdown("**Type any command below** - All agents will work automatically!")

# Display chat history
chat_container = st.container()
with chat_container:
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <div class="message-content">
                    <strong>👤 You:</strong> <b>{message["content"]}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <div class="message-content">
                    <strong>🤖 Assistant:</strong><br>
                    <b>{message["content"]}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

# Chat input with colorful container
st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)
col1, col2 = st.columns([4, 1])

with col1:
    user_input = st.chat_input(
        "Type your command or question...",
        key="user_input"
    )

with col2:
    if st.button("📋 **Help**", key="help_button"):
        st.session_state.help_requested = True

st.markdown('</div>', unsafe_allow_html=True)

# Resolve pending input (typed OR quick-action)
pending_input = None
if 'quick_command' in st.session_state and st.session_state.quick_command:
    pending_input = st.session_state.quick_command
    del st.session_state.quick_command
elif user_input:
    pending_input = user_input

# Process help request
if st.session_state.get('help_requested', False):
    with st.spinner("📋 **Loading help information...**"):
        help_response = st.session_state.orchestrator._show_help()
    st.session_state.chat_history.append({
        "timestamp": datetime.now(),
        "role": "assistant",
        "content": help_response
    })
    st.session_state.help_requested = False
    st.rerun()

# Process pending input
if pending_input:
    st.session_state.chat_history.append({
        "timestamp": datetime.now(),
        "role": "user",
        "content": pending_input
    })
 
    with st.spinner("🤖 **Processing your command...**"):
        response = st.session_state.orchestrator.process_input(pending_input)
 
    st.session_state.chat_history.append({
        "timestamp": datetime.now(),
        "role": "assistant",
        "content": response
    })
 
    st.rerun()

# Sidebar
with st.sidebar:
    st.header("🎛️ Control Panel")
    
    # Voice Control Section
    st.markdown("### 🎤 Voice Control")
    
    # Option selector for voice mode
    voice_mode = st.selectbox(
        "Voice Input Mode:",
        options=["Text only (no mic)", "Use microphone (WebRTC)", "LiveKit streaming (beta)"],
        index=0,
        key="voice_mode"
    )
    
    if voice_mode == "Use microphone (WebRTC)":
        if not WEBRTC_AVAILABLE:
            st.error("❌ Microphone capture requires `streamlit-webrtc`. Install it, then restart Streamlit: `pip install streamlit-webrtc`")
        else:
            st.info("🎤 Click **START** on the component below, allow microphone permission, speak 2–5 seconds, then click **Transcribe**.")

            _webrtc_key = "webrtc_audio_stream"

            def _process_audio_frame(frame: "av.AudioFrame") -> "av.AudioFrame":
                # NOTE: This callback runs in a separate thread.
                # Do NOT call Streamlit APIs here.
                try:
                    arr = frame.to_ndarray()
                    # Normalize shapes to (channels, samples)
                    if arr.ndim == 1:
                        arr = arr.reshape(1, -1)
                    if arr.dtype != np.int16:
                        arr = arr.astype(np.int16)
                    buf = _get_audio_buffer(_webrtc_key)
                    with _AUDIO_BUFFERS_LOCK:
                        buf.append({
                            "pcm": arr,
                            "sample_rate": frame.sample_rate or 16000,
                            "channels": len(frame.layout.channels) if frame.layout else arr.shape[0],
                        })
                except Exception:
                    # Best-effort: never break the stream
                    pass
                return frame

            def _frames_to_wav_bytes(frames: list["av.AudioFrame"]) -> bytes:
                if not frames:
                    return b""

                sample_rate = frames[0].sample_rate or 16000
                channels = len(frames[0].layout.channels) if frames[0].layout else 1

                chunks = []
                for frame in frames:
                    arr = frame.to_ndarray()
                    # Expected shape: (channels, samples). Sometimes it arrives as (samples,) for mono.
                    if arr.ndim == 1:
                        arr = arr.reshape(1, -1)
                    if arr.dtype != np.int16:
                        arr = arr.astype(np.int16)
                    chunks.append(arr)

                data = np.concatenate(chunks, axis=1)
                pcm = data.T.tobytes()

                buf = io.BytesIO()
                with wave.open(buf, "wb") as wf:
                    wf.setnchannels(int(channels or 1))
                    wf.setsampwidth(2)
                    wf.setframerate(int(sample_rate or 16000))
                    wf.writeframes(pcm)
                return buf.getvalue()

            # Start WebRTC audio stream (audio_frame_callback buffers frames)
            ctx = webrtc_streamer(
                key=_webrtc_key,
                mode=WebRtcMode.SENDRECV,
                audio_frame_callback=_process_audio_frame,
                media_stream_constraints={"audio": True, "video": False},
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            )

            if ctx and ctx.state.playing:
                st.success("🔴 Microphone is ON (streaming)")
            else:
                st.warning("Microphone is OFF. Click **START** above and allow mic permission.")

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("📝 Transcribe", key="transcribe_voice", type="primary"):
                    if not VOICE_AGENT_AVAILABLE or not hasattr(st.session_state, 'voice_agent'):
                        st.session_state.chat_history.append({
                            "timestamp": datetime.now(),
                            "role": "assistant",
                            "content": "❌ Voice agent not available"
                        })
                        st.rerun()

                    if not ctx or not ctx.state.playing:
                        st.session_state.chat_history.append({
                            "timestamp": datetime.now(),
                            "role": "assistant",
                            "content": "❌ Microphone stream not ready. Click **START** in the WebRTC component and allow mic access, then try again."
                        })
                        st.rerun()

                    with st.spinner("🎙️ Capturing audio..."):
                        with _AUDIO_BUFFERS_LOCK:
                            buf = _get_audio_buffer(_webrtc_key)
                            chunks = list(buf)
                            buf.clear()

                        if chunks:
                            sample_rate = int(chunks[-1].get("sample_rate", 16000) or 16000)
                            channels = int(chunks[-1].get("channels", 1) or 1)
                            data = np.concatenate([c["pcm"] for c in chunks], axis=1)
                            pcm = data.T.tobytes()

                            out = io.BytesIO()
                            with wave.open(out, "wb") as wf:
                                wf.setnchannels(channels)
                                wf.setsampwidth(2)
                                wf.setframerate(sample_rate)
                                wf.writeframes(pcm)
                            wav_bytes = out.getvalue()
                        else:
                            wav_bytes = b""

                    if not wav_bytes:
                        st.session_state.chat_history.append({
                            "timestamp": datetime.now(),
                            "role": "assistant",
                            "content": "❌ No audio captured yet. Click **START**, speak for a few seconds, then click **Transcribe**."
                        })
                        st.rerun()

                    with st.spinner("🎤 Transcribing with Groq Whisper..."):
                        try:
                            from groq import Groq
                            client = Groq(api_key=Config.GROQ_API_KEY)
                            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                                f.write(wav_bytes)
                                temp_path = f.name
                            with open(temp_path, "rb") as audio_file:
                                transcription = client.audio.transcriptions.create(
                                    file=(temp_path, audio_file),
                                    model=Config.VOICE_MODEL,
                                    language="en",
                                    response_format="json"
                                )
                            st.session_state.voice_input = getattr(transcription, "text", "") or ""
                        except Exception as e:
                            st.session_state.voice_input = ""
                            st.session_state.chat_history.append({
                                "timestamp": datetime.now(),
                                "role": "assistant",
                                "content": f"❌ **Transcription Error:** {str(e)}"
                            })
                            st.rerun()

                    # Now process transcript like a normal command (voice agent first, fallback to orchestrator)
                    if st.session_state.voice_input.strip():
                        spoken = st.session_state.voice_input.strip()
                        st.session_state.chat_history.append({
                            "timestamp": datetime.now(),
                            "role": "user",
                            "content": f"🎤 {spoken}"
                        })

                        with st.spinner("🧠 Processing voice command..."):
                            try:
                                voice_result = asyncio.run(
                                    st.session_state.voice_agent.process_voice_command(
                                        spoken,
                                        {"source": "mic"}
                                    )
                                )
                                # Voice agent returns dataclass; fallback to orchestrator for unknown
                                if hasattr(voice_result, "command") and getattr(voice_result, "command") in ["unknown", "error"]:
                                    response = st.session_state.orchestrator.process_input(spoken)
                                    st.session_state.chat_history.append({
                                        "timestamp": datetime.now(),
                                        "role": "assistant",
                                        "content": f"🎤 **Voice (Fallback to Orchestrator):**\n\n{response}"
                                    })
                                else:
                                    # For now, display transcript + command
                                    st.session_state.chat_history.append({
                                        "timestamp": datetime.now(),
                                        "role": "assistant",
                                        "content": f"🎤 **Transcript:** {spoken}\n\n✅ Voice agent processed command: {getattr(voice_result, 'command', 'ok')}"
                                    })
                            except Exception as e:
                                response = st.session_state.orchestrator.process_input(spoken)
                                st.session_state.chat_history.append({
                                    "timestamp": datetime.now(),
                                    "role": "assistant",
                                    "content": f"🎤 **Voice (Fallback to Orchestrator):**\n\n{response}\n\n(Voice agent error: {str(e)})"
                                })

                        st.session_state.voice_input = ""
                        st.rerun()

    elif voice_mode == "LiveKit streaming (beta)":
        st.info("This mode runs a real-time LiveKit voice agent (streaming STT/LLM/TTS) using Groq models.")

        if not LIVEKIT_AVAILABLE:
            st.error("❌ LiveKit is not installed (or failed to import). Install dependencies and restart: `pip install livekit-agents[groq]`")
            if LIVEKIT_IMPORT_ERROR:
                st.code(LIVEKIT_IMPORT_ERROR)
        else:
            missing = []
            if not os.getenv("GROQ_API_KEY"):
                missing.append("GROQ_API_KEY")
            if not os.getenv("LIVEKIT_URL"):
                missing.append("LIVEKIT_URL")
            if not os.getenv("LIVEKIT_API_KEY"):
                missing.append("LIVEKIT_API_KEY")
            if not os.getenv("LIVEKIT_API_SECRET"):
                missing.append("LIVEKIT_API_SECRET")

            if missing:
                st.warning("Missing required environment variables for LiveKit streaming:")
                st.code("\n".join(missing))

            st.markdown("**How to run the LiveKit voice agent:**")
            st.code("python livekit_voice_agent.py")

            st.markdown(
                "After starting the worker, connect from a LiveKit room/client (LiveKit Cloud) and the agent will greet you. "
                "This streaming mode is separate from the Streamlit UI mic mode."
            )

            if 'voice_input' in st.session_state:
                st.text_area("Latest transcript", value=st.session_state.voice_input, height=120)

    # Fallback text area (for mic-less mode)
    if voice_mode == "Text only (no mic)":
        st.session_state.voice_input = st.text_area(
            "Voice Input (will be processed with intent detection):",
            value=st.session_state.voice_input,
            key="voice_text_area",
            help="Speak your command - Voice Agent will detect intent and execute appropriate action"
        )
        
        if st.session_state.voice_input.strip():
            if st.button("🎤 Process Text Command", key="process_text_voice", type="primary"):
                with st.spinner("🎤 **Processing voice command with intent detection...**"):
                    try:
                        # Use voice agent to process command with intent
                        voice_result = asyncio.run(
                            st.session_state.voice_agent.process_voice_command(
                                st.session_state.voice_input, 
                                {"source": "voice_chat"}
                            )
                        )
                        
                        # Format voice agent response for chat
                        if voice_result and "action" in voice_result:
                            response = f"🎤 **Voice Command Processed:**\n\n"
                            response += f"**Action:** {voice_result.get('action', 'Unknown')}\n"
                            
                            if "message" in voice_result:
                                response += f"**Result:** {voice_result['message']}\n"
                            
                            if "results" in voice_result:
                                response += f"**Details:** {voice_result['results']}\n"
                            
                            # Add voice agent response to chat
                            st.session_state.chat_history.append({
                                "timestamp": datetime.now(),
                                "role": "assistant",
                                "content": response
                            })
                        else:
                            # Fallback to orchestrator if voice agent doesn't understand
                            orchestrator_response = st.session_state.orchestrator.process_input(st.session_state.voice_input)
                            st.session_state.chat_history.append({
                                "timestamp": datetime.now(),
                                "role": "assistant",
                                "content": f"🎤 **Voice Input (Fallback Processing):**\n\n{orchestrator_response}"
                            })
                        
                    except Exception as e:
                        st.session_state.chat_history.append({
                            "timestamp": datetime.now(),
                            "role": "assistant",
                            "content": f"❌ **Voice Processing Error:** {str(e)}"
                        })
                
                st.session_state.voice_input = ""
                st.rerun()
    
    # Command examples
    st.markdown("### 💡 Command Examples")
    examples = [
        ("🔍 Search", "search pricing plans"),
        ("💬 Chat", "chat What are customer concerns?"),
        ("📝 Draft", "draft email to john@example.com about meeting"),
        ("✉️ Reply", "reply to 'I need pricing info'"),
        ("🏷️ Categorize", "categorize all emails"),
        ("🔍 Filter", "filter high priority"),
        ("📊 Status", "status")
    ]
    
    for label, example in examples:
        if st.button(label, key=f"example_{example}"):
            st.session_state.example_command = example
    
    # Process example commands
    if 'example_command' in st.session_state:
        st.session_state.chat_history.append({
            "timestamp": datetime.now(),
            "role": "user",
            "content": st.session_state.example_command
        })
        
        with st.spinner("🚀 **Processing example command...**"):
            response = st.session_state.orchestrator.process_input(st.session_state.example_command)
        
        st.session_state.chat_history.append({
            "timestamp": datetime.now(),
            "role": "assistant",
            "content": response
        })
        
        del st.session_state.example_command
        st.rerun()
    
    st.markdown("---")
    
    # Session info
    st.markdown("### 📈 Session Info")
    st.metric("Messages", len(st.session_state.chat_history))
    
    if st.session_state.chat_history:
        last_message = st.session_state.chat_history[-1]
        time_ago = (datetime.now() - last_message["timestamp"]).total_seconds() / 60
        st.metric("Last Activity", f"{time_ago:.0f} min ago")
    
    # Clear chat
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
### 🎯 **How It Works:**
- **Single Interface**: All agents accessible through one chat window
- **Smart Routing**: Commands automatically routed to correct agent system
- **No Tab Switching**: Fully automated workflow
- **Voice Ready**: Voice agent can handle all commands via voice (when integrated)

### 🤖 **Agent Capabilities:**
- **RAG System**: Chat, search, and draft with email context
- **CrewAI Reply**: Multi-agent email generation (categorize → research → write)
- **Email Organizer**: Categorize, prioritize, and organize emails
- **Voice Agent**: Voice command processing (when microphone enabled)

**Just type your command and all agents work together automatically!**
""")

# Instructions expander
with st.expander("📖 Detailed Instructions", expanded=False):
    st.markdown("""
### 🔍 **Search Commands:**
- `search [query]` - Search through your email database
- `find [topic] in emails` - Alternative search syntax
- `lookup [information]` - Find specific information

### 💬 **Chat Commands:**
- `chat about [topic]` - Chat with your email database
- `ask [question]` - Ask questions about your emails
- `tell me about [subject]` - Get information from emails

### 📝 **Draft Commands:**
- `draft email to [recipient] about [topic]` - Draft new email
- `write email for [recipient] regarding [subject]` - Alternative draft syntax
- `compose [recipient] [topic]` - Quick draft composition

### ✉️ **Reply Commands:**
- `reply to [email content]` - Generate AI reply using CrewAI
- `respond to [message]` - Alternative reply syntax
- `generate reply for [text]` - Create intelligent response

### 🏷️ **Organization Commands:**
- `categorize [emails/dataset]` - Organize and categorize emails
- `classify [messages]` - Alternative categorization syntax
- `organize [inbox]` - Structure email organization

### 🔍 **Filter Commands:**
- `filter [category/priority]` - Filter emails by type
- `show [type] emails` - Display filtered results
- `list [criteria]` - List emails matching criteria

### 🎛️ **System Commands:**
- `status` - Show all system statuses
- `help` - Display this help information
- `clear` - Clear chat history
- `quit/exit` - End session

### 💡 **Pro Tips:**
- Commands are case-insensitive
- Natural language works: "Can you search for pricing emails?"
- Multiple agents coordinate automatically for complex tasks
- All responses include context from your email database
""")
