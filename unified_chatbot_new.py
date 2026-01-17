"""
Unified Email Chatbot - Single Streamlit Interface
All agents (RAG, CrewAI, Organizer, Voice) accessible via one chat interface
Fully automated - no tab switching required.
"""

import streamlit as st
import time
from datetime import datetime
from orchestrator import EmailOrchestrator
import asyncio

# Try to import voice agent, fallback if not available
try:
    from voice_agent import EmailVoiceAgent
    VOICE_AGENT_AVAILABLE = True
except ImportError:
    VOICE_AGENT_AVAILABLE = False
    EmailVoiceAgent = None

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
    if 'voice_mode' not in st.session_state:
        st.session_state.voice_mode = "Text only (no mic)"

init_session_state()

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
        options=["Text only (no mic)", "Use microphone (WebRTC)"],
        index=0,
        key="voice_mode"
    )
    
    if voice_mode == "Use microphone (WebRTC)":
        st.markdown("""
        <style>
        .voice-controls { border: 2px solid #ddd; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; }
        .voice-button { margin: 0.25rem; }
        </style>
        """, unsafe_allow_html=True)
        
        col_record, col_stop = st.columns(2)
        with col_record:
            if st.button("🎤 Start Recording", key="start_webrtc", type="primary"):
                st.session_state.is_recording = True
                st.session_state.voice_input = ""
                st.rerun()
        with col_stop:
            if st.button("⏹️ Stop & Process", key="stop_webrtc", type="secondary", disabled=not st.session_state.get('is_recording', False)):
                st.session_state.is_recording = False
                if st.session_state.get('voice_audio_bytes'):
                    with st.spinner("🎤 Transcribing..."):
                        try:
                            transcription = st.session_state.voice_agent.process_voice_input(st.session_state.voice_audio_bytes)
                            st.session_state.voice_input = transcription
                        except Exception as e:
                            st.session_state.voice_input = f"Transcription error: {e}"
                    st.session_state.voice_audio_bytes = None
                    st.rerun()
        
        # WebRTC audio capture
        webrtc_code = f"""
        <script>
        let mediaRecorder;
        let audioChunks = [];
        
        async function startRecording() {{
            console.log('Starting WebRTC recording...');
            const stream = await navigator.mediaDevices.getUserMedia({{
                audio: {{
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                }}
            }});
            
            mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.ondataavailable = (event) => {{
                if (event.data.size > 0) {{
                    audioChunks.push(event.data);
                }}
            }};
            
            mediaRecorder.onstop = () => {{
                console.log('Recording stopped, creating blob...');
                const audioBlob = new Blob(audioChunks, {{ type: 'audio/wav' }});
                const audioUrl = URL.createObjectURL(audioBlob);
                
                // Send to Python backend
                window.parent.postMessage({{
                    type: 'voice_audio',
                    audio: audioUrl
                }});
            }};
            
            mediaRecorder.start();
            return mediaRecorder;
        }}
        
        function stopRecording() {{
            console.log('Stopping recording...');
            if (mediaRecorder && mediaRecorder.state !== 'inactive') {{
                mediaRecorder.stop();
            }}
        }}
        
        window.startRecording = startRecording;
        window.stopRecording = stopRecording;
        </script>
        """
        
        st.components.v1.html(webrtc_code)
        
        # Hidden input to receive audio data
        st.session_state.voice_audio_bytes = st.text_input("Audio data will appear here", key="voice_audio_bytes", label="Voice audio capture")
        
        if st.session_state.is_recording:
            st.warning("🔴 Recording... Click 'Stop & Process' when done.")
        elif st.session_state.get('voice_audio_bytes'):
            st.success("✅ Audio captured. Click 'Stop & Process' to transcribe.")
    else:
        st.info("🎤 Click 'Start Recording' to begin. Microphone access will be requested.")
    
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
                        
                        st.session_state.voice_input = ""
                        st.rerun()
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
        ("💬 Chat", "chat What are the main customer concerns?"),
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
    
    st.markdown("---")
    
    # Footer
    st.markdown("""
    ### 🎯 **How It Works:**
    - **Single Interface**: All agents accessible through one chat window
    - **Smart Routing**: Commands automatically routed to correct agent system
    - **Voice Ready**: Microphone input with intent detection (when available)
    - **Full Features**: Search, chat, draft, categorize, organize, filter, and reply generation
    
    ### 🤖 **Agent Capabilities:**
    - **RAG System**: Chat with your email database, search emails, draft with context
    - **Email Organizer**: Categorize, prioritize, and suggest actions for emails
    - **CrewAI Reply**: Multi-agent email generation (categorize → research → write)
    - **Voice Agent**: Process voice commands and integrate with all systems
    
    ### 💡 **Pro Tips:**
    - Commands are case-insensitive
    - Natural language works: "show me urgent emails" or "list all emails from Alice"
    - Voice commands: "show urgent emails", "archive newsletters", "search pricing"
    - Type `help` anytime to see all commands
    
    **Just type your command and all agents work together automatically!**
    """)
    
    # Clear chat
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
