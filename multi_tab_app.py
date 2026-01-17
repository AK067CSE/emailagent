"""
Multi-Tab Email Management System
Integrates Email Organizer and Email Reply System in separate tabs
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import both app modules
from app import main as email_organizer_main
from email_reply_app import main as email_reply_main
from voice_streamlit_app import create_voice_interface, process_voice_command, create_agent_status

# Page configuration
st.set_page_config(
    page_title="Email Management System",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for tab navigation
st.markdown("""
<style>
.stTabs [data-baseweb="tab-list"] {
    background-color: #f8fafc;
    border-radius: 0.5rem;
    padding: 0.5rem;
    margin-bottom: 1rem;
}

.stTabs [data-baseweb="tab"] {
    background-color: white;
    border-radius: 0.25rem;
    padding: 0.75rem 1.5rem;
    font-weight: 600;
    transition: all 0.2s ease;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
    color: white;
}

.stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
    background-color: #e5e7eb;
}

.system-info {
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
    padding: 1rem;
    border-radius: 0.5rem;
    border: 1px solid #0ea5e9;
    margin-bottom: 1rem;
}

.feature-highlight {
    background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
    padding: 0.75rem;
    border-radius: 0.25rem;
    border-left: 4px solid #f59e0b;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# Main header
st.title("📧 Advanced Email Management System")
st.markdown("Comprehensive email organization and intelligent reply system powered by AI agents")

# System information
with st.container():
    st.markdown('<div class="system-info">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🤖 AI Agents", "6+ Agents")
    with col2:
        st.metric("📊 Features", "12+ Features")
    with col3:
        st.metric("⚡ Processing", "Real-time")
    with col4:
        st.metric("🎯 Accuracy", "85%+")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3 = st.tabs(["📋 Email Organizer", "✉️ Email Reply System", "🎤 Voice Assistant"])

with tab1:
    st.markdown("### 📋 Email Inbox Organizer")
    st.markdown("Automatically categorize, prioritize, and take action on your emails")
    
    # Feature highlights for Email Organizer
    with st.expander("🌟 Key Features", expanded=False):
        st.markdown('<div class="feature-highlight">🏷️ **Multi-Level Categorization** - Sophisticated classification with user-friendly names</div>', unsafe_allow_html=True)
        st.markdown('<div class="feature-highlight">🎯 **Priority Assignment** - Intelligent prioritization based on content and urgency</div>', unsafe_allow_html=True)
        st.markdown('<div class="feature-highlight">🤖 **Spam Detection** - Advanced spam filtering with AI analysis</div>', unsafe_allow_html=True)
        st.markdown('<div class="feature-highlight">💬 **RAG Chat** - Intelligent email chat and drafting assistant</div>', unsafe_allow_html=True)
        st.markdown('<div class="feature-highlight">📊 **Dashboard Analytics** - Comprehensive email insights and metrics</div>', unsafe_allow_html=True)
        st.markdown('<div class="feature-highlight">🔍 **Advanced Search** - Semantic email search and filtering</div>', unsafe_allow_html=True)
    
    # Load the email organizer app
    try:
        email_organizer_main()
    except Exception as e:
        st.error(f"Error loading Email Organizer: {str(e)}")
        st.info("Please ensure all dependencies are installed and API keys are configured.")

with tab2:
    st.markdown("### ✉️ Intelligent Email Reply System")
    st.markdown("AI-powered email categorization, research, and reply generation using CrewAI")
    
    # Feature highlights for Email Reply System
    with st.expander("🌟 Key Features", expanded=False):
        st.markdown('<div class="feature-highlight">🤖 **CrewAI Agents** - Multi-agent system for email processing</div>', unsafe_allow_html=True)
        st.markdown('<div class="feature-highlight">📋 **Smart Categorization** - Automatic email categorization (price, complaint, feedback, etc.)</div>', unsafe_allow_html=True)
        st.markdown('<div class="feature-highlight">🔍 **Web Research** - Automatic research for accurate responses</div>', unsafe_allow_html=True)
        st.markdown('<div class="feature-highlight">✍️ **Intelligent Replies** - Context-aware email generation</div>', unsafe_allow_html=True)
        st.markdown('<div class="feature-highlight">📊 **Performance Metrics** - Real-time agent performance tracking</div>', unsafe_allow_html=True)
        st.markdown('<div class="feature-highlight">📜 **Processing History** - Complete audit trail of email processing</div>', unsafe_allow_html=True)
    
    # Load the email reply app
    try:
        email_reply_main()
    except Exception as e:
        st.error(f"Error loading Email Reply System: {str(e)}")
        st.info("Please ensure CrewAI and other dependencies are installed: `pip install crewai duckduckgo-search`")

with tab3:
    st.markdown("### 🎤 Voice Assistant")
    st.markdown("AI-powered voice commands for hands-free email management")
    
    # Feature highlights for Voice Assistant
    with st.expander("🌟 Key Features", expanded=False):
        st.markdown('<div class="feature-highlight">🎤 **Voice Commands** - Natural language voice processing</div>', unsafe_allow_html=True)
        st.markdown('<div class="feature-highlight">🤖 **Voice Agent** - AI-powered voice command recognition</div>', unsafe_allow_html=True)
        st.markdown('<div class="feature-highlight">📊 **Real-time Processing** - Instant voice command execution</div>', unsafe_allow_html=True)
        st.markdown('<div class="feature-highlight">🔊 **Voice Responses** - Text-to-speech AI responses</div>', unsafe_allow_html=True)
        st.markdown('<div class="feature-highlight">📧 **Email Actions** - Voice-controlled email operations</div>', unsafe_allow_html=True)
        st.markdown('<div class="feature-highlight">⚡ **Multi-Agent Coordination** - CrewAI + LangGraph integration</div>', unsafe_allow_html=True)
    
    # Load the voice assistant
    try:
        # Import voice agent components
        from voice_agent import EmailVoiceAgent
        
        # Initialize voice agent
        if 'voice_agent' not in st.session_state:
            st.session_state.voice_agent = EmailVoiceAgent()
        
        # Create voice interface
        create_voice_interface()
        
        # Create agent status display
        create_agent_status()
        
    except Exception as e:
        st.error(f"Error loading Voice Assistant: {str(e)}")
        st.info("Please ensure voice dependencies are installed: `pip install streamlit-audio`")

# Footer with system information
st.markdown("---")
st.markdown("### 🚀 System Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **📋 Email Organizer Tab:**
    - Processes email datasets with AI agents
    - Multi-level categorization system
    - Spam detection and filtering
    - Priority-based organization
    - RAG-powered chat assistant
    - Interactive dashboard with analytics
    - Batch processing capabilities
    """)

with col2:
    st.markdown("""
    **✉️ Email Reply System Tab:**
    - Real-time email reply generation
    - CrewAI multi-agent workflow
    - Web research integration
    - Context-aware responses
    - Performance tracking
    - Processing history
    - Sample email templates
    """)

with col3:
    st.markdown("""
    **🎤 Voice Assistant Tab:**
    - Natural language voice commands
    - Real-time voice processing
    - AI-powered voice recognition
    - Text-to-speech responses
    - Email action automation
    - Multi-agent coordination
    - Hands-free email management
    """)

# Technical details
with st.expander("🔧 Technical Architecture", expanded=False):
    st.markdown("""
    **AI Frameworks Used:**
    - **LangChain** - Core agent framework
    - **CrewAI** - Multi-agent orchestration (Reply System)
    - **LangGraph** - Workflow management (Organizer)
    - **Groq API** - LLM inference with Llama3 models
    
    **Agent Architecture:**
    - **Email Organizer**: Categorization, Priority, Action, Sentiment, Spam, RAG agents
    - **Email Reply**: Categorizer, Researcher, Writer agents
    
    **Data Processing:**
    - CSV dataset processing
    - Vector database with Chroma
    - Semantic search capabilities
    - Real-time processing pipelines
    """)

# Quick start guide
with st.expander("🚀 Quick Start Guide", expanded=True):
    st.markdown("""
    ### Getting Started:
    
    **1. Email Organizer Tab:**
    - Click "Process Emails with AI" to analyze your email dataset
    - Use filters to find specific emails
    - Try the floating chat assistant for email queries
    - Download results for further analysis
    
    **2. Email Reply System Tab:**
    - Enter an email or use a sample template
    - Click "Process Email & Generate Reply"
    - Review the AI-generated response
    - Check processing statistics and history
    
    **3. API Configuration:**
    - Ensure `GROQ_API_KEY` is set in `config.py`
    - For RAG features, install: `pip install langchain_ollama langchain_chroma`
    - For reply system, install: `pip install crewai duckduckgo-search`
    """)

# System status
st.markdown("---")
st.markdown("### 📊 System Status")

col1, col2, col3 = st.columns(3)

with col1:
    try:
        from config import Config
        if Config.GROQ_API_KEY:
            st.success("✅ Groq API Configured")
        else:
            st.warning("⚠️ Groq API Key Missing")
    except:
        st.error("❌ Configuration Error")

with col2:
    try:
        import crewai
        st.success("✅ CrewAI Available")
    except ImportError:
        st.warning("⚠️ CrewAI Not Installed")

with col3:
    try:
        import langchain_chroma
        st.success("✅ RAG System Available")
    except ImportError:
        st.warning("⚠️ RAG Dependencies Missing")

st.markdown("---")
st.markdown("🤖 **Advanced Email Management System** - Powered by AI agents and modern frameworks")
st.markdown("Built with Streamlit, LangChain, CrewAI, and Groq API")
