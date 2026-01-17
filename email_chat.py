import streamlit as st
import pandas as pd
from email_rag import EmailRAGSystem
import time

# Page configuration
st.set_page_config(
    page_title="Email RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.chat-message {
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    flex-direction: column;
}

.user-message {
    background-color: #e3f2fd;
    align-items: flex-end;
}

.assistant-message {
    background-color: #f5f5f5;
    align-items: flex-start;
}

.email-result {
    background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #3b82f6;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_system' not in st.session_state:
    with st.spinner("🔄 Initializing Email RAG System..."):
        st.session_state.rag_system = EmailRAGSystem()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_mode' not in st.session_state:
    st.session_state.current_mode = "chat"

# Header
st.title("🤖 Email RAG Assistant")
st.markdown("Chat with your email database and draft comprehensive emails using AI")

# Sidebar for mode selection
with st.sidebar:
    st.header("🎛️ Controls")
    
    mode = st.radio(
        "Select Mode:",
        ["💬 Chat with Emails", "📝 Draft Email", "🔍 Search Emails"],
        key="mode_selector"
    )
    
    if mode == "💬 Chat with Emails":
        st.session_state.current_mode = "chat"
    elif mode == "📝 Draft Email":
        st.session_state.current_mode = "draft"
    else:
        st.session_state.current_mode = "search"

# Main content area
if st.session_state.current_mode == "chat":
    st.header("💬 Chat with Your Emails")
    st.markdown("Ask questions about your emails, get insights, or discuss email-related topics.")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>👤 You:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>🤖 Assistant:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
    
    # Chat input
    user_input = st.chat_input("Ask about your emails...")
    
    if user_input:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Get AI response
        with st.spinner("🤖 Thinking..."):
            response = st.session_state.rag_system.chat_with_emails(user_input)
        
        # Add assistant response to history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Rerun to update the interface
        st.rerun()

elif st.session_state.current_mode == "draft":
    st.header("📝 Draft Comprehensive Email")
    st.markdown("AI-powered email drafting with context from your email database.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📧 Email Details")
        
        recipient = st.text_input("Recipient:", placeholder="john.doe@company.com")
        purpose = st.text_area("Purpose:", placeholder="Purpose of this email...", height=100)
        
        key_points_input = st.text_area(
            "Key Points (one per line):",
            placeholder="Point 1\nPoint 2\nPoint 3",
            height=150
        )
        
        tone = st.selectbox(
            "Tone:",
            ["professional", "friendly", "formal", "casual", "urgent"]
        )
        
        draft_button = st.button("📝 Draft Email", type="primary")
        
        if draft_button and recipient and purpose:
            key_points = [point.strip() for point in key_points_input.split('\n') if point.strip()]
            
            with st.spinner("📝 Drafting email..."):
                draft = st.session_state.rag_system.draft_comprehensive_email(
                    recipient, purpose, key_points, tone
                )
            
            st.session_state.draft_result = draft
    
    with col2:
        st.subheader("📋 Draft Result")
        
        if 'draft_result' in st.session_state:
            st.markdown("""
            <div class="email-result">
                <h4>📧 Drafted Email</h4>
                <div style="white-space: pre-wrap; font-family: monospace; background: white; padding: 1rem; border-radius: 0.25rem;">
                {}
                </div>
            </div>
            """.format(st.session_state.draft_result), unsafe_allow_html=True)
            
            # Copy button
            if st.button("📋 Copy to Clipboard"):
                st.write("Email copied to clipboard! (In a real app, this would copy to system clipboard)")

elif st.session_state.current_mode == "search":
    st.header("🔍 Search Email Database")
    st.markdown("Search through your emails using semantic similarity.")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input(
            "Search Query:",
            placeholder="Search for emails about...",
            value=""
        )
        
        k_results = st.slider("Number of results:", min_value=1, max_value=20, value=5)
        
        search_button = st.button("🔍 Search", type="primary")
        
        if search_button and search_query:
            with st.spinner("🔍 Searching emails..."):
                results = st.session_state.rag_system.search_emails(search_query, k_results)
            
            st.session_state.search_results = results
    
    with col2:
        st.subheader("📊 Search Stats")
        if 'search_results' in st.session_state:
            st.metric("Results Found", len(st.session_state.search_results))
    
    # Display search results
    if 'search_results' in st.session_state and st.session_state.search_results:
        st.subheader(f"📧 Found {len(st.session_state.search_results)} Results")
        
        for i, result in enumerate(st.session_state.search_results, 1):
            with st.expander(f"📧 {i}. {result['subject']}", expanded=False):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**📧 Subject:** {result['subject']}")
                    st.markdown(f"**👤 From:** {result['sender']}")
                    st.markdown(f"**📅 Date:** {result['date']}")
                    st.markdown("**📄 Content:**")
                    st.text_area("", value=result['content'], height=200, disabled=True, key=f"content_{i}")
                
                with col2:
                    st.markdown(f"**🆔 Email ID:** {result['email_id']}")
                    if st.button(f"📋 Copy Email {i}", key=f"copy_{i}"):
                        st.write(f"Email {i} content copied!")

# Sidebar additional features
with st.sidebar:
    st.markdown("---")
    st.subheader("🔧 Features")
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
    
    if st.button("🔄 Reset System"):
        st.session_state.clear()
        st.rerun()
    
    st.markdown("---")
    st.subheader("📈 System Info")
    st.markdown(f"**Mode:** {st.session_state.current_mode}")
    st.markdown(f"**Chat Messages:** {len(st.session_state.chat_history)}")
    
    if hasattr(st.session_state.rag_system, 'vector_store'):
        st.markdown("**Vector Database:** ✅ Ready")
    else:
        st.markdown("**Vector Database:** ❌ Not Ready")

# Footer
st.markdown("---")
st.markdown("🤖 Powered by RAG (Retrieval-Augmented Generation) with Groq embeddings")
