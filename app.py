import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Union
import json
import sys

from data_processor import EmailDataProcessor
from agents import EmailOrchestrator, EmailAnalysis
from config import Config

# Page configuration
st.set_page_config(
    page_title="Email Inbox Organizer",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .email-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e1e5e9;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .priority-high {
        border-left: 4px solid #ff4b4b;
    }
    .priority-medium {
        border-left: 4px solid #ffa500;
    }
    .priority-low {
        border-left: 4px solid #00c851;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .stSelectbox > div > div > select {
        background-color: #f0f2f6;
    }

    /* Hide the chat input completely */
    [data-testid="stTextArea"] {
        height: 1px !important;
        min-height: 1px !important;
        padding: 0 !important;
        border: none !important;
        opacity: 0 !important;
        position: absolute;
        left: -9999px;
        top: -9999px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the email dataset"""
    processor = EmailDataProcessor()
    return processor.load_dataset()

@st.cache_data
def process_emails(_emails_df, start_idx=0, batch_size=20):
    """Process emails and cache results using simple agents (fallback for now)"""
    processor = EmailDataProcessor()
    orchestrator = EmailOrchestrator(use_langgraph=False)  # Use simple agents for now
    
    # Convert DataFrame to list of dictionaries
    emails_list = []
    for _, row in _emails_df.iloc[start_idx:start_idx + batch_size].iterrows():
        email_data = processor.preprocess_email(row.to_dict())
        emails_list.append(email_data)
    
    # Process emails with simple agents
    st.info(f"Processing emails {start_idx + 1} to {min(start_idx + batch_size, len(_emails_df))} with AI agents... This may take a moment.")
    progress_bar = st.progress(0)
    
    results = []
    for i, email in enumerate(emails_list):
        try:
            analysis = orchestrator.process_email(email)
            results.append(analysis)
            progress_bar.progress((i + 1) / len(emails_list))
        except Exception as e:
            st.error(f"Error processing email {email['email_id']}: {str(e)}")
            continue
    
    return results, emails_list, start_idx + batch_size

def create_email_card(email_data: Dict, analysis: Union[Dict, EmailAnalysis]):
    """Create a styled email card with expandable content"""
    # Handle both LangGraph dict output and EmailAnalysis object
    if isinstance(analysis, dict):
        # LangGraph output format
        category = analysis.get('categories', [{}])[0].get('category', 'Unknown')
        priority = analysis.get('priority', {}).get('priority', 'Medium')
        action = analysis.get('suggested_actions', [{}])[0].get('action', 'Review')
        reasoning = analysis.get('suggested_actions', [{}])[0].get('reasoning', 'No reasoning provided')
        confidence = analysis.get('confidence_score', 0) / 100
        draft_response = None
        sentiment = 'neutral'  # LangGraph tools don't include sentiment yet
        
        priority_class = f"priority-{priority.lower()}"
    else:
        # EmailAnalysis object format (simple agents)
        category = analysis.category.category
        priority = analysis.priority.priority
        action = analysis.action.action
        reasoning = analysis.action.reasoning
        confidence = analysis.category.confidence
        draft_response = analysis.action.draft_response
        sentiment = analysis.sentiment or 'neutral'
        priority_class = f"priority-{priority.lower()}"
    
    # Clean up HTML from email body for display
    import re
    body_text = re.sub(r'<[^>]+>', '', email_data['body'])
    body_text = body_text.replace('&quot;', '"').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    body_text = body_text.replace('\n', ' ').strip()
    # Also clean up any remaining HTML entities and quotes
    body_text = re.sub(r'&\w+;', '', body_text)
    body_text = re.sub(r'\s+', ' ', body_text)  # Clean up multiple spaces
    
    # Create unique keys for expandable sections
    email_id = email_data['email_id']
    show_body_key = f"show_body_{email_id}"
    show_analysis_key = f"show_analysis_{email_id}"
    
    with st.container():
        # Email header with subject and metadata
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); padding: 1rem; border-radius: 0.5rem; border-left: 4px solid {
                '#ef4444' if priority == 'High' 
                else '#f59e0b' if priority == 'Medium' 
                else '#10b981'
            }; margin-bottom: 0.5rem;">
                <h3 style="margin: 0; color: #1f2937; font-size: 1.2rem;">{email_data['subject']}</h3>
                <p style="margin: 0.5rem 0; color: #6b7280; font-size: 0.9rem;">
                    <strong>From:</strong> {email_data['sender_name']} ({email_data['sender_email']})
                </p>
                <p style="margin: 0; color: #9ca3af; font-size: 0.8rem;">
                    {email_data['timestamp'].strftime('%B %d, %Y at %I:%M %p')}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="text-align: right; padding: 1rem;">
                <div style="background-color: #3b82f6; color: white; padding: 0.5rem 1rem; border-radius: 0.5rem; margin-bottom: 0.5rem; text-align: center;">
                    <strong>{category}</strong>
                </div>
                <div style="background-color: {
                    '#ef4444' if priority == 'High' 
                    else '#f59e0b' if priority == 'Medium' 
                    else '#10b981'
                }; color: white; padding: 0.5rem 1rem; border-radius: 0.5rem; margin-bottom: 0.5rem; text-align: center;">
                    <strong>{priority}</strong>
                </div>
                <div style="background-color: #6b7280; color: white; padding: 0.5rem 1rem; border-radius: 0.5rem; text-align: center;">
                    <small>{confidence:.1%}</small>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Email content section
        with st.expander("📧 Email Content", expanded=False):
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); padding: 1.5rem; border-radius: 0.5rem; border: 1px solid #0ea5e9; line-height: 1.6;">
                <p style="margin: 0; color: #0c4a6e; font-weight: 500; font-size: 1rem;">
                    {body_text}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # AI Analysis section
        with st.expander("🤖 AI Analysis & Actions", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #f59e0b; margin-bottom: 1rem;">
                    <h4 style="margin: 0 0 0.5rem 0; color: #92400e; font-weight: bold;">🎯 Recommended Action</h4>
                    <p style="margin: 0; color: #78350f; font-weight: bold; font-size: 1rem;">{action}</p>
                    <p style="margin: 0.5rem 0 0 0; color: #92400e; font-weight: 500; font-size: 0.95rem;">{reasoning}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #3b82f6; margin-bottom: 1rem;">
                    <h4 style="margin: 0 0 0.5rem 0; color: #1e3a8a; font-weight: bold;">📊 Analysis Details</h4>
                    <p style="margin: 0; color: #1e3a8a; font-weight: 600;"><strong>Category:</strong> {category}</p>
                    <p style="margin: 0; color: #1e3a8a; font-weight: 600;"><strong>Priority:</strong> {priority}</p>
                    <p style="margin: 0; color: #1e3a8a; font-weight: 600;"><strong>Confidence:</strong> {confidence:.1%}</p>
                    <p style="margin: 0; color: #1e3a8a; font-weight: 600;"><strong>Sentiment:</strong> {sentiment}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Draft response if available
            if draft_response:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%); padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #10b981;">
                    <h4 style="margin: 0 0 0.5rem 0; color: #065f46; font-weight: bold;">✉️ Draft Response</h4>
                    <p style="margin: 0; color: #065f46; font-weight: 500; font-style: italic; font-size: 1rem;">{draft_response}</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")  # Separator between emails

def create_dashboard(analyses: List[Dict], emails_df: pd.DataFrame):
    """Create dashboard with metrics and charts"""
    
    # Metrics - Handle both LangGraph dict and EmailAnalysis object formats
    total_emails = len(analyses)
    
    # Count high priority emails
    high_priority = 0
    categories = {}
    
    for analysis in analyses:
        if isinstance(analysis, dict):
            # LangGraph format
            priority = analysis.get('priority', {}).get('priority', 'Medium')
            category_list = analysis.get('categories', [])
            if category_list:
                category = category_list[0].get('category', 'Unknown')
            else:
                category = 'Unknown'
        else:
            # EmailAnalysis object format
            priority = analysis.priority.priority
            category = analysis.category.category
        
        if priority == 'High':
            high_priority += 1
        
        categories[category] = categories.get(category, 0) + 1
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3 style="margin: 0; font-size: 2rem;">{}</h3>
            <p style="margin: 0; opacity: 0.9;">Total Emails</p>
        </div>
        """.format(total_emails), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3 style="margin: 0; font-size: 2rem;">{}</h3>
            <p style="margin: 0; opacity: 0.9;">High Priority</p>
        </div>
        """.format(high_priority), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3 style="margin: 0; font-size: 2rem;">{}</h3>
            <p style="margin: 0; opacity: 0.9;">Categories</p>
        </div>
        """.format(len(categories)), unsafe_allow_html=True)
    
    with col4:
        # Calculate average confidence
        confidences = []
        for analysis in analyses:
            if isinstance(analysis, dict):
                conf = analysis.get('confidence_score', 0)
            else:
                conf = analysis.category.confidence * 100
            confidences.append(conf)
        
        avg_confidence = np.mean(confidences) if confidences else 0
        st.markdown("""
        <div class="metric-card">
            <h3 style="margin: 0; font-size: 2rem;">{:.1f}%</h3>
            <p style="margin: 0; opacity: 0.9;">Avg Confidence</p>
        </div>
        """.format(avg_confidence), unsafe_allow_html=True)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Category distribution
        if categories:
            fig = px.pie(
                values=list(categories.values()),
                names=list(categories.keys()),
                title="Email Categories Distribution"
            )
            st.plotly_chart(fig, use_container_width=False)
    
    with col2:
        # Priority distribution
        priority_counts = {'High': 0, 'Medium': 0, 'Low': 0}
        for analysis in analyses:
            if isinstance(analysis, dict):
                priority = analysis.get('priority', {}).get('priority', 'Medium')
            else:
                priority = analysis.priority.priority
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
        
        fig = px.bar(
            x=list(priority_counts.keys()),
            y=list(priority_counts.values()),
            title="Priority Distribution",
            color=list(priority_counts.keys()),
            color_discrete_map={'High': '#ff4b4b', 'Medium': '#ffa500', 'Low': '#00c851'}
        )
        st.plotly_chart(fig, use_container_width=False)

def main():
    """Main application"""
    # Only run the full app if this is the main script
    if __name__ != "__main__" and 'streamlit' not in sys.modules:
        return
    
    st.title("📧 Email Inbox Organizer")
    st.markdown("Intelligently organize, prioritize, and take action on your emails")
    
    # Load data
    with st.spinner("Loading email dataset..."):
        emails_df = load_data()
    
    st.success(f"Loaded {len(emails_df)} emails from dataset")
    
    # Initialize session state variables
    if 'current_start_idx' not in st.session_state:
        st.session_state.current_start_idx = 0
    if 'all_analyses' not in st.session_state:
        st.session_state.all_analyses = []
    if 'all_processed_emails' not in st.session_state:
        st.session_state.all_processed_emails = []
    if 'orchestrator' not in st.session_state:
        st.session_state.orchestrator = EmailOrchestrator(use_langgraph=False)
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    if 'chat_mode' not in st.session_state:
        st.session_state.chat_mode = 'chat'
    
    # Hidden component to capture chat requests
    chat_request = st.text_area("Chat Request", key="chat_input", height=1)
    
    # Handle RAG chat interactions
    if chat_request:
        try:
            request_data = json.loads(chat_request)
            
            # Handle different request types
            if request_data.get('type') == 'toggle_chat':
                st.session_state.show_chat = not st.session_state.show_chat
                st.session_state.chat_input = ""
                st.rerun()
            elif request_data.get('type') == 'change_mode':
                st.session_state.chat_mode = request_data.get('message', 'chat')
                st.session_state.chat_input = ""
                st.rerun()
            else:
                # Regular chat message
                orchestrator = st.session_state.orchestrator
                
                # Add user message to history
                st.session_state.chat_messages.append({"role": "user", "content": request_data['message']})
                
                if request_data['type'] == 'chat':
                    response = orchestrator.chat_with_emails(request_data['message'])
                elif request_data['type'] == 'draft':
                    response = orchestrator.draft_email(request_data['message'])
                elif request_data['type'] == 'search':
                    response = orchestrator.search_emails(request_data['message'])
                
                # Add assistant response to history
                st.session_state.chat_messages.append({"role": "assistant", "content": response})
                
                # Clear input after processing
                st.session_state.chat_input = ""
                st.rerun()
            
        except json.JSONDecodeError:
            pass
        except Exception as e:
            st.error(f"Chat error: {str(e)}")
    
    # Display response if available
    if 'rag_response' in st.session_state and st.session_state.rag_response:
        response = st.session_state.rag_response
        if isinstance(response, str):
            st.info(f"🤖 Assistant: {response}")
        else:
            st.info(f"🔍 Found {len(response)} results")
            for result in response[:3]:
                st.write(f"📧 {result['subject']} - {result['sender']}")
        
        # Clear response after displaying
        st.session_state.rag_response = None
    
    # Sidebar filters
    st.sidebar.title("🔍 Filters & Options")
    
    # Email Assistant Chatbot in Sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 Email Assistant")
    
    # Chat modes
    chat_mode_col1, chat_mode_col2, chat_mode_col3 = st.sidebar.columns(3)
    with chat_mode_col1:
        if st.sidebar.button("💬 Chat", key="sidebar_mode_chat", type="primary" if st.session_state.chat_mode == 'chat' else "secondary"):
            st.session_state.chat_mode = 'chat'
            st.rerun()
    with chat_mode_col2:
        if st.sidebar.button("✉️ Draft", key="sidebar_mode_draft", type="primary" if st.session_state.chat_mode == 'draft' else "secondary"):
            st.session_state.chat_mode = 'draft'
            st.rerun()
    with chat_mode_col3:
        if st.sidebar.button("🔍 Search", key="sidebar_mode_search", type="primary" if st.session_state.chat_mode == 'search' else "secondary"):
            st.session_state.chat_mode = 'search'
            st.rerun()
    
    st.sidebar.markdown("---")
    
    # Display chat messages
    if st.session_state.chat_messages:
        for message in st.session_state.chat_messages:
            if message["role"] == "user":
                st.sidebar.chat_message("You", message["content"])
            else:
                st.sidebar.chat_message("Assistant", message["content"])
    else:
        st.sidebar.info("👋 Hi! I'm your email assistant. I can help you chat with your emails, draft responses, or search through them. How can I help you today?")
    
    # Chat input
    user_input = st.sidebar.chat_input(
        "Type your message here...", 
        key="sidebar_user_chat_input"
    )
    
    chat_col1, chat_col2 = st.sidebar.columns([3, 1])
    with chat_col1:
        if st.sidebar.button("💬 Send", key="sidebar_send_chat", type="primary"):
            if user_input:
                # Add user message
                st.session_state.chat_messages.append({"role": "user", "content": user_input})
                
                # Process with orchestrator
                orchestrator = st.session_state.orchestrator
                try:
                    if st.session_state.chat_mode == 'chat':
                        response = orchestrator.chat_with_emails(user_input)
                    elif st.session_state.chat_mode == 'draft':
                        response = orchestrator.draft_email(user_input)
                    elif st.session_state.chat_mode == 'search':
                        response = orchestrator.search_emails(user_input)
                    
                    # Add assistant response
                    st.session_state.chat_messages.append({"role": "assistant", "content": response})
                    
                    # Clear input
                    st.session_state.sidebar_user_chat_input = ""
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Chat error: {str(e)}")
                    st.session_state.chat_messages.append({"role": "assistant", "content": f"Error: {str(e)}"})
                    st.rerun()
    with chat_col2:
        if st.sidebar.button("🗑️ Clear", key="sidebar_clear_chat"):
            st.session_state.chat_messages = []
            st.rerun()
    
    st.sidebar.markdown("---")
    
    # Process emails button
    if st.sidebar.button("🚀 Process Emails with AI", type="primary"):
        with st.spinner("Processing emails with AI agents..."):
            analyses, processed_emails, next_idx = process_emails(emails_df, st.session_state.current_start_idx)
            
            # Add to session state
            st.session_state.all_analyses.extend(analyses)
            st.session_state.all_processed_emails.extend(processed_emails)
            st.session_state.current_start_idx = next_idx
            
            st.success(f"Successfully processed {len(analyses)} emails!")
    
    # Check if emails have been processed
    if not st.session_state.all_analyses:
        st.info("Click 'Process Emails with AI' to start analyzing your emails with AI agents.")
        return
    
    analyses = st.session_state.all_analyses
    processed_emails = st.session_state.all_processed_emails
    
    # Show current batch info
    st.info(f"Currently showing {len(analyses)} processed emails (from {st.session_state.current_start_idx - len(analyses) + 1} to {st.session_state.current_start_idx})")
    
    # Load Next 20 button
    if st.session_state.current_start_idx < len(emails_df):
        if st.button("📭 Load Next 20 Emails", type="secondary"):
            with st.spinner("Processing next batch of emails..."):
                new_analyses, new_processed_emails, next_idx = process_emails(emails_df, st.session_state.current_start_idx)
                
                # Add to session state
                st.session_state.all_analyses.extend(new_analyses)
                st.session_state.all_processed_emails.extend(new_processed_emails)
                st.session_state.current_start_idx = next_idx
                
                st.success(f"Successfully processed {len(new_analyses)} more emails!")
                st.rerun()
    else:
        st.success("🎉 All emails have been processed!")
    
    # Filters
    st.sidebar.subheader("Filters")
    
    # Category filter
    all_categories = set()
    all_priorities = set()
    all_sentiments = set()
    
    for analysis in analyses:
        if isinstance(analysis, dict):
            # LangGraph format
            category_list = analysis.get('categories', [])
            if category_list:
                all_categories.add(category_list[0].get('category', 'Unknown'))
            priority = analysis.get('priority', {}).get('priority', 'Medium')
            all_priorities.add(priority)
            all_sentiments.add('neutral')  # LangGraph doesn't include sentiment yet
        else:
            # EmailAnalysis object format
            all_categories.add(analysis.category.category)
            all_priorities.add(analysis.priority.priority)
            all_sentiments.add(analysis.sentiment or 'neutral')
    
    selected_categories = st.sidebar.multiselect(
        "Category",
        list(all_categories),
        default=list(all_categories)
    )
    
    # Priority filter
    selected_priorities = st.sidebar.multiselect(
        "Priority",
        list(all_priorities),
        default=list(all_priorities)
    )
    
    # Sentiment filter
    selected_sentiments = st.sidebar.multiselect(
        "Sentiment",
        list(all_sentiments),
        default=list(all_sentiments)
    )
    
    # Search
    search_query = st.sidebar.text_input("🔍 Search emails...", placeholder="Search in subject or body...")
    
    # Filter analyses
    filtered_indices = []
    
    for i, analysis in enumerate(analyses):
        email = processed_emails[i]
        
        # Extract category and priority based on format
        if isinstance(analysis, dict):
            # LangGraph format
            category_list = analysis.get('categories', [])
            if category_list:
                category = category_list[0].get('category', 'Unknown')
            else:
                category = 'Unknown'
            priority = analysis.get('priority', {}).get('priority', 'Medium')
            sentiment = 'neutral'
        else:
            # EmailAnalysis object format
            category = analysis.category.category
            priority = analysis.priority.priority
            sentiment = analysis.sentiment or 'neutral'
        
        # Apply filters
        if category not in selected_categories:
            continue
        if priority not in selected_priorities:
            continue
        if sentiment not in selected_sentiments:
            continue
        
        # Search filter
        if search_query:
            search_lower = search_query.lower()
            if (search_lower not in email['subject'].lower() and 
                search_lower not in email['body'].lower()):
                continue
        
        filtered_indices.append(i)
    
    # Display dashboard
    create_dashboard(analyses, emails_df)
    
    # Display filtered emails
    st.header(f"📧 Email List ({len(filtered_indices)} filtered emails)")
    
    if not filtered_indices:
        st.warning("No emails match your filters.")
        return
    
    # Display emails
    for i in filtered_indices:
        email_data = processed_emails[i]
        analysis = analyses[i]
        create_email_card(email_data, analysis)
    
    # Download results
    if st.sidebar.button("📥 Download Results"):
        # Prepare data for download
        results_data = []
        for i in filtered_indices:
            email = processed_emails[i]
            analysis = analyses[i]
            
            # Handle both LangGraph dict and EmailAnalysis object formats
            if isinstance(analysis, dict):
                # LangGraph format
                category_list = analysis.get('categories', [])
                if category_list:
                    category = category_list[0].get('category', 'Unknown')
                    category_confidence = category_list[0].get('confidence', 0)
                else:
                    category = 'Unknown'
                    category_confidence = 0
                
                priority = analysis.get('priority', {}).get('priority', 'Medium')
                urgency_score = analysis.get('priority', {}).get('urgency_score', 5)
                
                action_list = analysis.get('suggested_actions', [])
                if action_list:
                    action = action_list[0].get('action', 'Review')
                    action_reasoning = action_list[0].get('reasoning', 'No reasoning')
                else:
                    action = 'Review'
                    action_reasoning = 'No reasoning'
                
                confidence = analysis.get('confidence_score', 0)
                sentiment = 'neutral'
            else:
                # EmailAnalysis object format
                category = analysis.category.category
                category_confidence = analysis.category.confidence
                priority = analysis.priority.priority
                urgency_score = analysis.priority.urgency_score
                action = analysis.action.action
                action_reasoning = analysis.action.reasoning
                confidence = 0.8  # Default confidence for simple agents
                sentiment = analysis.sentiment or 'neutral'
            
            results_data.append({
                'email_id': email['email_id'],
                'sender': email['sender_email'],
                'subject': email['subject'],
                'category': category,
                'category_confidence': category_confidence,
                'priority': priority,
                'urgency_score': urgency_score,
                'action': action,
                'action_reasoning': action_reasoning,
                'confidence': confidence,
                'sentiment': sentiment,
                'timestamp': email['timestamp'].isoformat()
            })
        
        results_df = pd.DataFrame(results_data)
        csv = results_df.to_csv(index=False)
        st.sidebar.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"email_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
