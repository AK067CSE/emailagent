"""
Intelligent Email Reply System - Streamlit Interface
Separate tab for email reply functionality using CrewAI agents
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import json
import sys
from typing import Dict, Any, List

from email_reply_agents import EmailReplyOrchestrator
from config import Config

def validate_email_content(email_content: str) -> bool:
    """Validate email content for processing"""
    if not email_content or not email_content.strip():
        return False
    if len(email_content.strip()) < 10:
        return False
    return True

def main():
    """Main email reply application"""
    # Only run the full app if this is the main script
    if __name__ != "__main__" and 'streamlit' not in sys.modules:
        return
    
    # Page configuration (only set if this is the main app)
    if 'streamlit' in sys.modules:
        st.set_page_config(
            page_title="Intelligent Email Reply System",
            page_icon="✉️",
            layout="wide",
            initial_sidebar_state="expanded"
        )

    # Custom CSS
    st.markdown("""
<style>
.email-input-container {
    background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    padding: 1.5rem;
    border-radius: 0.5rem;
    border: 1px solid #e5e7eb;
    margin-bottom: 1rem;
}

.reply-output-container {
    background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    padding: 1.5rem;
    border-radius: 0.5rem;
    border: 1px solid #bbf7d0;
    margin-bottom: 1rem;
}

.category-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 1rem;
    font-size: 0.875rem;
    font-weight: 600;
    margin-right: 0.5rem;
}

.category-price_enquiry { background-color: #fef3c7; color: #92400e; }
.category-customer_complaint { background-color: #fee2e2; color: #991b1b; }
.category-product_enquiry { background-color: #dbeafe; color: #1e3a8a; }
.category-customer_feedback { background-color: #f3e8ff; color: #6b21a8; }
.category-off_topic { background-color: #f3f4f6; color: #374151; }
.category-Price_Enquiry { background-color: #fef3c7; color: #92400e; }
.category-Customer_Complaint { background-color: #fee2e2; color: #991b1b; }
.category-Product_Enquiry { background-color: #dbeafe; color: #1e3a8a; }
.category-Customer_Feedback { background-color: #f3e8ff; color: #6b21a8; }
.category-Off_Topic { background-color: #f3f4f6; color: #374151; }

.metric-card {
    background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 0.5rem;
    text-align: center;
    border: 1px solid #2563eb;
}

.processing-timeline {
    background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #3b82f6;
    margin: 0.5rem 0;
}

.agent-log {
    background: #f9fafb;
    padding: 0.75rem;
    border-radius: 0.25rem;
    border-left: 3px solid #6b7280;
    margin: 0.25rem 0;
    font-family: monospace;
    font-size: 0.875rem;
}
</style>
""", unsafe_allow_html=True)

    # Initialize session state
    def init_session_state():
        """Initialize session state variables"""
        if 'reply_history' not in st.session_state:
            st.session_state.reply_history = []
        if 'current_email' not in st.session_state:
            st.session_state.current_email = ""
        if 'processing_result' not in st.session_state:
            st.session_state.processing_result = None
    
    init_session_state()

    # Header
    st.title("✉️ Intelligent Email Reply System")
    st.markdown("AI-powered email categorization, research, and reply generation using CrewAI agents")

    # Sidebar
    with st.sidebar:
        st.header("🔧 System Controls")
        
        # System status
        st.write("**CrewAI Reply Status:**")
        st.write("- Categorizer: ✅ Ready")
        st.write("- Researcher: ✅ Ready")
        st.write("- Writer: ✅ Ready")
        
        # Process email button
        if st.button("🤖 Process Email", type="primary", use_container_width=True):
            email_content = st.session_state.current_email
            
            if not email_content:
                st.warning("Please enter an email content first!")
                return
            
            with st.spinner("Processing with CrewAI agents..."):
                try:
                    orchestrator = EmailReplyOrchestrator()
                    result = orchestrator.process_email_reply(email_content)
                    st.session_state.processing_result = result
                    st.session_state.reply_history.append({
                        'timestamp': datetime.now(),
                        'email': email_content[:100] + "..." if len(email_content) > 100 else email_content,
                        'result': result,
                        'processing_time': getattr(result, 'processing_time', 0.0)
                    })
                    st.success("Email processed successfully!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Processing error: {str(e)}")
                    st.session_state.processing_result = {'error': str(e)}
        
        # Sample emails
        st.markdown("### 📋 Sample Emails")
        sample_emails = {
            "Price Enquiry": """Hi,
I'm interested in your product pricing. Can you send me a detailed price list?
Thanks,
John""",
            
            "Customer Complaint": """Hello,
I'm having issues with my recent order. The product arrived damaged and I need a replacement.
Please help resolve this ASAP.
Regards,
Sarah""",
            
            "Product Enquiry": """Dear Team,
I would like to know more about your product features and specifications.
Can you provide detailed information?
Best regards,
Mike""",
            
            "Customer Feedback": """Hi,
Just wanted to say your product is amazing! It exceeded my expectations.
Keep up the great work!
Cheers,
Lisa""",
            
            "Off Topic": """Hi,
Do you know what the weather will be like next week?
Just curious.
Thanks,
Bob"""
        }
        
        selected_sample = st.selectbox("Choose a sample email:", list(sample_emails.keys()))
        
        if st.button("📋 Use Sample Email"):
            st.session_state.current_email = sample_emails[selected_sample]
            st.rerun()

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📝 Email Input")
        
        # Email input
        with st.container():
            st.markdown('<div class="email-input-container">', unsafe_allow_html=True)
            
            email_content = st.text_area(
                "Enter Email Content:",
                value=st.session_state.current_email,
                height=300,
                placeholder="Paste or type the email content here...",
                key="email_input"
            )
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Validation
        is_valid = validate_email_content(email_content)
        
        if not is_valid and email_content:
            st.warning("⚠️ Please enter a valid email with sufficient content (at least 10 characters).")
        
        # Process button
        process_button = st.button(
            "🤖 Process Email & Generate Reply",
            type="primary",
            disabled=not is_valid,
            use_container_width=True
        )
        
        if process_button and is_valid:
            with st.spinner("🤖 AI agents are working... This may take a moment."):
                try:
                    orchestrator = EmailReplyOrchestrator()
                    result = orchestrator.process_email_reply(email_content)
                    st.session_state.processing_result = result
                    
                    # Add to history
                    st.session_state.reply_history.append({
                        "timestamp": datetime.now(),
                        "email": email_content[:100] + "..." if len(email_content) > 100 else email_content,
                        "result": result,
                        "processing_time": getattr(result, 'processing_time', 0.0)
                    })
                    
                    st.success("✅ Email processed successfully!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error processing email: {str(e)}")
                    st.error("Please check your API keys and try again.")

    with col2:
        st.header("📊 Processing Results")
        
        # Display results
        if st.session_state.processing_result:
            result = st.session_state.processing_result
            
            # Handle error case
            if isinstance(result, dict) and 'error' in result:
                st.error(f"❌ Error: {result['error']}")
                return
            
            # Handle CrewAI result (string output)
            if isinstance(result, str):
                st.markdown('<div class="reply-output-container">', unsafe_allow_html=True)
                st.markdown("### ✉️ Generated Reply")
                st.markdown("```")
                st.markdown(result)
                st.markdown("```")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                # Original structured result handling
                st.markdown('<div class="reply-output-container">', unsafe_allow_html=True)
                
                col_cat, col_conf = st.columns([2, 1])
                with col_cat:
                    # Create CSS-safe class name
                    category = getattr(result.category, 'category', 'Unknown') if hasattr(result, 'category') else 'Unknown'
                    css_class = category.replace(' ', '_')
                    st.markdown(f'<span class="category-badge category-{css_class}">{category}</span>', unsafe_allow_html=True)
                with col_conf:
                    confidence = getattr(result.category, 'confidence', 0) if hasattr(result, 'category') else 0
                    st.metric("Confidence", f"{confidence:.1%}")
                
                reasoning = getattr(result.category, 'reasoning', 'No reasoning available') if hasattr(result, 'category') else 'No reasoning available'
                st.markdown(f"**Reasoning:** {reasoning}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Research findings
                if hasattr(result, 'research') and result.research:
                    with st.expander("🔍 Research Findings", expanded=True):
                        st.markdown(result.research)
                
                # Generated reply
                if hasattr(result, 'reply'):
                    with st.expander("✉️ Generated Reply", expanded=True):
                        st.markdown("```")
                        st.markdown(result.reply.reply_content)
                        st.markdown("```")
                        
                        # Reply metadata
                        col_tone, col_points = st.columns(2)
                        with col_tone:
                            st.metric("Tone", result.reply.tone.title())
                        with col_points:
                            st.metric("Key Points", len(result.reply.key_points))
                
                # Processing timeline
                with st.expander("⏱️ Processing Details"):
                    st.markdown('<div class="processing-timeline">', unsafe_allow_html=True)
                    processing_time = getattr(result, 'processing_time', 0)
                    st.markdown(f"**Processing Time:** {processing_time:.2f} seconds")
                    st.markdown(f"**Tasks Completed:** {result.agent_performance.get('tasks_completed', 3) if hasattr(result, 'agent_performance') else 3}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Agent logs (if available)
                    if hasattr(result, 'agent_performance') and 'agent_logs' in result.agent_performance:
                        st.markdown("**Agent Activity:**")
                        for log in result.agent_performance['agent_logs'][-5:]:  # Show last 5 logs
                            st.markdown(f'<div class="agent-log">{log["agent"]}: {log["output_type"]}</div>', unsafe_allow_html=True)
        
        else:
            st.info("📋 Enter an email and click 'Process Email' to see results.")

    # Statistics Dashboard
    if st.session_state.reply_history:
        st.markdown("---")
        st.header("📈 Processing Statistics")
        
        # Calculate stats
        stats = {
            "total_processed": len(st.session_state.reply_history),
            "average_processing_time": 0,
            "category_distribution": {}
        }
        
        # Calculate average processing time
        if st.session_state.reply_history:
            total_time = sum([item.get('processing_time', 0) for item in st.session_state.reply_history])
            stats["average_processing_time"] = total_time / len(st.session_state.reply_history)
        
        # Calculate category distribution
        categories = []
        for item in st.session_state.reply_history:
            # Extract category from result if available, otherwise use "Processed"
            if isinstance(item.get('result'), str):
                categories.append("Processed")
            elif hasattr(item.get('result', {}), 'category'):
                categories.append(item['result'].category.category if item['result'].category else "Unknown")
            else:
                categories.append("Unknown")
        
        for category in categories:
            stats["category_distribution"][category] = stats["category_distribution"].get(category, 0) + 1
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3 style="margin: 0; font-size: 2rem;">{}</h3>
                <p style="margin: 0; opacity: 0.9;">Total Processed</p>
            </div>
            """.format(len(st.session_state.reply_history)), unsafe_allow_html=True)
        
        with col2:
            if stats["total_processed"] > 0:
                avg_time = stats["average_processing_time"]
                st.markdown("""
                <div class="metric-card">
                    <h3 style="margin: 0; font-size: 2rem;">{:.1f}s</h3>
                    <p style="margin: 0; opacity: 0.9;">Avg Time</p>
                </div>
                """.format(avg_time), unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="metric-card">
                    <h3 style="margin: 0; font-size: 2rem;">-</h3>
                    <p style="margin: 0; opacity: 0.9;">Avg Time</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            if stats["category_distribution"]:
                most_common = max(stats["category_distribution"], key=stats["category_distribution"].get)
                st.markdown("""
                <div class="metric-card">
                    <h3 style="margin: 0; font-size: 1.2rem;">{}</h3>
                    <p style="margin: 0; opacity: 0.9;">Most Common</p>
                </div>
                """.format(most_common.replace("_", " ").title()), unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="metric-card">
                    <h3 style="margin: 0; font-size: 2rem;">-</h3>
                    <p style="margin: 0; opacity: 0.9;">Most Common</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col4:
            if st.session_state.reply_history:
                last_time = st.session_state.reply_history[-1]["timestamp"]
                time_ago = (datetime.now() - last_time).total_seconds() / 60
                st.markdown("""
                <div class="metric-card">
                    <h3 style="margin: 0; font-size: 2rem;">{:.0f}m</h3>
                    <p style="margin: 0; opacity: 0.9;">Last Activity</p>
                </div>
                """.format(time_ago), unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="metric-card">
                    <h3 style="margin: 0; font-size: 2rem;">-</h3>
                    <p style="margin: 0; opacity: 0.9;">Last Activity</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Category distribution chart
        if stats["category_distribution"]:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = px.bar(
                    x=list(stats["category_distribution"].keys()),
                    y=list(stats["category_distribution"].values()),
                    title="Email Category Distribution",
                    labels={"x": "Category", "y": "Count"},
                    color=list(stats["category_distribution"].keys())
                )
                fig.update_xaxis(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.pie(
                    values=list(stats["category_distribution"].values()),
                    names=list(stats["category_distribution"].keys()),
                    title="Category Breakdown"
                )
                st.plotly_chart(fig, use_container_width=True)

    # Processing History
    if st.session_state.reply_history:
        st.markdown("---")
        st.header("📜 Processing History")
        
        # Create DataFrame for history
        history_df = pd.DataFrame(st.session_state.reply_history)
        
        # Display history table
        st.dataframe(
            history_df[['timestamp', 'category', 'processing_time']].rename(columns={
                'timestamp': 'Time',
                'category': 'Category',
                'processing_time': 'Time (s)'
            }),
            use_container_width=True
        )
        
        # Download history
        csv = history_df.to_csv(index=False)
        st.download_button(
            label="📥 Download History",
            data=csv,
            file_name=f"email_reply_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

    # Footer
    st.markdown("---")
    st.markdown("🤖 Powered by CrewAI agents with Llama3 and Groq API")

    # Instructions
    with st.expander("📖 How to Use", expanded=False):
        st.markdown("""
        ### Steps:
        1. **Enter Email Content**: Type or paste the email you want to respond to
        2. **Process Email**: Click the "Process Email & Generate Reply" button
        3. **Review Results**: See the categorization, research findings, and generated reply
        4. **Copy Reply**: Use the generated reply as-is or modify as needed
        
        ### Agent Workflow:
        - **Categorizer Agent**: Analyzes and categorizes the email
        - **Researcher Agent**: Finds relevant information for the response
        - **Writer Agent**: Crafts a professional email reply
        
        ### Categories:
        - **Price Enquiry**: Questions about pricing and costs
        - **Customer Complaint**: Issues or complaints about products/services
        - **Product Enquiry**: Questions about features and capabilities
        - **Customer Feedback**: Positive or negative feedback
        - **Off Topic**: Emails unrelated to business
        """)

# Run the app if this is the main script
if __name__ == "__main__":
    main()
