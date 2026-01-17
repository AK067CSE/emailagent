# Email Inbox Organizer
AGENTIC DESIGN PATTERN

<img width="1656" height="716" alt="image" src="https://github.com/user-attachments/assets/22c1e1e7-049e-4b44-9bff-38959ba16ca5" />

<img width="435" height="580" alt="image" src="https://github.com/user-attachments/assets/0da5d690-29af-4ab8-90c3-9649aace8729" />
<img width="241" height="581" alt="image" src="https://github.com/user-attachments/assets/1d876bda-1140-4d12-b1c3-dcd047630493" />
<img width="216" height="580" alt="image" src="https://github.com/user-attachments/assets/1d34d7a8-7c37-4c24-ac63-51d0ee12c60b" />
<img width="201" height="576" alt="image" src="https://github.com/user-attachments/assets/1da3f6f5-98e1-4ca2-b38b-641a88783098" />
<img width="225" height="590" alt="image" src="https://github.com/user-attachments/assets/d17256d5-ee07-47b8-ba7f-817af034c188" />
<img width="212" height="578" alt="image" src="https://github.com/user-attachments/assets/21cc5085-255c-4308-a5ec-a44ab53d3c43" />
<img width="210" height="579" alt="image" src="https://github.com/user-attachments/assets/982b65bf-22bd-4602-8bb1-388fc940dd0f" />






An intelligent email inbox organizer that automatically categorizes, prioritizes, and suggests actions for incoming emails using multi-agent AI systems.

## 🚀 Features

### Core Functionality
- **Email Organization**: Automatically categorizes emails into meaningful categories
- **Priority Assignment**: Assigns priority levels (High/Medium/Low) to help users focus on what matters
- **Action Recommendations**: Suggests appropriate actions (reply, schedule meeting, archive, etc.) with reasoning
- **Sentiment Analysis**: Analyzes email sentiment to understand emotional tone
- **Draft Responses**: Generates draft responses for emails requiring replies

### User Interface
- **Modern Streamlit UI**: Clean, intuitive interface with filtering and search capabilities
- **Real-time Dashboard**: Visual analytics with charts and metrics
- **Advanced Filtering**: Filter by category, priority, sentiment, sender, or date range
- **Search Functionality**: Full-text search across email subjects and bodies
- **Export Results**: Download analysis results as CSV

### Architecture
- **Multi-Agent System**: Uses specialized AI agents for different tasks
- **Groq LLM Integration**: Fast, efficient AI processing with Groq
- **LangGraph Framework**: Robust agent orchestration and workflow management
- **Error Handling**: Comprehensive retry logic and graceful error handling
- **Modular Design**: Clean, maintainable codebase with separation of concerns

## 📋 Requirements

- Python 3.8+
- Groq API key (free tier available)
- All dependencies listed in `requirements.txt`

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd emailorganizer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Copy `.env.example` → `.env` and fill in values.
   
   Minimum required:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```
   
   Optional (only needed for LiveKit / streaming voice setups):
   ```env
   LIVEKIT_URL=wss://...
   LIVEKIT_API_KEY=...
   LIVEKIT_API_SECRET=...
   ```

## 🚀 Running the Application

### Option A: Unified Chatbot (recommended)
Run the single interface that routes to all agents (RAG + Organizer + CrewAI + Voice):

```bash
streamlit run unified_chatbot.py
```

### Option B: Inbox Organizer UI
```bash
streamlit run app.py
```

### Option C: Email Reply UI (CrewAI)
```bash
streamlit run email_reply_app.py
```

### Option D: Voice UI (legacy)
```bash
streamlit run voice_streamlit_app.py
```

2. **Open your browser**
   Navigate to `http://localhost:8501` (or the URL shown in your terminal)

3. **Process emails**
   - Click "Process Emails with AI" to analyze the dataset
   - Wait for the AI agents to process the emails (this may take a moment)
   - Explore the dashboard and filtered email list

## 📊 Usage Guide

## 🎤 Voice (Microphone) Setup

The unified chatbot supports real microphone capture using `streamlit-webrtc` and Groq Whisper.

1. Run:
   ```bash
   streamlit run unified_chatbot.py
   ```
2. In the sidebar, set **Voice Input Mode** to **Use microphone (WebRTC)**.
3. In the WebRTC component area:
   - Click **START**
   - Allow microphone permissions in your browser
   - Speak for 2–5 seconds
   - Click **📝 Transcribe**

If you see “Microphone is OFF”, you did not press **START** or your browser blocked mic permissions.

## 💬 Unified Chatbot Commands

### RAG (Search/Chat/Draft/List)
- `search pricing plans`
- `chat What are the main customer concerns?`
- `draft email to client@example.com about follow-up meeting`
- `list all`
- `list from John`
- `list thread thread_001`
- `emails invoice`

### Email Organizer (Categorize/Priority/Actions)
- `organize all`
- `categorize all emails`
- `filter high priority`

### CrewAI Reply (Categorize + Research + Write)
- `reply to "I need pricing details"`

### Processing Emails
1. Load the application in your browser
2. Click the "Process Emails with AI" button in the sidebar
3. Wait for the multi-agent system to analyze all emails
4. View results in the dashboard and email list

### Filtering and Search
- **Category Filter**: Select specific email categories to view
- **Priority Filter**: Filter by High/Medium/Low priority emails
- **Sentiment Filter**: Filter by positive/negative/neutral sentiment
- **Search**: Use the search bar to find specific emails

### Dashboard Features
- **Metrics Overview**: Total emails, high priority count, categories, and average confidence
- **Category Distribution**: Pie chart showing email category breakdown
- **Priority Distribution**: Bar chart showing priority level distribution

### Email Cards
Each email card displays:
- Subject, sender, and timestamp
- Category and priority badges
- Email body preview
- Recommended action with reasoning
- Draft response (if applicable)
- Sentiment analysis

## 🏗️ Architecture Overview

### Multi-Agent System

The application uses a multi-agent architecture with specialized agents:

1. **Categorization Agent**: Classifies emails into predefined categories
2. **Priority Agent**: Assigns priority levels based on urgency and importance
3. **Action Agent**: Recommends appropriate actions and generates draft responses
4. **Sentiment Agent**: Analyzes emotional tone of emails
5. **Orchestrator**: Coordinates all agents and manages the workflow

### Data Flow

```
Email Dataset → Data Processor → Email Orchestrator → Specialized Agents → Analysis Results → Streamlit UI
```

### Key Components

- **`config.py`**: Configuration settings and constants
- **`data_processor.py`**: Handles dataset loading and preprocessing
- **`agents.py`**: Multi-agent system with specialized AI agents
- **`app.py`**: Streamlit user interface and main application logic

## 🤖 AI Agent Details

### Categorization Agent
- **Categories**: Work/Professional, Personal, Marketing/Newsletter, Notifications/System, Billing/Financial, Support/Customer Service, HR/Administrative, Security/Alert, Social/Community, Other
- **Output**: Category, confidence score, and reasoning

### Priority Agent
- **Priority Levels**: High (immediate attention), Medium (address soon), Low (address later)
- **Urgency Score**: 1-10 scale indicating time sensitivity
- **Factors**: Time sensitivity, sender importance, content urgency, deadlines

### Action Agent
- **Actions**: Reply Immediately, Schedule Meeting, Archive, Delete, Flag for Follow-up, Forward, Review Later, No Action Needed
- **Draft Responses**: Automatically generates response drafts when applicable
- **Reasoning**: Provides clear justification for recommended actions

### Sentiment Agent
- **Sentiments**: Positive, Negative, Neutral
- **Analysis**: Considers tone, language, and emotional indicators

## 📁 Dataset

The application works with a CSV dataset containing:
- `email_id`: Unique identifier
- `sender_email`: Email address of sender
- `sender_name`: Name of sender
- `subject`: Email subject line
- `body`: Email content
- `timestamp`: When email was received (ISO format)
- `has_attachment`: Boolean indicating attachments
- `thread_id`: Identifier for email conversations

A sample dataset is automatically generated if the provided dataset cannot be loaded.

## 🔧 Configuration

### Model Settings
- **Default Model**: Llama3-70B-8192 via Groq
- **Temperature**: 0.1 (for consistent outputs)
- **Max Tokens**: 2048
- **Retry Logic**: Up to 3 retries with exponential backoff

### Customization
You can customize:
- Email categories in `Config.EMAIL_CATEGORIES`
- Priority levels in `Config.PRIORITY_LEVELS`
- Action types in `Config.ACTION_TYPES`
- Model parameters in `Config`

## 🚨 Error Handling

- **API Failures**: Automatic retry with exponential backoff
- **Data Parsing**: Fallback to sample dataset if parsing fails
- **Individual Email Errors**: Continue processing other emails if one fails
- **Graceful Degradation**: Application remains functional even with partial failures

## 📈 Performance Considerations

- **Batch Processing**: Processes emails in batches to optimize API usage
- **Caching**: Streamlit caching for data and results
- **Rate Limiting**: Built-in delays to respect API rate limits
- **Memory Management**: Efficient data structures for large datasets

## 🔄 Future Enhancements

### Bonus Features (Planned)
- **Voice Integration**: Bidirectional voice streaming using Google ADK
- **Email Threading**: Conversation grouping and thread analysis
- **Bulk Actions**: Process multiple emails simultaneously
- **Custom Categories**: User-defined email categories
- **Advanced Analytics**: More sophisticated email insights

### Technical Improvements
- **Database Integration**: Store results in database for persistence
- **Real-time Processing**: Process emails as they arrive
- **Multi-language Support**: Handle emails in different languages
- **Integration**: Connect with email providers (Gmail, Outlook, etc.)

## 🐛 Troubleshooting

### Common Issues

1. **API Key Issues**
   - Ensure your Groq API key is valid and has sufficient credits
   - Check that the API key is properly set in environment variables

2. **Voice (Microphone) Issues**
   - Make sure you clicked **START** in the WebRTC component
   - Allow mic permissions in your browser
   - Install `streamlit-webrtc` and `av`:
     ```bash
     pip install streamlit-webrtc av
     ```

2. **Dataset Loading Issues**
   - Verify the CSV file format matches the expected structure
   - The application will fall back to a sample dataset if loading fails

3. **Processing Errors**
   - Individual email processing errors won't stop the entire process
   - Check the console for detailed error messages

4. **Performance Issues**
   - Processing is limited to 20 emails at a time for demo purposes
   - Adjust the limit in `app.py` for larger datasets

## 📄 License

This project is provided as part of an assignment. Please refer to the assignment guidelines for usage restrictions.

## 🤝 Support

For questions or issues related to this assignment, please contact the assignment provider within the specified timeframe.

---

**Note**: This application is designed for demonstration purposes and uses sample data. For production use, additional security measures and optimizations would be required.
