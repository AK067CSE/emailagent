# Unified Email Chatbot - Command Plan & Architecture

## 🎯 **Core Design Principle**
Single chatbot interface where user types natural commands → orchestrator routes to appropriate agents → fully automated workflow

---

## 📋 **Command Categories & Routing Logic**

### 1. **Routing Commands** (Primary - handled by orchestrator)
```
Pattern → Agent System → Action
```

#### 🔍 **Search Commands**
- `search [query]` → RAG System → Search emails
- `find [topic] in emails` → RAG System → Semantic search
- `lookup [information]` → RAG System → Find specific data

#### 💬 **Chat Commands**  
- `chat about [topic]` → RAG System → Chat with emails
- `ask [question]` → RAG System → Query email database
- `tell me about [subject]` → RAG System → Get email insights

#### 📝 **Draft Commands**
- `draft email to [recipient] about [topic]` → RAG System → Draft email
- `write email for [recipient] regarding [subject]` → RAG System → Compose
- `compose [recipient] [topic]` → RAG System → Quick draft

#### ✉️ **Reply Commands**
- `reply to [email content]` → CrewAI Reply → Multi-agent generation
- `respond to [message]` → CrewAI Reply → Create response
- `generate reply for [text]` → CrewAI Reply → AI-powered reply

#### 🏷️ **Organization Commands**
- `categorize [emails/dataset]` → Email Organizer → Classification
- `classify [messages]` → Email Organizer → Categorization
- `organize [inbox]` → Email Organizer → Structure emails

#### 🔍 **Filter Commands**
- `filter [category/priority]` → Email Organizer → Filter results
- `show [type] emails` → Email Organizer → Display filtered
- `list [criteria]` → Email Organizer → List matching emails

#### 🎛️ **System Commands**
- `status` → Orchestrator → Show all system statuses
- `help` → Orchestrator → Display command guide
- `clear` → UI → Clear chat history
- `quit/exit` → Orchestrator → End session

---

## 🤖 **Agent System Integration**

### **RAG System** (`email_rag.py`)
- **Capabilities**: Chat, Search, Draft
- **Data Source**: Chroma vector database from CSV
- **LLM**: Groq Llama-3.1-8b-instant
- **Use Cases**: 
  - Answer questions about email content
  - Find relevant emails by semantic search
  - Draft contextually-aware emails

### **CrewAI Reply System** (`email_reply_agents.py`)
- **Capabilities**: Categorize, Research, Write
- **Agents**: Categorizer → Researcher → Writer
- **LLM**: Groq Llama-3.1-8b-instant
- **Tools**: DuckDuckGo web search
- **Use Cases**:
  - Analyze incoming email content
  - Research relevant information online
  - Generate professional replies

### **Email Organizer** (`agents.py`)
- **Capabilities**: Categorize, Prioritize, Action, Spam detection
- **LLM**: Groq Llama-3.1-8b-instant
- **Use Cases**:
  - Process email datasets
  - Multi-level categorization
  - Priority assignment
  - Action recommendations

### **Voice Agent** (when integrated)
- **Capabilities**: Voice command processing
- **LLM**: Groq Whisper-large-v3 for transcription
- **Use Cases**:
  - Process voice commands
  - Convert speech to text for routing
  - Provide voice feedback

---

## 🔄 **Workflow Examples**

### **Example 1: Customer Inquiry Processing**
```
User: "search pricing plans"
↓
RAG System: Finds 5 relevant emails about pricing
↓
User: "draft email to customer@example.com about pricing follow-up"
↓
RAG System: Drafts email using previous pricing context
↓
User: "reply to 'I need enterprise pricing info'"
↓
CrewAI: Categorizes as "Price Enquiry" → Researches current pricing → Writes professional reply
```

### **Example 2: Email Organization**
```
User: "categorize all emails"
↓
Email Organizer: Processes dataset → Categories (Work, Personal, etc.) → Shows statistics
↓
User: "filter high priority"
↓
Email Organizer: Shows only high-priority emails with action recommendations
```

### **Example 3: Multi-Agent Coordination**
```
User: "chat What are the main customer complaints?"
↓
RAG System: Searches emails → Finds complaint patterns → Summarizes main issues
↓
User: "generate reply for customer complaint about delivery"
↓
CrewAI: Categorizes complaint → Researches delivery policies → Writes empathetic response
```

---

## 🎮 **Natural Language Processing**

### **Command Variations Supported**
- **Case insensitive**: all commands work in any case
- **Natural language**: "Can you find emails about pricing?" works
- **Partial commands**: "draft email to john" prompts for recipient
- **Contextual**: System remembers previous commands in session

### **Error Handling**
- **Graceful fallbacks**: If agent unavailable, suggests alternatives
- **Clear error messages**: User-friendly error descriptions
- **Recovery**: System continues operating even if one agent fails

---

## 🚀 **Implementation Status**

✅ **Completed Components:**
- [x] Orchestrator (`orchestrator.py`)
- [x] RAG System (`email_rag.py`) 
- [x] CrewAI Reply (`email_reply_agents.py`)
- [x] Email Organizer (`agents.py`)
- [x] Streamlit Chatbot UI (`unified_chatbot.py`)

✅ **Testing Ready:**
- [x] Command routing logic
- [x] Multi-agent coordination
- [x] Error handling
- [x] Session management

🎯 **Next Steps:**
1. Run comprehensive test script
2. Verify all command categories work
3. Test Streamlit chatbot interface
4. Validate agent interoperation

---

## 💡 **Usage Instructions**

**For Development:**
```bash
# Test orchestrator directly
python orchestrator.py

# Test Streamlit interface
streamlit run unified_chatbot.py
```

**For Users:**
1. Open chatbot interface
2. Type natural commands
3. All agents work automatically
4. No manual tab switching needed
