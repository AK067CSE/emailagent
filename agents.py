from typing import Dict, Any, List, Optional, TypedDict, Annotated, Sequence
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from tenacity import retry, stop_after_attempt, wait_exponential
from config import Config
import re

# LangGraph imports
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

# Pydantic models for structured outputs
class EmailCategory(BaseModel):
    category: str = Field(description="The email category")
    confidence: float = Field(description="Confidence score from 0 to 1")
    reasoning: str = Field(description="Reasoning for categorization")
    keywords: List[str] = Field(description="Keywords found in email", default_factory=list)
    coarse_genre: str = Field(description="Level 1 coarse genre classification", default="")
    content_type: str = Field(description="Level 2 content type classification", default="")
    primary_topic: str = Field(description="Level 3 primary topic classification", default="")
    emotional_tone: str = Field(description="Level 4 emotional tone classification", default="neutral")

class EmailPriority(BaseModel):
    priority: str = Field(description="Priority level: High, Medium, or Low")
    urgency_score: int = Field(description="Urgency score from 1 to 10")
    reasoning: str = Field(description="Reasoning for the priority assignment")

class EmailAction(BaseModel):
    action: str = Field(description="Recommended action")
    reasoning: str = Field(description="Reasoning for the action recommendation")
    draft_response: Optional[str] = Field(description="Draft response if applicable", default=None)

class EmailAnalysis(BaseModel):
    email_id: int
    category: EmailCategory
    priority: EmailPriority
    action: EmailAction
    sentiment: Optional[str] = Field(description="Email sentiment: positive, negative, or neutral", default=None)
    spam_detection: Optional[Dict[str, Any]] = Field(description="Spam detection results", default=None)

class BaseAgent:
    """Base class for all email processing agents"""
    
    def __init__(self, model_name: str = None):
        self.llm = ChatGroq(
            api_key=Config.GROQ_API_KEY,
            model=model_name or Config.DEFAULT_MODEL,
            temperature=Config.TEMPERATURE,
            max_tokens=Config.MAX_TOKENS
        )
    
    @retry(stop=stop_after_attempt(Config.MAX_RETRIES), 
           wait=wait_exponential(multiplier=1, min=4, max=10))
    def invoke_with_retry(self, prompt, **kwargs):
        """Invoke LLM with retry logic"""
        return self.llm.invoke(prompt, **kwargs)

class CategorizationAgent(BaseAgent):
    """Agent responsible for email categorization with multi-level classification"""
    
    def __init__(self):
        super().__init__()
        self.parser = JsonOutputParser(pydantic_object=EmailCategory)
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are an expert email categorization specialist using a sophisticated multi-level classification system.
            
            Your task is to categorize emails and provide USER-FRIENDLY category names (not codes).
            
            Create a MEANINGFUL COMBINED CATEGORY NAME that reflects the email's nature.
            
            AVAILABLE CATEGORIES (use these exact names):
            
            COARSE GENRE (Level 1):
            - Company Business Strategy
            - Purely Personal
            - Personal but Professional
            - Logistic Arrangements
            - Employment Arrangements
            - Document Editing Checking
            - Empty Message
            
            CONTENT TYPE (Level 2):
            - Includes New Text Forwarded
            - Forwarded Emails with Replies
            - Business Letters Documents
            - News Articles
            - Government Academic Reports
            - Government Actions
            - Press Releases
            - Legal Documents
            - Pointers to URLs
            - Newsletters
            - Business Humor
            - Non-Business Humor
            - Missing Attachments
            
            PRIMARY TOPICS (Level 3 - for business emails):
            - Regulations and Regulators
            - Internal Projects Progress Strategy
            - Company Image Current
            - Company Image Changing Influencing
            - Political Influence Contributions
            - California Energy Crisis Politics
            - Internal Company Policy
            - Internal Company Operations
            - Alliances Partnerships
            - Legal Advice
            - Talking Points
            - Meeting Minutes
            - Trip Reports
            
            EMOTIONAL TONE (Level 4 - if not neutral):
            - Jubilation
            - Hope Anticipation
            - Humor
            - Camaraderie
            - Admiration
            - Gratitude
            - Friendship Affection
            - Sympathy Support
            - Sarcasm
            - Secrecy Confidentiality
            - Worry Anxiety
            - Concern
            - Competitiveness Aggressiveness
            - Triumph Gloating
            - Pride
            - Anger Agitation
            - Sadness Despair
            - Shame
            - Dislike Scorn
            
            Analyze the email and provide a JSON response with this exact format:
            {{
                "category": "MEANINGFUL_COMBINED_CATEGORY_NAME",
                "confidence": 0.85,
                "reasoning": "clear_reasoning_for_classification",
                "keywords": ["keyword1", "keyword2", "keyword3"],
                "coarse_genre": "level_1_category_name",
                "content_type": "level_2_category_name", 
                "primary_topic": "level_3_category_name",
                "emotional_tone": "level_4_category_name_or_neutral"
            }}
            
            For the main "category" field, create a MEANINGFUL COMBINED NAME like:
            - "Business Strategy Document"
            - "Personal Meeting Request"
            - "Customer Feedback with Concern"
            - "Internal Project Update"
            - "Legal Document Review"
            - "Personal News Article"
            - "Employment Application"
            - "Company Policy Announcement"
            
            Consider:
            - Email content and subject matter
            - Sender-recipient relationship
            - Business vs personal context
            - Forwarded/replied content
            - Emotional indicators in language
            - Specific topics mentioned
            - Document attachments or references
            
            IMPORTANT: Use ONLY user-friendly names, no codes. Create meaningful combined category names. Return valid JSON only, no additional text.
            """),
            ("human", "Email to categorize:\n\nSender: {sender_name}\nEmail: {sender_email}\nSubject: {subject}\nBody: {body}\nHas Attachment: {has_attachment}\nTimestamp: {timestamp}")
        ])
    
    def categorize(self, email_data: Dict[str, Any]) -> EmailCategory:
        """Categorize an email using enhanced classification system"""
        try:
            prompt = self.prompt_template.format_messages(
                sender_name=email_data.get('sender_name', ''),
                sender_email=email_data.get('sender_email', ''),
                subject=email_data.get('subject', ''),
                body=email_data.get('body', ''),
                has_attachment=email_data.get('has_attachment', False),
                timestamp=email_data.get('timestamp', '')
            )
            
            response = self.invoke_with_retry(prompt)
            
            # Try to parse the response
            try:
                result = self.parser.parse(response.content)
                return EmailCategory(**result)
            except Exception as parse_error:
                # Fallback: try to extract JSON manually
                content = response.content.strip()
                if content.startswith('```json'):
                    content = content[7:-3]  # Remove ```json and ```
                elif content.startswith('```'):
                    content = content[3:-3]   # Remove ```
                
                # Try to fix common JSON issues
                content = content.replace('""', '"')  # Fix double quotes
                content = content.replace('\n', ' ')  # Remove newlines
                
                import json
                result = json.loads(content)
                return EmailCategory(**result)
                
        except Exception as e:
            # Ultimate fallback: return a default category
            return EmailCategory(
                category="Company Business Strategy",
                confidence=0.5,
                reasoning=f"Unable to determine specific category due to parsing error: {str(e)}",
                keywords=["business", "email"],
                coarse_genre="Company Business Strategy",
                content_type="Business Letters Documents",
                primary_topic="Internal Projects Progress Strategy",
                emotional_tone="neutral"
            )

class PriorityAgent(BaseAgent):
    """Agent responsible for assigning priority to emails"""
    
    def __init__(self):
        super().__init__()
        self.parser = JsonOutputParser(pydantic_object=EmailPriority)
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are an expert email prioritization specialist.
            Your task is to assign priority levels to emails based on urgency and importance.
            
            Priority Levels:
            - High: Requires immediate attention (urgent deadlines, critical issues, time-sensitive matters)
            - Medium: Should be addressed soon (important but not urgent, standard business communications)
            - Low: Can be addressed later (newsletters, general updates, non-essential information)
            
            Analyze the email and provide a JSON response with this exact format:
            {{
                "priority": "High|Medium|Low",
                "urgency_score": 7,
                "reasoning": "clear_reasoning_for_priority_assignment"
            }}
            
            Consider factors like:
            - Time sensitivity mentioned in the email
            - Importance of the sender
            - Content urgency indicators
            - Deadlines or action required dates
            
            IMPORTANT: Return valid JSON only, no additional text.
            """),
            ("human", "Email to prioritize:\n\nSender: {sender_name}\nEmail: {sender_email}\nSubject: {subject}\nBody: {body}\nHas Attachment: {has_attachment}\nTimestamp: {timestamp}")
        ])
    
    def assign_priority(self, email_data: Dict[str, Any]) -> EmailPriority:
        """Assign priority to an email"""
        try:
            prompt = self.prompt_template.format_messages(
                sender_name=email_data.get('sender_name', ''),
                sender_email=email_data.get('sender_email', ''),
                subject=email_data.get('subject', ''),
                body=email_data.get('body', ''),
                has_attachment=email_data.get('has_attachment', False),
                timestamp=email_data.get('timestamp', '')
            )
            
            response = self.invoke_with_retry(prompt)
            
            # Try to parse the response
            try:
                result = self.parser.parse(response.content)
                return EmailPriority(**result)
            except Exception as parse_error:
                # Fallback: try to extract JSON manually
                content = response.content.strip()
                if content.startswith('```json'):
                    content = content[7:-3]  # Remove ```json and ```
                elif content.startswith('```'):
                    content = content[3:-3]   # Remove ```
                
                # Try to fix common JSON issues
                content = content.replace('""', '"')  # Fix double quotes
                content = content.replace('\n', ' ')  # Remove newlines
                
                import json
                result = json.loads(content)
                return EmailPriority(**result)
                
        except Exception as e:
            # Ultimate fallback: return a default priority
            return EmailPriority(
                priority="Medium",
                urgency_score=5,
                reasoning=f"Unable to determine specific priority due to parsing error: {str(e)}"
            )

class ActionAgent(BaseAgent):
    """Agent responsible for recommending actions for emails"""
    
    def __init__(self):
        super().__init__()
        self.parser = JsonOutputParser(pydantic_object=EmailAction)
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are an expert email action recommendation specialist.
            Your task is to recommend the most appropriate action for an email.
            
            Available Actions:
            {actions}
            
            Analyze the email and provide a JSON response with this exact format:
            {{
                "action": "recommended_action_name",
                "reasoning": "clear_reasoning_for_recommendation",
                "draft_response": "optional_draft_response_if_applicable_or_null"
            }}
            
            Consider factors like:
            - Email content and purpose
            - Sender and relationship
            - Urgency and importance
            - Whether a response is needed
            - Whether the email requires follow-up
            
            IMPORTANT: Return valid JSON only, no additional text.
            """),
            ("human", "Email for action recommendation:\n\nSender: {sender_name}\nEmail: {sender_email}\nSubject: {subject}\nBody: {body}\nCategory: {category}\nPriority: {priority}")
        ])
    
    def recommend_action(self, email_data: Dict[str, Any], category: str, priority: str) -> EmailAction:
        """Recommend action for an email"""
        try:
            prompt = self.prompt_template.format_messages(
                actions=", ".join(Config.ACTION_TYPES),
                sender_name=email_data.get('sender_name', ''),
                sender_email=email_data.get('sender_email', ''),
                subject=email_data.get('subject', ''),
                body=email_data.get('body', ''),
                category=category,
                priority=priority
            )
            
            response = self.invoke_with_retry(prompt)
            
            # Try to parse the response
            try:
                result = self.parser.parse(response.content)
                return EmailAction(**result)
            except Exception as parse_error:
                # Fallback: try to extract JSON manually
                content = response.content.strip()
                if content.startswith('```json'):
                    content = content[7:-3]  # Remove ```json and ```
                elif content.startswith('```'):
                    content = content[3:-3]   # Remove ```
                
                # Try to fix common JSON issues
                content = content.replace('""', '"')  # Fix double quotes
                content = content.replace('\n', ' ')  # Remove newlines
                
                import json
                result = json.loads(content)
                return EmailAction(**result)
                
        except Exception as e:
            # Ultimate fallback: return a default action
            return EmailAction(
                action="Review Later",
                reasoning=f"Unable to determine specific action due to parsing error: {str(e)}",
                draft_response=None
            )

class SpamDetectionAgent(BaseAgent):
    """Agent responsible for detecting spam emails"""
    
    def __init__(self):
        super().__init__()
        
        # Safe domains whitelist
        self.safe_domains = {
            'linkedin.com', 'github.com', 'google.com', 'microsoft.com', 'apple.com',
            'amazon.com', 'slack.com', 'zoom.us', 'dropbox.com', 'atlassian.com',
            'salesforce.com', 'zendesk.com', 'jira.com', 'asana.com', 'chase.com'
        }
        
        # Spam patterns with weights
        self.spam_patterns = [
            (r'\b(viagra|cialis|pharmacy)\b', 2.0),
            (r'\$([\d,]+)', 1.0),
            (r'\b(lottery|winner|prize|won)\b', 1.5),
            (r'\b(urgent|action required|immediate)\b', 0.5),
            (r'\b(cryptocurrency|crypto|bitcoin|btc)\b', 1.0),
            (r'\b(survey|feedback|opinion)\b', 0.3),
            (r'[!]{2,}', 0.5),
            (r'\b(unsubscribe|opt[ -]?out)\b', 0.2),
            (r'\b(limited time|offer expires|act now)\b', 1.0),
            (r'\b(no obligation|risk[ -]?free)\b', 1.0),
        ]
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are an expert spam detector. Your task is to analyze emails and determine if they are spam or legitimate.
            
            Analyze the email and provide a JSON response with this exact format:
            {{
                "is_spam": true/false,
                "confidence": 0.85,
                "spam_score": 3.5,
                "reasoning": "clear_reasoning_for_spam_detection",
                "spam_indicators": ["indicator1", "indicator2"],
                "legitimate_indicators": ["indicator1", "indicator2"]
            }}
            
            Spam indicators to look for:
            - Unsolicited commercial content
            - Urgency/pressure tactics
            - Suspicious sender domains
            - Grammar/spelling errors
            - Too good to be true offers
            - Request for personal/financial information
            
            Legitimate indicators:
            - Professional communication
            - Known/safe sender domains
            - Context-appropriate content
            - Proper business communication
            - Personalized content
            
            IMPORTANT: Return valid JSON only, no additional text.
            """),
            ("human", "Email to analyze for spam:\n\nSender: {sender_name}\nEmail: {sender_email}\nSubject: {subject}\nBody: {body}\nHas Attachment: {has_attachment}\nTimestamp: {timestamp}")
        ])
    
    def check_basic_spam_score(self, email_data: Dict[str, Any]) -> float:
        """Calculate basic spam score using rules"""
        score = 0.0
        text = f"{email_data.get('sender_email', '')} {email_data.get('subject', '')} {email_data.get('body', '')}".lower()
        
        # Check spam patterns
        for pattern, weight in self.spam_patterns:
            if re.search(pattern, text):
                score += weight
        
        # Check for excessive capitalization
        subject = email_data.get('subject', '')
        if subject:
            caps_ratio = sum(1 for c in subject if c.isupper()) / (len(subject) + 1)
            if caps_ratio > 0.3:
                score += 1.0
        
        # Check safe domain
        sender_email = email_data.get('sender_email', '').lower()
        if any(domain in sender_email for domain in self.safe_domains):
            score -= 2.0  # Reduce score for safe domains
        
        return max(0.0, score)
    
    def detect_spam(self, email_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect if an email is spam"""
        try:
            # First check basic spam score
            basic_score = self.check_basic_spam_score(email_data)
            
            # If high confidence spam or safe, return basic result
            if basic_score > 3.0:
                return {
                    "is_spam": True,
                    "confidence": 0.9,
                    "spam_score": basic_score,
                    "reasoning": f"High spam score ({basic_score:.1f}) based on pattern matching",
                    "spam_indicators": ["High spam score", "Pattern matching"],
                    "legitimate_indicators": []
                }
            elif basic_score < 1.0:
                return {
                    "is_spam": False,
                    "confidence": 0.85,
                    "spam_score": basic_score,
                    "reasoning": f"Low spam score ({basic_score:.1f}) and safe patterns detected",
                    "spam_indicators": [],
                    "legitimate_indicators": ["Low spam score", "Safe patterns"]
                }
            
            # For uncertain cases, use AI analysis
            prompt = self.prompt_template.format_messages(
                sender_name=email_data.get('sender_name', ''),
                sender_email=email_data.get('sender_email', ''),
                subject=email_data.get('subject', ''),
                body=email_data.get('body', ''),
                has_attachment=email_data.get('has_attachment', False),
                timestamp=email_data.get('timestamp', '')
            )
            
            response = self.invoke_with_retry(prompt)
            
            # Try to parse the response
            try:
                import json
                content = response.content.strip()
                if content.startswith('```json'):
                    content = content[7:-3]
                elif content.startswith('```'):
                    content = content[3:-3]
                
                result = json.loads(content)
                
                # Combine basic score with AI analysis
                final_score = (basic_score + result.get('spam_score', 0)) / 2
                result['spam_score'] = final_score
                
                return result
                
            except Exception as parse_error:
                # Fallback to basic analysis
                return {
                    "is_spam": basic_score > 2.0,
                    "confidence": 0.6,
                    "spam_score": basic_score,
                    "reasoning": f"Basic analysis only due to parsing error: {str(parse_error)}",
                    "spam_indicators": ["Parsing error fallback"],
                    "legitimate_indicators": []
                }
                
        except Exception as e:
            return {
                "is_spam": False,
                "confidence": 0.3,
                "spam_score": 0.0,
                "reasoning": f"Error in spam detection: {str(e)}",
                "spam_indicators": [],
                "legitimate_indicators": ["Detection error - marked as safe"]
            }

class EmailRAGAgent(BaseAgent):
    """Agent responsible for RAG-based email chat and drafting"""
    
    def __init__(self):
        super().__init__()
        self.vector_store = None
        self.retriever = None
        self._initialize_rag()
    
    def _initialize_rag(self):
        """Initialize RAG system"""
        try:
            # Use Groq and HuggingFace instead of Ollama
            from langchain_groq import ChatGroq
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from langchain_chroma import Chroma
            from langchain_core.documents import Document
            import os
            import pandas as pd
            
            # Initialize embeddings
            self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            self.db_location = "./email_chroma_db"
            
            # Create vector store
            self.vector_store = Chroma(
                collection_name="email_database",
                persist_directory=self.db_location,
                embedding_function=self.embeddings
            )
            
            # Check if we need to populate
            add_documents = not os.path.exists(self.db_location)
            
            if add_documents:
                self._populate_vector_store()
            
            # Create retriever
            self.retriever = self.vector_store.as_retriever(
                search_kwargs={"k": 3}
            )
            
        except ImportError:
            print("⚠️ RAG dependencies not installed. Install: pip install langchain_groq langchain_huggingface langchain_chroma")
            self.retriever = None
        except Exception as e:
            print(f"⚠️ RAG initialization error: {str(e)}")
            self.retriever = None
    
    def _populate_vector_store(self):
        """Populate vector store with email data"""
        try:
            from langchain_core.documents import Document
            import pandas as pd
            
            df = pd.read_csv("dataset_emails - Sheet1.csv")
            documents = []
            ids = []
            
            for i, row in df.iterrows():
                content = f"""
                Subject: {row.get('Subject', '')}
                From: {row.get('From', '')}
                Date: {row.get('Date', '')}
                Body: {row.get('Body', '')}
                """
                
                document = Document(
                    page_content=content.strip(),
                    metadata={
                        "subject": row.get('Subject', ''),
                        "sender": row.get('From', ''),
                        "date": row.get('Date', ''),
                        "body": row.get('Body', ''),
                        "email_id": str(i)
                    },
                    id=str(i)
                )
                ids.append(str(i))
                documents.append(document)
            
            self.vector_store.add_documents(documents=documents, ids=ids)
            
        except Exception as e:
            print(f"Error populating vector store: {str(e)}")
    
    def chat_with_emails(self, question: str) -> str:
        """Chat with the email database"""
        if not self.retriever:
            return "🤖 RAG system not available. Please install required dependencies."
        
        try:
            # Get relevant emails
            relevant_emails = self.retriever.invoke(question)
            
            # Create context
            context = "\n\n".join([f"Email: {doc.page_content}" for doc in relevant_emails])
            
            # Create prompt
            prompt = f"""
            You are an expert email assistant with access to a database of previous emails.
            
            Here are some relevant emails:
            {context}
            
            Question: {question}
            
            Provide a helpful response based on the emails above. Reference specific emails when relevant.
            """
            
            # Get response
            response = self.invoke_with_retry([
                {"role": "system", "content": "You are an expert email assistant."},
                {"role": "user", "content": prompt}
            ])
            
            return response.content
            
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def draft_email(self, recipient: str, purpose: str, key_points: List[str], tone: str = "professional") -> str:
        """Draft an email using RAG context"""
        if not self.retriever:
            return "🤖 RAG system not available for email drafting."
        
        try:
            # Get context from similar emails
            context_query = f"email to {recipient} about {purpose}"
            relevant_emails = self.retriever.invoke(context_query)
            
            context = "\n\n".join([f"Email: {doc.page_content}" for doc in relevant_emails])
            
            prompt = f"""
            Draft a comprehensive email with these specifications:
            
            Recipient: {recipient}
            Purpose: {purpose}
            Key Points: {', '.join(key_points)}
            Tone: {tone}
            
            Here are some relevant emails for context:
            {context}
            
            Please draft a complete email with subject, greeting, body, and closing.
            """
            
            response = self.invoke_with_retry([
                {"role": "system", "content": "You are an expert email drafting assistant."},
                {"role": "user", "content": prompt}
            ])
            
            return response.content
            
        except Exception as e:
            return f"❌ Error drafting email: {str(e)}"
    
    def search_emails(self, query: str) -> List[Dict]:
        """Search for relevant emails"""
        if not self.retriever:
            return []
        
        try:
            results = self.retriever.invoke(query)
            
            formatted_results = []
            for doc in results:
                formatted_results.append({
                    "subject": doc.metadata.get("subject", ""),
                    "sender": doc.metadata.get("sender", ""),
                    "date": doc.metadata.get("date", ""),
                    "content": doc.page_content,
                    "email_id": doc.metadata.get("email_id", "")
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"Error searching emails: {str(e)}")
            return []

class SentimentAgent(BaseAgent):
    """Agent responsible for sentiment analysis"""
    
    def __init__(self):
        super().__init__()
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are an expert sentiment analysis specialist.
            Your task is to analyze the sentiment of an email and classify it as:
            - positive: Happy, satisfied, appreciative, excited
            - negative: Angry, frustrated, disappointed, concerned
            - neutral: Informational, factual, neutral tone
            
            Analyze the email content and provide just the sentiment classification.
            Consider the overall tone, language used, and emotional indicators.
            """),
            ("human", "Email for sentiment analysis:\n\nSubject: {subject}\nBody: {body}")
        ])
    
    def analyze_sentiment(self, email_data: Dict[str, Any]) -> str:
        """Analyze sentiment of an email"""
        prompt = self.prompt_template.format_messages(
            subject=email_data.get('subject', ''),
            body=email_data.get('body', '')
        )
        
        response = self.invoke_with_retry(prompt)
        sentiment = response.content.strip().lower()
        
        # Normalize sentiment
        if 'positive' in sentiment:
            return 'positive'
        elif 'negative' in sentiment:
            return 'negative'
        else:
            return 'neutral'

# ============================================================================
# LANGGRAPH STATE SCHEMA - Defined before use
# ============================================================================

class EmailProcessingState(TypedDict):
    """State schema for the email processing workflow"""
    # Original email data
    email_id: str
    sender_email: str
    sender_name: str
    subject: str
    body: str
    timestamp: str
    has_attachment: bool
    thread_id: str

    # Processing state
    current_step: str
    processing_complete: bool

    # Analysis results
    categories: List[Dict]
    priority: Dict
    suggested_actions: List[Dict]
    summary: Dict
    thread_context: Optional[str]

    # Agent communication
    messages: Annotated[Sequence[BaseMessage], add_messages]
    tool_calls: List[Dict]
    tool_responses: List[Dict]

    # Error handling
    error_count: int
    last_error: Optional[str]

    # Final output
    final_organization: Optional[Dict]

    # New field: email_data
    email_data: Dict[str, Any]

# ============================================================================
# LANGGRAPH WORKFLOW CREATION - Defined before EmailOrchestrator
# ============================================================================

# Initialize LLM for agent reasoning
llm = ChatGroq(
    api_key=Config.GROQ_API_KEY,
    model=Config.DEFAULT_MODEL,
    temperature=Config.TEMPERATURE,
    max_tokens=Config.MAX_TOKENS
)

# Create prompt template for the agent
system_message = """You are an expert email organization assistant. Your job is to analyze incoming emails and provide structured organization recommendations.

You have access to specialized tools for:
- Categorizing emails into appropriate categories
- Assigning priority levels based on urgency and importance
- Suggesting appropriate actions for each email
- Creating concise summaries of email content

For each tool call, provide clear and specific arguments based on the email content."""

# Agent will be created after tool functions are defined
agent = None

def create_agent_instance():
    """Create the agent instance after tools are defined"""
    global agent
    if agent is None:
        from langchain.agents import create_agent
        agent = create_agent(
            model=llm,
            tools=[categorize_email, prioritize_email, suggest_actions, summarize_email],
            system_prompt=system_message
        )
    return agent

def create_email_workflow() -> StateGraph:
    """Create and compile the LangGraph workflow"""
    
    # Create the graph
    workflow = StateGraph(EmailProcessingState)
    
    # Add nodes to the graph
    workflow.add_node("email_agent", email_processing_agent)
    workflow.add_node("tool_executor", tool_execution_node)
    workflow.add_node("analysis_integrator", integrate_analysis_node)
    workflow.add_node("error_handler", error_handling_node)
    
    # Set entry point
    workflow.set_entry_point("email_agent")
    
    # Add edges - go directly to tool_executor instead of tools node
    workflow.add_edge("email_agent", "tool_executor")
    workflow.add_edge("tool_executor", "analysis_integrator")
    workflow.add_conditional_edges(
        "analysis_integrator",
        lambda state: "end" if state.get("processing_complete") else "error_handler",
        {
            "end": END,
            "error_handler": "error_handler"
        }
    )
    workflow.add_edge("error_handler", "email_agent")
    
    # Compile graph
    app = workflow.compile()
    
    return app

class EmailOrchestrator:
    """Orchestrates the email processing workflow"""
    
    def __init__(self, use_langgraph: bool = True):
        self.use_langgraph = use_langgraph
        self.categorization_agent = CategorizationAgent()
        self.priority_agent = PriorityAgent()
        self.action_agent = ActionAgent()
        self.sentiment_agent = SentimentAgent()
        self.spam_detection_agent = SpamDetectionAgent()
        self.rag_agent = EmailRAGAgent()  # Add RAG agent
        
        if use_langgraph:
            self.workflow = self._create_langgraph_workflow()
        else:
            self.workflow = None
    
    def process_email(self, email_data: Dict[str, Any]) -> EmailAnalysis:
        """Process a single email through all agents"""
        try:
            # Step 1: Spam Detection (first priority)
            spam_result = self.spam_detection_agent.detect_spam(email_data)
            
            # If it's high confidence spam, skip other analysis
            if spam_result.get('is_spam', False) and spam_result.get('confidence', 0) > 0.8:
                return EmailAnalysis(
                    email_id=email_data.get('email_id', 0),
                    category=EmailCategory(
                        category="Spam",
                        confidence=spam_result.get('confidence', 0.8),
                        reasoning="High confidence spam detected",
                        keywords=["spam"],
                        coarse_genre="spam",
                        content_type="spam",
                        primary_topic="spam",
                        emotional_tone="neutral"
                    ),
                    priority=EmailPriority(
                        priority="Low",
                        urgency_score=1,
                        reasoning="Spam emails are low priority"
                    ),
                    action=EmailAction(
                        action="Delete/Mark as Spam",
                        reasoning="Spam should be deleted or marked as spam",
                        draft_response=None
                    ),
                    sentiment="negative",
                    spam_detection=spam_result
                )
            
            # Step 2: Categorization (if not spam)
            category = self.categorization_agent.categorize(email_data)
            
            # Step 3: Priority Assignment
            priority = self.priority_agent.assign_priority(email_data)
            
            # Step 4: Action Recommendation
            action = self.action_agent.recommend_action(email_data, category.category, priority.priority)
            
            # Step 5: Sentiment Analysis
            sentiment = self.sentiment_agent.analyze_sentiment(email_data)
            
            return EmailAnalysis(
                email_id=email_data.get('email_id', 0),
                category=category,
                priority=priority,
                action=action,
                sentiment=sentiment,
                spam_detection=spam_result
            )
            
        except Exception as e:
            # Ultimate fallback: return a default category
            return EmailAnalysis(
                email_id=email_data.get('email_id', 0),
                category=EmailCategory(
                    category="Company Business Strategy",
                    confidence=0.5,
                    reasoning=f"Error during categorization: {str(e)}",
                    keywords=["error"],
                    coarse_genre="Company Business Strategy",
                    content_type="Business Letters Documents",
                    primary_topic="Internal Projects Progress Strategy",
                    emotional_tone="neutral"
                ),
                priority=EmailPriority(
                    priority="Medium",
                    urgency_score=5,
                    reasoning=f"Error during priority assignment: {str(e)}"
                ),
                action=EmailAction(
                    action="Manual Review Required",
                    reasoning=f"Error during action recommendation: {str(e)}",
                    draft_response=None
                ),
                sentiment="neutral",
                spam_detection={"is_spam": False, "confidence": 0.5, "error": str(e)}
            )
    
    def chat_with_emails(self, question: str) -> str:
        """Chat with the email database using RAG"""
        return self.rag_agent.chat_with_emails(question)
    
    def draft_email(self, recipient: str, purpose: str, key_points: List[str], tone: str = "professional") -> str:
        """Draft an email using RAG context"""
        return self.rag_agent.draft_email(recipient, purpose, key_points, tone)
    
    def search_emails(self, query: str) -> List[Dict]:
        """Search for relevant emails"""
        return self.rag_agent.search_emails(query)
    
    def _create_langgraph_workflow(self):
        """Create LangGraph workflow (placeholder for future implementation)"""
        # This would integrate with LangGraph in the future
        return None
    
    def process_batch(self, emails: List[Dict[str, Any]]) -> List[EmailAnalysis]:
        """Process multiple emails"""
        results = []
        for email in emails:
            try:
                analysis = self.process_email(email)
                results.append(analysis)
            except Exception as e:
                print(f"Error processing email {email.get('email_id', 'unknown')}: {str(e)}")
                continue
        return results

# Import pandas for the workflow
import pandas as pd
import json

# ============================================================================
# LANGGRAPH WORKFLOW IMPLEMENTATION
# ============================================================================

# LangGraph Tools
@tool
def categorize_email(email_content: str, sender_info: Optional[str] = None, subject: Optional[str] = None) -> str:
    """Categorize an email into appropriate categories based on content, sender, and subject"""
    import json
    categories = []
    
    if "meeting" in email_content.lower() or "schedule" in email_content.lower():
        categories.append({
            "category": "Business Meeting Request",
            "confidence": 0.85,
            "keywords": ["meeting", "schedule", "availability"]
        })
    
    if "alert" in email_content.lower() or "error" in email_content.lower() or "critical" in email_content.lower():
        categories.append({
            "category": "System Alert Notification",
            "confidence": 0.92,
            "keywords": ["alert", "error", "critical", "system"]
        })
    
    if "invoice" in email_content.lower() or "payment" in email_content.lower() or "billing" in email_content.lower():
        categories.append({
            "category": "Financial Document Invoice",
            "confidence": 0.88,
            "keywords": ["invoice", "payment", "billing", "due"]
        })
    
    if "project" in email_content.lower() or "update" in email_content.lower():
        categories.append({
            "category": "Internal Project Update",
            "confidence": 0.80,
            "keywords": ["project", "update", "progress"]
        })
    
    if "policy" in email_content.lower() or "procedure" in email_content.lower():
        categories.append({
            "category": "Company Policy Announcement",
            "confidence": 0.83,
            "keywords": ["policy", "procedure", "guidelines"]
        })
    
    if "legal" in email_content.lower() or "contract" in email_content.lower():
        categories.append({
            "category": "Legal Document Review",
            "confidence": 0.87,
            "keywords": ["legal", "contract", "agreement"]
        })
    
    if not categories:
        categories.append({
            "category": "General Business Communication",
            "confidence": 0.75,
            "keywords": ["general", "communication", "information"]
        })
    
    return json.dumps({"categories": categories})

@tool
def prioritize_email(email_content: str, categories: List[str], sender_info: Optional[str] = None) -> str:
    """Assign priority level to an email based on content and categories"""
    import json
    urgency_score = 5  # Base score
    
    # Increase urgency for specific keywords
    urgent_keywords = ["urgent", "asap", "immediately", "deadline", "critical", "emergency"]
    for keyword in urgent_keywords:
        if keyword in email_content.lower():
            urgency_score += 1
    
    # Adjust based on categories
    if "system_alert" in categories:
        urgency_score += 3
    elif "invoice" in categories:
        urgency_score += 1
    
    # Determine priority level
    if urgency_score >= 8:
        priority = "High"
    elif urgency_score >= 5:
        priority = "Medium"
    else:
        priority = "Low"
    
    return json.dumps({
        "priority": priority,
        "urgency_score": min(10, urgency_score),
        "reasoning": f"Assigned based on content analysis and categories: {categories}"
    })

@tool
def suggest_actions(email_content: str, categories: List[str], priority: str) -> str:
    """Suggest appropriate actions for an email based on its content and priority"""
    import json
    actions = []
    
    # Base actions on category
    if "meeting_request" in categories:
        actions.append({
            "action": "Schedule Meeting",
            "reasoning": "Email contains meeting request that needs scheduling",
            "priority": "High"
        })
    elif "system_alert" in categories:
        actions.append({
            "action": "Reply Immediately",
            "reasoning": "System alert requires immediate attention",
            "priority": "High"
        })
    elif "invoice" in categories:
        actions.append({
            "action": "Forward to Finance",
            "reasoning": "Invoice should be forwarded to finance department",
            "priority": "Medium"
        })
    else:
        if priority == "High":
            actions.append({
                "action": "Reply Immediately",
                "reasoning": "High priority email requires prompt response",
                "priority": "High"
            })
        else:
            actions.append({
                "action": "Review Later",
                "reasoning": "Can be reviewed when time permits",
                "priority": "Low"
            })
    
    return json.dumps({"suggested_actions": actions})

@tool
def summarize_email(email_content: str) -> str:
    """Create a concise summary of the email content"""
    import json
    # Simple extractive summarization
    sentences = email_content.split('.')
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) <= 2:
        summary = email_content[:100]
    else:
        summary = '. '.join(sentences[:2]) + '.'
        if len(summary) > 100:
            summary = summary[:100] + '...'
    
    return json.dumps({
        "summary": summary,
        "original_length": len(email_content),
        "summary_length": len(summary)
    })

# LangGraph Nodes
def email_processing_agent(state: EmailProcessingState) -> EmailProcessingState:
    """Main agent that coordinates the email processing workflow"""
    new_state = state.copy()

    try:
        # Simple approach: directly call tools based on email content
        email_content = f"Subject: {state['subject']}\n\nBody: {state['body']}"
        
        # Prepare tool calls directly
        tool_calls = []
        
        # Always categorize
        tool_calls.append({
            "name": "categorize_email",
            "arguments": {
                "email_content": email_content,
                "sender_info": state['sender_email'],
                "subject": state['subject']
            }
        })
        
        # Always prioritize
        tool_calls.append({
            "name": "prioritize_email",
            "arguments": {
                "email_content": email_content,
                "categories": ["general_communication"],  # Will be updated after categorization
                "sender_info": state['sender_email']
            }
        })
        
        # Always summarize
        tool_calls.append({
            "name": "summarize_email",
            "arguments": {
                "email_content": email_content
            }
        })
        
        new_state["tool_calls"] = tool_calls
        new_state["current_step"] = "tools_executed"

    except Exception as e:
        new_state["error_count"] += 1
        new_state["last_error"] = str(e)
        new_state["current_step"] = "error"

    return new_state

def tool_execution_node(state: EmailProcessingState) -> EmailProcessingState:
    """Execute the tools called by the agent"""
    try:
        import json
        tool_responses = []
        
        for tool_call in state.get("tool_calls", []):
            tool_name = tool_call["name"]
            arguments = tool_call["arguments"]
            
            # Call the underlying function directly
            if tool_name == "categorize_email":
                # Get the underlying function from the tool
                result = categorize_email.func(**arguments)
                tool_responses.append({"tool": tool_name, "result": json.loads(result)})
                
            elif tool_name == "prioritize_email":
                result = prioritize_email.func(**arguments)
                tool_responses.append({"tool": tool_name, "result": json.loads(result)})
                
            elif tool_name == "suggest_actions":
                result = suggest_actions.func(**arguments)
                tool_responses.append({"tool": tool_name, "result": json.loads(result)})
                
            elif tool_name == "summarize_email":
                result = summarize_email.func(**arguments)
                tool_responses.append({"tool": tool_name, "result": json.loads(result)})
        
        state["tool_responses"] = tool_responses
        state["current_step"] = "tools_completed"
        
        return state
        
    except Exception as e:
        state["error_count"] += 1
        state["last_error"] = str(e)
        state["current_step"] = "error"
        return state

def integrate_analysis_node(state: EmailProcessingState) -> EmailProcessingState:
    """Integrate all analysis results into final organization"""
    try:
        import json
        # Extract results from tool responses
        categories = []
        priority = {}
        summary = {}
        suggested_actions = []
        
        for response in state.get("tool_responses", []):
            tool_name = response["tool"]
            result = response["result"]
            
            if tool_name == "categorize_email":
                categories = result.get("categories", [])
            elif tool_name == "prioritize_email":
                priority = result
            elif tool_name == "summarize_email":
                summary = result
        
        # Generate action suggestions based on analysis
        if categories and priority:
            category_names = [cat.get("category", "general") for cat in categories]
            priority_level = priority.get("priority", "Medium")
            
            action_result = suggest_actions(
                email_content=f"Subject: {state['subject']}\n\nBody: {state['body']}",
                categories=category_names,
                priority=priority_level
            )
            suggested_actions = json.loads(action_result).get("suggested_actions", [])
        
        # Create final organization
        final_organization = {
            "email_id": state["email_id"],
            "categories": categories,
            "priority": priority,
            "summary": summary,
            "suggested_actions": suggested_actions,
            "processing_timestamp": str(pd.Timestamp.now()),
            "confidence_score": calculate_confidence_score(categories, priority),
            "status": "success"
        }
        
        state["final_organization"] = final_organization
        state["processing_complete"] = True
        state["current_step"] = "completed"
        
        return state
        
    except Exception as e:
        state["error_count"] += 1
        state["last_error"] = str(e)
        state["current_step"] = "error"
        return state

def error_handling_node(state: EmailProcessingState) -> EmailProcessingState:
    """Handle errors and retry logic"""
    if state["error_count"] < 3:
        # Retry the processing
        state["current_step"] = "retry"
        state["last_error"] = None
    else:
        # Max retries reached, mark as failed
        state["processing_complete"] = True
        state["current_step"] = "failed"
        state["final_organization"] = {
            "email_id": state["email_id"],
            "error": state["last_error"],
            "status": "failed",
            "confidence_score": 0
        }
    
    return state

def calculate_confidence_score(categories: List[Dict], priority: Dict) -> float:
    """Calculate overall confidence score"""
    if not categories:
        return 0.0
    
    category_confidence = sum(cat.get("confidence", 0) for cat in categories) / len(categories)
    priority_weight = 0.8 if priority.get("priority") == "High" else 0.6
    
    return (category_confidence + priority_weight) / 2 * 100

# ============================================================================
# LANGGRAPH WORKFLOW CREATION - Moved to end after tool definitions
# ============================================================================

# Initialize LLM for agent reasoning
llm = ChatGroq(
    api_key=Config.GROQ_API_KEY,
    model=Config.DEFAULT_MODEL,
    temperature=Config.TEMPERATURE,
    max_tokens=Config.MAX_TOKENS
)

# Create prompt template for the agent
system_message = """You are an expert email organization assistant. Your job is to analyze incoming emails and provide structured organization recommendations.

You have access to specialized tools for:
- Categorizing emails into appropriate categories
- Assigning priority levels based on urgency and importance
- Suggesting appropriate actions for each email
- Creating concise summaries of email content

For each tool call, provide clear and specific arguments based on the email content."""

# Agent will be created after tool functions are defined
agent = None

def create_agent_instance():
    """Create the agent instance after tools are defined"""
    global agent
    if agent is None:
        from langchain.agents import create_agent
        agent = create_agent(
            model=llm,
            tools=[categorize_email, prioritize_email, suggest_actions, summarize_email],
            system_prompt=system_message
        )
    return agent

def create_email_workflow() -> StateGraph:
    """Create and compile the LangGraph workflow"""
    
    # Create the graph
    workflow = StateGraph(EmailProcessingState)
    
    # Add nodes to the graph
    workflow.add_node("email_agent", email_processing_agent)
    workflow.add_node("tool_executor", tool_execution_node)
    workflow.add_node("analysis_integrator", integrate_analysis_node)
    workflow.add_node("error_handler", error_handling_node)
    
    # Set entry point
    workflow.set_entry_point("email_agent")
    
    # Add edges - go directly to tool_executor instead of tools node
    workflow.add_edge("email_agent", "tool_executor")
    workflow.add_edge("tool_executor", "analysis_integrator")
    workflow.add_conditional_edges(
        "analysis_integrator",
        lambda state: "end" if state.get("processing_complete") else "error_handler",
        {
            "end": END,
            "error_handler": "error_handler"
        }
    )
    workflow.add_edge("error_handler", "email_agent")
    
    # Compile graph
    app = workflow.compile()
    
    return app
