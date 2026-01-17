"""
Intelligent Email Reply System using CrewAI-based Email Reply Agents
Enhanced version of the YouTube email reply system with improved categorization and user-friendly outputs
"""

import os
from langchain_core.agents import AgentFinish
from langchain_community.tools import DuckDuckGoSearchRun
from datetime import datetime
from random import randint
from typing import Union, List, Tuple, Dict, Optional, Any
from crewai import Crew, Process, Task, Agent
from crewai.tools import BaseTool
from config import Config
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_groq import ChatGroq

class WebSearchInput(BaseModel):
    query: str = Field(..., description="Search query")


class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = "Search the web for information about a given query using DuckDuckGo."
    args_schema: type[BaseModel] = WebSearchInput

    def _run(self, query: str) -> str:
        search_tool = DuckDuckGoSearchRun()
        try:
            result = search_tool.run(query)
            return str(result)
        except Exception as e:
            return f"Search error: {str(e)}"

# ============================================================================
# PYDANTIC MODELS FOR STRUCTURED OUTPUTS
# ============================================================================

class EmailCategory(BaseModel):
    """Model for email categorization"""
    category: str = Field(description="Email category: Price Enquiry, Customer Complaint, Product Enquiry, Customer Feedback, or Off Topic")
    confidence: float = Field(description="Confidence score 0-1")
    reasoning: str = Field(description="Reasoning for categorization")

class EmailReply(BaseModel):
    """Model for email reply"""
    reply_content: str = Field(description="The complete email reply")
    tone: str = Field(description="Tone of the reply: professional, friendly, empathetic, etc.")
    key_points: List[str] = Field(description="Key points addressed in the reply")
    research_used: List[str] = Field(description="Research information used in the reply")

class ReplyWorkflow(BaseModel):
    """Complete workflow results"""
    original_email: Optional[str] = Field(default="", description="Original email content")
    category: Optional[EmailCategory] = Field(default=None, description="Email categorization result")
    research: Optional[str] = Field(default="", description="Research findings")
    reply: Optional[EmailReply] = Field(default=None, description="Generated email reply")
    processing_time: Optional[float] = Field(default=0.0, description="Time taken to process")
    agent_performance: Optional[Dict[str, Any]] = Field(default={}, description="Performance metrics")
    
    def create_workflow(self, agents: List[Agent], tasks: List[Task]) -> Crew:
        """Create and return a CrewAI workflow"""
        # Assign agents to tasks
        if len(agents) >= 1 and len(tasks) >= 1:
            tasks[0].agent = agents[0]  # categorizer
        if len(agents) >= 2 and len(tasks) >= 2:
            tasks[1].agent = agents[1]  # researcher  
        if len(agents) >= 3 and len(tasks) >= 3:
            tasks[2].agent = agents[2]  # writer
            
        return Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=True
        )

# ============================================================================
# AGENT CLASSES
# ============================================================================

class EmailReplyAgents:
    """Factory class for creating email reply agents"""
    
    def __init__(self):
        self.llm = "groq/llama-3.1-8b-instant"
        self.agent_logs = []
    
    def make_categorizer_agent(self) -> Agent:
        """Create email categorizer agent"""
        return Agent(
            role='Email Categorizer Agent',
            goal="""Categorize customer emails into one of the following categories:
            
            - Price Enquiry: Customer asking for pricing information
            - Customer Complaint: Customer complaining about products/services
            - Product Enquiry: Customer asking about product features/benefits (not pricing)
            - Customer Feedback: Customer providing feedback about products/services
            - Off Topic: Email doesn't relate to any other category
            
            Analyze the email content, tone, and intent to determine the most appropriate category.
            
            IMPORTANT: Return the category name exactly as shown above (with spaces and capitalization), not codes.""",
            backstory="""You are an expert at understanding customer intent and categorizing emails accurately. 
            You have years of experience in customer service and can quickly identify what a customer needs.""",
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            max_iter=3,
            memory=True,
            step_callback=lambda x: self._log_agent_output(x, "Email Categorizer Agent"),
        )
    
    def make_researcher_agent(self) -> Agent:
        """Create researcher agent"""
        return Agent(
            role='Info Researcher Agent',
            goal="""Research relevant information needed to write helpful email responses.
            
            Focus on finding:
            - Product information and specifications
            - Pricing details (if available)
            - Company policies and procedures
            - Best practices for customer communication
            - Relevant industry standards
            
            Use web search when necessary to find accurate, up-to-date information.""",
            backstory="""You are a skilled researcher with expertise in finding accurate information 
            quickly. You know how to evaluate sources for credibility and extract the most 
            relevant details for crafting professional email responses.""",
            tools=[WebSearchTool()],
            verbose=True,
            max_iter=3,
            allow_delegation=False,
            memory=True,
            llm=self.llm,
            step_callback=lambda x: self._log_agent_output(x, "Info Researcher Agent"),
        )
    
    def make_email_writer_agent(self) -> Agent:
        """Create email writer agent"""
        return Agent(
            role='Email Writer Agent',
            goal="""Write professional, helpful email responses to customers based on:
            
            1. Original email content
            2. Email category from categorizer
            3. Research findings from researcher
            
            Response guidelines:
            - Price Enquiry: If exact pricing is not explicitly known from the email or research, do NOT invent numbers. Ask clarifying questions (e.g., seats/users, SLA, features) and offer to share a quote or schedule a call.
            - Customer Complaint: Acknowledge issues, apologize sincerely, offer solutions
            - Product Enquiry: Explain features/benefits clearly and concisely
            - Customer Feedback: Thank customer, acknowledge feedback, mention improvements
            - Off Topic: Ask clarifying questions to better understand needs
            
            Always:
            - Be professional yet friendly
            - Address all customer concerns
            - Use research information when available
            - Never make up information not provided
            - Sign off appropriately from Customer Support
            - Keep responses concise but comprehensive""",
            backstory="""You are an expert customer service representative with excellent writing skills. 
            You can craft responses that are empathetic, informative, and professional while addressing 
            customer needs effectively.""",
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            max_iter=3,
            memory=True,
            step_callback=lambda x: self._log_agent_output(x, "Email Writer Agent"),
        )
    
    def _log_agent_output(self, agent_output: Union[str, List, AgentFinish], agent_name: str):
        """Log agent outputs for debugging and monitoring"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent_name,
            "output_type": type(agent_output).__name__,
            "output": str(agent_output)[:500]  # Limit output size
        }
        self.agent_logs.append(log_entry)

# ============================================================================
# TASK CLASSES
# ============================================================================

class EmailReplyTasks:
    """Factory class for creating email reply tasks"""
    
    def __init__(self):
        pass
    
    def categorize_email(self, email_content: str) -> Task:
        """Create email categorization task"""
        return Task(
            description=f"""Analyze the following email and categorize it into one of these categories:
            
            Categories:
            - Price Enquiry: Customer asking for pricing information
            - Customer Complaint: Customer complaining about products/services  
            - Product Enquiry: Customer asking about product features/benefits (not pricing)
            - Customer Feedback: Customer providing feedback about products/services
            - Off Topic: Email doesn't relate to any other category
            
            EMAIL CONTENT:
            {email_content}
            
            Provide only the category name as output. Use the exact category names as shown above (with spaces and capitalization).""",
            expected_output="A single category name from: Price Enquiry, Customer Complaint, Product Enquiry, Customer Feedback, Off Topic",
            agent=None,  # Will be set when crew is created
            output_file="email_category.txt"
        )
    
    def research_info(self, email_content: str) -> Task:
        """Create research task"""
        return Task(
            description=f"""Based on the email content and category, research relevant information 
            needed to write a helpful response.
            
            EMAIL CONTENT:
            {email_content}
            
            Search for:
            - Pricing information (if Price Enquiry)
            - Product details/features (if Product Enquiry)
            - Support policies/solutions (if Customer Complaint)
            - Response strategies (if Customer Feedback)
            - Relevant context (if Off Topic)
            
            Only provide key information needed for the email writer. Do not write the email itself.""",
            expected_output="Bullet points of useful information or 'NO SEARCH NEEDED'/'NO USEFUL RESEARCH FOUND'",
            agent=None,  # Will be set when crew is created
            context=[],  # Will depend on categorization task
            output_file="research_info.txt"
        )
    
    def draft_reply(self, email_content: str) -> Task:
        """Create email drafting task"""
        return Task(
            description=f"""Write a professional email response based on:
            
            1. Original email content
            2. Email category from categorizer
            3. Research findings from researcher
            
            EMAIL CONTENT:
            {email_content}
            
            Guidelines:
            - Address all customer concerns
            - Use research information when available
            - Be professional yet friendly
            - Never make up information not provided
            - Sign off from Customer Support
            - Keep response concise but comprehensive
            
            Write the complete email response.""",
            expected_output="A complete, professional email response addressing the customer's needs",
            agent=None,  # Will be set when crew is created
            context=[],  # Will depend on categorization and research tasks
            output_file="draft_email.txt"
        )

# ============================================================================
# MAIN EMAIL REPLY ORCHESTRATOR
# ============================================================================

class EmailReplyOrchestrator:
    """Main orchestrator for the email reply system"""
    
    def __init__(self):
        self.agents = EmailReplyAgents()
        self.tasks = EmailReplyTasks()
        self.crew = None
        self.processing_history = []
    
    def process_email_reply(self, email_content: str) -> ReplyWorkflow:
        """Process an email through the complete reply workflow"""
        start_time = datetime.now()
        
        try:
            # Create agents
            categorizer = self.agents.make_categorizer_agent()
            researcher = self.agents.make_researcher_agent()
            writer = self.agents.make_email_writer_agent()
            
            # Create tasks
            categorize_task = self.tasks.categorize_email(email_content)
            research_task = self.tasks.research_info(email_content)
            reply_task = self.tasks.draft_reply(email_content)
            
            # Set task agents and context
            categorize_task.agent = categorizer
            research_task.agent = researcher
            reply_task.agent = writer
            
            research_task.context = [categorize_task]
            reply_task.context = [categorize_task, research_task]
            
            # Create and run crew
            crew = Crew(
                agents=[categorizer, researcher, writer],
                tasks=[categorize_task, research_task, reply_task],
                verbose=True,
                process=Process.sequential,
                full_output=True,
                share_crew=False,
                step_callback=lambda x: self.agents._log_agent_output(x, "Master Crew")
            )
            
            # Execute the workflow
            result = crew.kickoff()
            
            # Extract results
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Parse results (simplified - in production would parse actual outputs)
            category_result = self._extract_category_result(categorize_task)
            research_result = self._extract_research_result(research_task)
            reply_result = self._extract_reply_result(reply_task)
            
            # Create workflow result
            workflow_result = ReplyWorkflow(
                original_email=email_content,
                category=category_result,
                research=research_result,
                reply=reply_result,
                processing_time=processing_time,
                agent_performance={
                    "usage_metrics": getattr(crew, 'usage_metrics', {}),
                    "agent_logs": self.agents.agent_logs[-10:],  # Last 10 log entries
                    "tasks_completed": len(result.tasks) if hasattr(result, 'tasks') else 3
                }
            )
            
            # Store in history
            self.processing_history.append(workflow_result)
            
            return workflow_result
            
        except Exception as e:
            # Create error result
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ReplyWorkflow(
                original_email=email_content,
                category=EmailCategory(
                    category="error",
                    confidence=0.0,
                    reasoning=f"Error during categorization: {str(e)}"
                ),
                research=f"Error during research: {str(e)}",
                reply=EmailReply(
                    reply_content="We apologize, but we encountered an error processing your email. Please try again later.",
                    tone="professional",
                    key_points=["Error occurred"],
                    research_used=[]
                ),
                processing_time=processing_time,
                agent_performance={"error": str(e)}
            )
    
    def _extract_category_result(self, task) -> EmailCategory:
        """Extract category result from task"""
        try:
            if hasattr(task, 'output') and task.output:
                output = task.output.strip()
                # Map various possible outputs to standard category names
                category_mapping = {
                    'price_enquiry': 'Price Enquiry',
                    'price enquiry': 'Price Enquiry',
                    'price_inquiry': 'Price Enquiry',
                    'pricing_enquiry': 'Price Enquiry',
                    'customer_complaint': 'Customer Complaint',
                    'customer complaint': 'Customer Complaint',
                    'product_enquiry': 'Product Enquiry',
                    'product enquiry': 'Product Enquiry',
                    'product_inquiry': 'Product Enquiry',
                    'customer_feedback': 'Customer Feedback',
                    'customer feedback': 'Customer Feedback',
                    'off_topic': 'Off Topic',
                    'off topic': 'Off Topic'
                }
                
                # Normalize output
                normalized_output = output.lower().replace('_', ' ').replace('-', ' ')
                
                # Check if output matches any of our valid categories
                valid_categories = ['price enquiry', 'customer complaint', 'product enquiry', 'customer feedback', 'off topic']
                
                if normalized_output in valid_categories:
                    # Convert to proper format
                    category_name = category_mapping.get(output, output.title())
                    return EmailCategory(
                        category=category_name,
                        confidence=0.9,
                        reasoning=f"Agent categorized as {category_name}"
                    )
                else:
                    # Try to find closest match
                    for valid_cat in valid_categories:
                        if valid_cat in normalized_output or normalized_output in valid_cat:
                            category_name = category_mapping.get(output, valid_cat.title())
                            return EmailCategory(
                                category=category_name,
                                confidence=0.8,
                                reasoning=f"Agent categorized as {category_name} (normalized from '{output}')"
                            )
        except Exception as e:
            pass
        
        return EmailCategory(
            category="Unknown",
            confidence=0.5,
            reasoning="Unable to extract category from agent output"
        )
    
    def _extract_research_result(self, task) -> Optional[str]:
        """Extract research result from task"""
        try:
            if hasattr(task, 'output') and task.output:
                return task.output.strip()
        except Exception:
            pass
        return None
    
    def _extract_reply_result(self, task) -> EmailReply:
        """Extract reply result from task"""
        try:
            if hasattr(task, 'output') and task.output:
                reply_content = task.output.strip()
                
                # Extract key points (simplified)
                key_points = []
                if "pricing" in reply_content.lower():
                    key_points.append("Pricing information provided")
                if "feature" in reply_content.lower():
                    key_points.append("Product features explained")
                if "apologize" in reply_content.lower() or "sorry" in reply_content.lower():
                    key_points.append("Customer concern acknowledged")
                
                return EmailReply(
                    reply_content=reply_content,
                    tone="professional",
                    key_points=key_points,
                    research_used=[]
                )
        except Exception as e:
            pass
        
        return EmailReply(
            reply_content="Thank you for your email. We will respond to your inquiry shortly.",
            tone="professional",
            key_points=["Standard response"],
            research_used=[]
        )
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get statistics about processed emails"""
        if not self.processing_history:
            return {"total_processed": 0}
        
        total_processed = len(self.processing_history)
        avg_time = sum(h.processing_time for h in self.processing_history) / total_processed
        
        category_counts = {}
        for history in self.processing_history:
            cat = history.category.category
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        return {
            "total_processed": total_processed,
            "average_processing_time": avg_time,
            "category_distribution": category_counts,
            "last_processed": self.processing_history[-1].original_email[:100] + "..." if self.processing_history else None
        }

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def process_email_with_retry(orchestrator: EmailReplyOrchestrator, email_content: str) -> ReplyWorkflow:
    """Process email with retry logic"""
    return orchestrator.process_email_reply(email_content)

def validate_email_content(email_content: str) -> bool:
    """Validate email content before processing"""
    if not email_content or not email_content.strip():
        return False
    
    # Basic validation - should have some content
    if len(email_content.strip()) < 10:
        return False
    
    return True

def format_reply_for_display(reply_result: ReplyWorkflow) -> Dict[str, Any]:
    """Format reply result for display in UI"""
    return {
        "category": reply_result.category.category,
        "confidence": reply_result.category.confidence,
        "reasoning": reply_result.category.reasoning,
        "research": reply_result.research,
        "reply": reply_result.reply.reply_content,
        "tone": reply_result.reply.tone,
        "key_points": reply_result.reply.key_points,
        "processing_time": reply_result.processing_time,
        "performance": reply_result.agent_performance
    }
