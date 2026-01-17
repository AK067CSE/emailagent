"""
Unified Email Orchestrator - Single Chatbot Interface
Routes user commands to appropriate agents (Organizer, RAG, CrewAI, Voice)
Fully automated - no manual tab switching required.
"""

import re
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

# Import all agent systems
from email_rag import EmailRAGSystem
from email_reply_agents import EmailReplyAgents, EmailReplyTasks, ReplyWorkflow, EmailReplyOrchestrator
from agents import EmailOrchestrator as EmailOrganizerSystem
from config import Config
from data_processor import EmailDataProcessor

class EmailOrchestrator:
    """Central orchestrator that routes commands to appropriate specialized agents"""
    
    def __init__(self):
        print("🚀 Initializing Unified Email Orchestrator...")
        
        # Initialize session history
        self.session_history = []
        
        # Initialize all agent systems
        try:
            self.rag_system = EmailRAGSystem()
            print("✅ RAG System Ready - Dataset loaded for chat/search/filter")
        except Exception as e:
            print(f"❌ RAG System Error: {e}")
            self.rag_system = None
            
        try:
            self.email_organizer = EmailOrganizerSystem()
            print("✅ Email Organizer Ready - Categorization, Priority, Actions")
        except Exception as e:
            print(f"❌ Email Organizer Error: {e}")
            self.email_organizer = None
            
        try:
            self.reply_agents = EmailReplyAgents()
            self.reply_tasks = EmailReplyTasks()
            self.reply_workflow = ReplyWorkflow()
            print("✅ CrewAI Reply System Ready - Multi-agent email generation")
        except Exception as e:
            print(f"❌ CrewAI Reply System Error: {e}")
            self.reply_agents = None
            self.reply_tasks = None
            self.reply_workflow = None
        
        print("\n🎯 Email Organizer Agent System Ready!")
        print("📊 Dataset emails loaded for RAG chat/search/filter")
        print("🏷️ Email categorization, priority, and action recommendations available")
        print("✉️ Draft and reply generation with context awareness")
        print("🎯 All Systems Initialized - Ready for Commands!")
    
    def parse_command(self, user_input: str) -> Dict[str, Any]:
        """Parse user input into structured command"""
        raw_input = (user_input or "").strip()
        if not raw_input:
            return {
                'command': 'help',
                'raw': '',
                'groups': tuple(),
                'confidence': 0.9
            }
        
        # Command patterns
        patterns = {
            'help': r'^(?:help|commands|what can you do)\s*$',
            'status': r'^(?:status|systems|ready)\s*$',
            'search': r'^(?:search|find|lookup)\s+(.+)$',
            'chat': r'^(?:chat|ask|talk|tell me)\s+(.+)$',
            'draft': r'^(?:draft|write|compose)\s+(?:an?\s+)?(?:email|mail)\s+(?:to\s+)?([^\n]+?)(?:\s+(?:about|regarding|for)\s+(.+))?$',
            'organize': r'^(?:organize|organise)\s*(.*)$',
            'categorize': r'^(?:categorize|classify)\s*(.*)$',
            'reply': r'^(?:reply|respond)\s+(?:to\s+)?(.+)$',
            'filter': r'^(?:filter|show)\s+(.+?)(?:\s+(?:emails?|mail))?\s*$',
            'list': r'^(?:list|show|ls)\s+(all|from\s+.+|by\s+.+|thread\s+.+|emails?\s*.*)$',
            'emails': r'^(?:emails?|mail)\s+(.+)$'
        }
        
        for cmd_type, pattern in patterns.items():
            match = re.match(pattern, raw_input, flags=re.IGNORECASE)
            if match:
                groups = match.groups()
                return {
                    'command': cmd_type,
                    'raw': raw_input,
                    'groups': groups,
                    'confidence': 0.9
                }
        
        # Fallback to chat if no specific command
        return {
            'command': 'chat',
            'raw': raw_input,
            'groups': (raw_input,),
            'confidence': 0.5
        }
    
    def execute_command(self, command: Dict[str, Any]) -> str:
        """Execute the parsed command using appropriate agent"""
        cmd = command['command']
        
        try:
            if cmd == 'search':
                return self._handle_search(command)
            elif cmd == 'chat':
                return self._handle_chat(command)
            elif cmd == 'draft':
                return self._handle_draft(command)
            elif cmd == 'organize':
                return self._handle_organize(command)
            elif cmd == 'categorize':
                return self._handle_categorize(command)
            elif cmd == 'reply':
                return self._handle_reply(command)
            elif cmd == 'filter':
                return self._handle_filter(command)
            elif cmd == 'list':
                return self._handle_list(command)
            elif cmd == 'emails':
                return self._handle_emails_query(command)
            elif cmd == 'help':
                return self._show_help()
            elif cmd == 'status':
                return self._show_status()
            else:
                return self._handle_chat(command)
        except Exception as e:
            return f"❌ Error executing {cmd}: {str(e)}"

    def _load_email_dataset(self):
        """Load the email dataset robustly and normalize expected column names."""
        processor = EmailDataProcessor()
        df = processor.load_dataset("dataset_emails - Sheet1.csv")

        # Defensive normalization for cases where CSV is read with unexpected casing/spaces
        if hasattr(df, "columns"):
            rename_map = {}
            for col in df.columns:
                if not isinstance(col, str):
                    continue
                normalized = col.strip().lower().replace(" ", "_")
                rename_map[col] = normalized
            df = df.rename(columns=rename_map)

        return df
    
    def _handle_search(self, command: Dict[str, Any]) -> str:
        """Handle search commands using RAG system"""
        if not self.rag_system:
            return "❌ RAG System not available"
        
        groups = command.get('groups') or tuple()
        query = (groups[0] if len(groups) > 0 else None) or "general"
        
        print(f"Searching emails for: {query}")
        results = self.rag_system.search_emails(query, k=5)
        
        if not results:
            return f"No emails found for: {query}"
        
        response = f"Routed to: RAG `search_emails(query={query!r}, k=5)`\n\n"
        response += f"Found {len(results)} relevant emails:\n\n"
        for i, result in enumerate(results[:3], 1):
            response += f"{i}. **Subject:** {result['subject']}\n"
            response += f"   **From:** {result['sender']}\n"
            response += f"   **Date:** {result['date']}\n"
            response += f"   **Preview:** {result['content'][:100]}...\n\n"
        
        return response
    
    def _handle_chat(self, command: Dict[str, Any]) -> str:
        """Handle chat commands using RAG system"""
        if not self.rag_system:
            return "❌ RAG System not available"
        
        groups = command.get('groups') or tuple()
        query = (groups[0] if len(groups) > 0 else None) or "general chat"
        print(f"💬 Chatting about: {query}")
        response = self.rag_system.chat_with_emails(query)
        
        return f"🔁 **Routed to:** RAG `chat_with_emails(question={query!r})`\n\n🤖 {response}"
    
    def _handle_draft(self, command: Dict[str, Any]) -> str:
        """Handle draft commands using RAG system"""
        if not self.rag_system:
            return "❌ RAG System not available"
        
        groups = command.get('groups') or tuple()
        recipient = (groups[0] if len(groups) > 0 else None) or "unknown@example.com"
        purpose = (groups[1] if len(groups) > 1 else None) or "General inquiry"
        
        print(f"📝 Drafting email to {recipient} about {purpose}")
        draft = self.rag_system.draft_comprehensive_email(
            recipient=recipient,
            purpose=purpose,
            key_points=[],
            tone="professional"
        )
        
        return (
            f"🔁 **Routed to:** RAG `draft_comprehensive_email(recipient={recipient!r}, purpose={purpose!r})`\n\n"
            f"📧 **Drafted Email:**\n\n{draft}"
        )
    
    def _handle_categorize(self, command: Dict[str, Any]) -> str:
        """Handle categorize commands using Email Organizer"""
        if not self.email_organizer:
            return "❌ Email Organizer not available"
        
        target = command['groups'][0] if command['groups'][0] else "all emails"
        
        print(f"🏷️ Categorizing: {target}")
        
        # Use email organizer agents to categorize
        try:
            df = self._load_email_dataset()

            required_cols = {"sender_name", "sender_email", "subject", "body"}
            missing = required_cols - set([c for c in df.columns])
            if missing:
                return (
                    "❌ Categorization Error: dataset missing required columns: "
                    + ", ".join(sorted(missing))
                    + "\n\nTip: your CSV likely has parsing issues due to multiline quoted fields."
                )
            
            # Initialize agents
            from agents import CategorizationAgent, PriorityAgent, ActionAgent
            
            categorizer = CategorizationAgent()
            priority_agent = PriorityAgent()
            action_agent = ActionAgent()
            
            # Process first few emails as example
            processed_count = 0
            categories = {}
            priorities = {}
            actions = []
            
            for idx, row in df.head(5).iterrows():
                email_data = {
                    'sender_name': row['sender_name'],
                    'subject': row['subject'],
                    'body': row['body'],
                    'sender_email': row['sender_email']
                }
                
                # Categorize
                category_result = categorizer.categorize(email_data)
                categories[category_result.category] = categories.get(category_result.category, 0) + 1
                
                # Priority
                priority_result = priority_agent.assign_priority(email_data)
                priorities[priority_result.priority] = priorities.get(priority_result.priority, 0) + 1
                
                # Action
                action_result = action_agent.recommend_action(
                    email_data, 
                    category_result.category, 
                    priority_result.priority
                )
                actions.append(action_result.action)
                
                processed_count += 1
            
            response = f"🏷️ **Email Categorization Results:**\n\n"
            response += f"📊 **Processed:** {processed_count} emails\n\n"
            response += f"📋 **Categories Found:**\n"
            for cat, count in categories.items():
                response += f"  • {cat}: {count}\n"
            response += f"\n🚨 **Priority Distribution:**\n"
            for pri, count in priorities.items():
                response += f"  • {pri}: {count}\n"
            response += f"\n⚡ **Recommended Actions:**\n"
            for action in set(actions[:3]):
                response += f"  • {action}\n"
            
            return response
            
        except Exception as e:
            return f"❌ Categorization Error: {str(e)}"

    def _handle_organize(self, command: Dict[str, Any]) -> str:
        """Handle organize commands using the full organizer pipeline"""
        if not self.email_organizer:
            return "❌ Email Organizer not available"

        groups = command.get('groups') or tuple()
        target = (groups[0] if len(groups) > 0 else "") or "all"

        try:
            df = self._load_email_dataset()

            required_cols = {"email_id", "sender_name", "sender_email", "subject", "body", "timestamp", "has_attachment", "thread_id"}
            missing = required_cols - set([c for c in df.columns])
            if missing:
                return (
                    "❌ Organize Error: dataset missing required columns: "
                    + ", ".join(sorted(missing))
                    + "\n\nTip: your CSV likely has parsing issues due to multiline quoted fields."
                )

            processor = EmailDataProcessor()

            max_to_process = len(df)
            if target.strip().lower() in {"", "all", "all emails", "inbox"}:
                # Safety cap to avoid very long runs in the chat UI
                max_to_process = min(len(df), 25)
            else:
                m = re.search(r'(\d+)', target)
                if m:
                    max_to_process = min(len(df), max(1, int(m.group(1))))
                else:
                    max_to_process = min(len(df), 10)

            emails = []
            for _, row in df.head(max_to_process).iterrows():
                email_data = processor.preprocess_email(row.to_dict())
                emails.append(email_data)

            results = self.email_organizer.process_batch(emails)

            category_counts: Dict[str, int] = {}
            priority_counts: Dict[str, int] = {}
            action_counts: Dict[str, int] = {}
            spam_count = 0

            high_priority_items = []
            for email, analysis in zip(emails, results):
                cat = getattr(analysis.category, 'category', 'Unknown') if hasattr(analysis, 'category') else 'Unknown'
                pri = getattr(analysis.priority, 'priority', 'Medium') if hasattr(analysis, 'priority') else 'Medium'
                act = getattr(analysis.action, 'action', 'Review') if hasattr(analysis, 'action') else 'Review'

                category_counts[cat] = category_counts.get(cat, 0) + 1
                priority_counts[pri] = priority_counts.get(pri, 0) + 1
                action_counts[act] = action_counts.get(act, 0) + 1

                if str(cat).lower() == 'spam':
                    spam_count += 1

                if str(pri).lower() == 'high' and str(cat).lower() != 'spam':
                    high_priority_items.append({
                        'subject': email.get('subject', ''),
                        'sender': email.get('sender_name', '') or email.get('sender_email', ''),
                        'action': act,
                        'email_id': email.get('email_id')
                    })

            response_lines = []
            response_lines.append("🔁 **Routed to:** Organizer `EmailOrchestrator.process_batch(...)`")
            response_lines.append("")
            response_lines.append(f"📊 **Processed:** {len(results)} emails" + (f" (showing first {len(results)} of {len(df)})" if len(results) < len(df) else ""))
            response_lines.append(f"🛡️ **Spam detected:** {spam_count}")
            response_lines.append("")

            response_lines.append("🏷️ **Top Categories:**")
            for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                response_lines.append(f"- {cat}: {count}")

            response_lines.append("")
            response_lines.append("🚨 **Priority Distribution:**")
            for pri, count in sorted(priority_counts.items(), key=lambda x: x[1], reverse=True):
                response_lines.append(f"- {pri}: {count}")

            response_lines.append("")
            response_lines.append("⚡ **Top Recommended Actions:**")
            for act, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                response_lines.append(f"- {act}: {count}")

            if high_priority_items:
                response_lines.append("")
                response_lines.append("🔥 **High Priority Items (sample):**")
                for item in high_priority_items[:5]:
                    response_lines.append(f"- #{item.get('email_id')}: {item.get('subject', '')} (From: {item.get('sender', '')}) → {item.get('action', '')}")
            else:
                response_lines.append("")
                response_lines.append("No high priority items found in this batch.")

            response_lines.append("")
            response_lines.append("Tip: try `filter high priority` or `reply to \"...\"` for drafting responses.")

            return "\n".join(response_lines)
        except Exception as e:
            return f"❌ Organize Error: {str(e)}"
    
    def _handle_list(self, command: Dict[str, Any]) -> str:
        """Handle list commands using RAG system for structured email listing."""
        if not self.rag_system:
            return "❌ RAG System not available"
        
        groups = command.get('groups') or tuple()
        target = (groups[0] if len(groups) > 0 else "").strip()
        
        try:
            if target.lower() == "all":
                emails = self.rag_system.list_all_emails(limit=15)
                response = "🔁 **Routed to:** RAG `list_all_emails(limit=15)`\n\n"
                response += f"📧 **Recent emails (showing {len(emails)}):\n\n"
                for i, e in enumerate(emails, 1):
                    response += f"{i}. **{e.get('subject', '')}**\n"
                    response += f"   From: {e.get('sender_display', e.get('sender_name', '') + ' (' + e.get('sender_email', '') + ')')}\n"
                    response += f"   Date: {e.get('timestamp', '')}\n"
                    response += f"   Preview: {e.get('body_preview', '')}\n\n"
                return response
            elif target.lower().startswith("from "):
                sender = target[5:].strip()
                emails = self.rag_system.list_emails_by_sender(sender, limit=10)
                response = f"🔁 **Routed to:** RAG `list_emails_by_sender(sender={sender!r}, limit=10)`\n\n"
                response += f"📧 **Emails from {sender} (showing {len(emails)}):\n\n"
                for i, e in enumerate(emails, 1):
                    response += f"{i}. **{e.get('subject', '')}**\n"
                    response += f"   Date: {e.get('timestamp', '')}\n"
                    response += f"   Preview: {e.get('body_preview', '')}\n\n"
                return response
            elif target.lower().startswith("thread "):
                thread_id = target[7:].strip()
                emails = self.rag_system.filter_by_thread_id(thread_id)
                response = f"🔁 **Routed to:** RAG `filter_by_thread_id(thread_id={thread_id!r})`\n\n"
                response += f"🧵 **Thread {thread_id} ({len(emails)} emails):\n\n"
                for i, e in enumerate(emails, 1):
                    response += f"{i}. **{e.get('subject', '')}**\n"
                    response += f"   From: {e.get('sender_display', e.get('sender_name', '') + ' (' + e.get('sender_email', '') + ')')}\n"
                    response += f"   Date: {e.get('timestamp', '')}\n"
                    response += f"   Preview: {e.get('body_preview', '')}\n\n"
                return response
            else:
                return f"❌ Unrecognized list target: {target}\n💡 Try: `list all`, `list from <sender>`, `list thread <thread_id>`"
        except Exception as e:
            return f"❌ List Error: {str(e)}"
    
    def _handle_emails_query(self, command: Dict[str, Any]) -> str:
        """Handle generic 'emails <query>' using RAG natural language query."""
        if not self.rag_system:
            return "❌ RAG System not available"
        
        groups = command.get('groups') or tuple()
        query = (groups[0] if len(groups) > 0 else "").strip()
        
        if not query:
            return "❌ Please provide a query after 'emails'. Example: `emails invoice`"
        
        try:
            emails = self.rag_system.query_emails(query, limit=12)
            response = f"🔁 **Routed to:** RAG `query_emails(query={query!r}, limit=12)`\n\n"
            response += f"📧 **Emails matching '{query}' (showing {len(emails)}):\n\n"
            for i, e in enumerate(emails, 1):
                response += f"{i}. **{e.get('subject', '')}**\n"
                response += f"   From: {e.get('sender_display', e.get('sender_name', '') + ' (' + e.get('sender_email', '') + ')')}\n"
                response += f"   Date: {e.get('timestamp', '')}\n"
                response += f"   Preview: {e.get('body_preview', '')}\n\n"
            return response
        except Exception as e:
            return f"❌ Emails Query Error: {str(e)}"
    
    def _handle_reply(self, command: Dict[str, Any]) -> str:
        """Handle reply commands using CrewAI system"""
        if not self.reply_agents:
            return "❌ CrewAI Reply System not available"
        
        groups = command.get('groups') or tuple()
        email_content = (groups[0] if len(groups) > 0 else None) or "Sample email content"
        
        print(f"✉️ Generating reply using CrewAI...")
        try:
            reply_orchestrator = EmailReplyOrchestrator()
            workflow = reply_orchestrator.process_email_reply(email_content)
 
            category = workflow.category.category if workflow.category else "Unknown"
            reply_text = workflow.reply.reply_content if workflow.reply else ""
            research_text = workflow.research or ""
 
            return (
                f"🔁 **Routed to:** CrewAI `process_email_reply(email_content=...)`\n\n"
                f"🏷️ **Detected Category:** {category}\n\n"
                f"✅ **Generated Reply:**\n\n{reply_text}\n\n"
                f"🔎 **Research Used:**\n{research_text}"
            )
        except Exception as e:
            return f"❌ CrewAI Error: {str(e)}"
    
    def _handle_filter(self, command: Dict[str, Any]) -> str:
        """Handle filter commands using Email Organizer"""
        if not self.email_organizer:
            return "❌ Email Organizer not available"
        
        filter_type = command['groups'][0] if command['groups'][0] else "all"
        
        print(f"🔍 Filtering emails by: {filter_type}")
        
        try:
            df = self._load_email_dataset()

            required_cols = {"sender_name", "sender_email", "subject", "body"}
            missing = required_cols - set([c for c in df.columns])
            if missing:
                return (
                    "❌ Filter Error: dataset missing required columns: "
                    + ", ".join(sorted(missing))
                    + "\n\nTip: your CSV likely has parsing issues due to multiline quoted fields."
                )
            
            # Initialize agents
            from agents import PriorityAgent, CategorizationAgent
            
            priority_agent = PriorityAgent()
            categorizer = CategorizationAgent()
            
            filtered_emails = []
            
            for idx, row in df.head(10).iterrows():
                email_data = {
                    'sender_name': row['sender_name'],
                    'subject': row['subject'],
                    'body': row['body'],
                    'sender_email': row['sender_email']
                }
                
                # Check if email matches filter criteria
                matches_filter = False
                
                if filter_type.lower() in ['high', 'urgent', 'priority']:
                    priority_result = priority_agent.assign_priority(email_data)
                    if priority_result.priority.lower() in ['high', 'urgent']:
                        matches_filter = True
                
                elif filter_type.lower() in ['work', 'business', 'professional']:
                    category_result = categorizer.categorize(email_data)
                    if 'work' in category_result.category.lower() or 'business' in category_result.category.lower():
                        matches_filter = True
                
                elif filter_type.lower() in ['personal', 'private']:
                    category_result = categorizer.categorize(email_data)
                    if 'personal' in category_result.category.lower():
                        matches_filter = True
                
                else:
                    # General filter - check if filter type appears in content
                    content = f"{email_data['subject']} {email_data['body']}".lower()
                    if filter_type.lower() in content:
                        matches_filter = True
                
                if matches_filter:
                    filtered_emails.append({
                        'subject': email_data['subject'],
                        'sender': email_data['sender_name'],
                        'priority': priority_agent.assign_priority(email_data).priority if 'priority' in filter_type.lower() else 'Normal'
                    })
            
            response = f"🔍 **Filter Results ({filter_type}):**\n\n"
            if filtered_emails:
                response += f"📧 Found {len(filtered_emails)} matching emails:\n\n"
                for i, email in enumerate(filtered_emails[:5], 1):
                    response += f"{i}. **{email['subject']}**\n"
                    response += f"   From: {email['sender']}\n"
                    response += f"   Priority: {email['priority']}\n\n"
            else:
                response += f"📭 No emails found matching filter: {filter_type}\n"
                response += "💡 Try filters like: high, work, personal, urgent"
            
            return response
            
        except Exception as e:
            return f"❌ Filter Error: {str(e)}"
    
    def _show_help(self) -> str:
        """Show available commands"""
        help_text = """
🤖 **Unified Email Assistant Commands:**

📧 **Email Operations:**
• `search [query]` - Search emails using RAG
• `chat [topic]` - Chat with your email database  
• `draft email to [recipient] about [topic]` - Draft email using RAG
• `reply to [email content]` - Generate reply using CrewAI agents
• `categorize [emails]` - Categorize emails using Organizer
• `organize all` - Run full organizer pipeline (spam → categorize → priority → action)
• `filter [type]` - Filter emails by category/priority
• `list all` - List recent emails with sender/timestamp/preview
• `list from <sender>` - List emails from a specific sender
• `list thread <thread_id>` - Show full conversation thread
• `emails <query>` - Natural language search over sender/subject/body/thread

🔧 **System Commands:**
• `status` - Show system status
• `help` - Show this help message

💡 **Examples:**
• `search pricing plans`
• `chat What are our main customer concerns?`
• `draft email to john@example.com about meeting follow-up`
• `reply to "I'm interested in your enterprise pricing"`
• `categorize all emails`
• `organize all`
• `filter high priority`
• `list all`
• `list from Alice`
• `list thread thread_001`
• `emails invoice`
• `status`

🎯 **All agents work together automatically - just type your command!**
        """
        return help_text
    
    def _show_status(self) -> str:
        """Show status of all agent systems"""
        lines = []
        
        if self.rag_system:
            lines.append("✅ RAG System: Ready")
            lines.append("Suggested commands:")
            lines.append("- search pricing")
            lines.append("- search invoice")
            lines.append("- chat What are the main urgent items this week?")
            lines.append("- draft email to client@example.com about follow-up meeting")
            lines.append("- list all")
            lines.append("- list from John")
            lines.append("- list thread thread_001")
            lines.append("- emails invoice")
        else:
            lines.append("❌ RAG System: Not Available")
        
        lines.append("")
        
        if self.email_organizer:
            lines.append("✅ Email Organizer: Ready")
            lines.append("Suggested commands:")
            lines.append("- organize all")
            lines.append("- categorize all emails")
            lines.append("- filter high priority")
            lines.append("- filter work")
            lines.append("- filter personal")
        else:
            lines.append("❌ Email Organizer: Not Available")
        
        lines.append("")
        
        if self.reply_agents:
            lines.append("✅ CrewAI Reply System: Ready")
            lines.append("Suggested commands:")
            lines.append("- reply to \"I am interested in your enterprise pricing\"")
            lines.append("- reply to \"Your invoice is overdue\"")
            lines.append("- reply to \"I want a refund\"")
        else:
            lines.append("❌ CrewAI Reply System: Not Available")
        
        lines.append("")
        lines.append("Tip: Type 'help' for the full command list.")
        
        return "\n".join(lines)
    
    def process_input(self, user_input: str) -> str:
        """Main processing pipeline"""
        if not user_input or not user_input.strip():
            return "Please enter a command. Type 'help' to see available commands."
        
        # Parse command
        command = self.parse_command(user_input)
        
        # Add to session history
        self.session_history.append({
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'command': command['command'],
            'response': None
        })
        
        # Execute command
        response = self.execute_command(command)
        
        # Update session history
        if self.session_history:
            self.session_history[-1]['response'] = response
        
        return response
    
    def start_interactive(self):
        """Start interactive chatbot interface"""
        print("🚀 **Unified Email Assistant Started**")
        print("=" * 60)
        print("🤖 All agents are ready! Type your commands below.")
        print("💡 Type 'help' to see all available commands.")
        print("🎯 RAG + CrewAI + Organizer + Voice - Fully Automated!")
        print("=" * 60)
        
        while True:
            try:
                user_input = input("\n💬 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\n👋 Goodbye! All systems shutting down.")
                    break
                
                if user_input.lower() in ['clear', 'cls']:
                    print("\n" * 50)
                    continue
                
                # Process the command
                response = self.process_input(user_input)
                print(f"\n🤖 Assistant: {response}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Type 'quit' to exit gracefully.")
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")

# Main execution
if __name__ == "__main__":
    orchestrator = EmailOrchestrator()
    orchestrator.start_interactive()
