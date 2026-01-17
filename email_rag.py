from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import os
import pandas as pd
import json
from typing import List, Dict, Any
from crewai import Crew, Process, Task, Agent
from config import Config
from data_processor import EmailDataProcessor
from datetime import datetime
import re

class EmailRAGSystem:
    """RAG system for intelligent email chat and drafting"""
    
    def __init__(self, csv_path: str = "dataset_emails - Sheet1.csv"):
        self.csv_path = csv_path
        # Use Groq embeddings instead of Ollama
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.db_location = "./email_chroma_db_v3"
        self.vector_store = None
        # Use Groq LLM instead of Ollama
        self.model = ChatGroq(
            api_key=Config.GROQ_API_KEY,
            model_name="llama-3.1-8b-instant",  # Use llama for email organizer
            temperature=0.7
        )
        
        self._initialize_vector_store()
        self._setup_rag_chain()

    def _load_emails_df(self) -> pd.DataFrame:
        try:
            return pd.read_csv(self.csv_path, engine="python")
        except Exception:
            return pd.read_csv(self.csv_path)

    def _load_emails_clean(self) -> List[Dict[str, Any]]:
        """Load emails using EmailDataProcessor for robust parsing."""
        processor = EmailDataProcessor()
        df = processor.load_dataset(self.csv_path)
        emails = []
        for _, row in df.iterrows():
            emails.append({
                "email_id": row.get("email_id"),
                "sender_email": row.get("sender_email"),
                "sender_name": row.get("sender_name"),
                "subject": row.get("subject"),
                "body": row.get("body"),
                "timestamp": row.get("timestamp"),
                "has_attachment": row.get("has_attachment"),
                "thread_id": row.get("thread_id")
            })
        return emails

    def list_all_emails(self, limit: int = 20) -> List[Dict[str, Any]]:
        """List all emails with sender, timestamp, subject, and body preview."""
        emails = self._load_emails_clean()
        sorted_emails = sorted(emails, key=lambda e: e.get("timestamp", ""), reverse=True)
        for e in sorted_emails[:limit]:
            e["body_preview"] = (e.get("body", "")[:120] + "...") if len(e.get("body", "")) > 120 else e.get("body", "")
            e["sender_display"] = f"{e.get('sender_name', '')} ({e.get('sender_email', '')})"
        return sorted_emails[:limit]

    def list_emails_by_sender(self, sender: str, limit: int = 10) -> List[Dict[str, Any]]:
        """List emails from a specific sender (matches name or email)."""
        emails = self._load_emails_clean()
        sender_lower = sender.lower()
        filtered = [
            e for e in emails
            if sender_lower in e.get("sender_name", "").lower() or sender_lower in e.get("sender_email", "").lower()
        ]
        sorted_filtered = sorted(filtered, key=lambda e: e.get("timestamp", ""), reverse=True)
        for e in sorted_filtered[:limit]:
            e["body_preview"] = (e.get("body", "")[:120] + "...") if len(e.get("body", "")) > 120 else e.get("body", "")
            e["sender_display"] = f"{e.get('sender_name', '')} ({e.get('sender_email', '')})"
        return sorted_filtered[:limit]

    def filter_by_thread_id(self, thread_id: str) -> List[Dict[str, Any]]:
        """Return all emails in a specific thread."""
        emails = self._load_emails_clean()
        thread_emails = [e for e in emails if str(e.get("thread_id", "")) == str(thread_id)]
        sorted_thread = sorted(thread_emails, key=lambda e: e.get("timestamp", ""))
        for e in sorted_thread:
            e["body_preview"] = (e.get("body", "")[:120] + "...") if len(e.get("body", "")) > 120 else e.get("body", "")
            e["sender_display"] = f"{e.get('sender_name', '')} ({e.get('sender_email', '')})"
        return sorted_thread

    def query_emails(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Natural language query over emails (sender, subject, body, thread_id)."""
        emails = self._load_emails_clean()
        query_lower = query.lower()
        matched = []
        for e in emails:
            searchable = (
                e.get("sender_name", "") + " " +
                e.get("sender_email", "") + " " +
                e.get("subject", "") + " " +
                e.get("body", "") + " " +
                e.get("thread_id", "")
            ).lower()
            if query_lower in searchable:
                matched.append(e)
        # Sort by timestamp descending
        sorted_matched = sorted(matched, key=lambda e: e.get("timestamp", ""), reverse=True)
        for e in sorted_matched[:limit]:
            e["body_preview"] = (e.get("body", "")[:120] + "...") if len(e.get("body", "")) > 120 else e.get("body", "")
            e["sender_display"] = f"{e.get('sender_name', '')} ({e.get('sender_email', '')})"
        return sorted_matched[:limit]
    
    def _initialize_vector_store(self):
        """Initialize or load the vector store"""
        add_documents = not os.path.exists(self.db_location)
        
        # Create vector store
        self.vector_store = Chroma(
            collection_name="email_database_v3",
            persist_directory=self.db_location,
            embedding_function=self.embeddings
        )
        
        if add_documents:
            self._populate_vector_store()
        
        # Create retriever
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 5}
        )
    
    def _populate_vector_store(self):
        """Populate vector store with email data"""
        print("📧 Loading email data for RAG system...")
        
        try:
            processor = EmailDataProcessor()
            df = processor.load_dataset(self.csv_path)
            documents = []
            ids = []
            
            for i, row in df.iterrows():
                # Create comprehensive document content
                subject = str(row.get('subject', '') or '')
                sender_name = str(row.get('sender_name', '') or '')
                sender_email = str(row.get('sender_email', '') or '')
                ts_val = row.get('timestamp', '')
                timestamp = ts_val.isoformat() if hasattr(ts_val, "isoformat") else str(ts_val or '')
                body = str(row.get('body', '') or '')
                email_id = str(row.get('email_id', i) or i)
                has_attachment = row.get('has_attachment', '')
                thread_id = str(row.get('thread_id', '') or '')

                content = (
                    f"email_id: {email_id}\n"
                    f"subject: {subject}\n"
                    f"sender_name: {sender_name}\n"
                    f"sender_email: {sender_email}\n"
                    f"timestamp: {timestamp}\n"
                    f"has_attachment: {has_attachment}\n"
                    f"thread_id: {thread_id}\n\n"
                    f"body:\n{body}\n"
                )
                
                document = Document(
                    page_content=content.strip(),
                    metadata={
                        "subject": subject,
                        "sender": sender_name or sender_email,
                        "sender_email": sender_email,
                        "date": timestamp,
                        "thread_id": thread_id,
                        "has_attachment": bool(has_attachment) if isinstance(has_attachment, (bool, int)) else str(has_attachment).lower() == "true",
                        "email_id": email_id
                    },
                    id=email_id
                )
                ids.append(email_id)
                documents.append(document)
            
            # Add to vector store
            print(f"📚 Adding {len(documents)} emails to vector database...")
            self.vector_store.add_documents(documents=documents, ids=ids)
            print("✅ Email database populated successfully!")
            
        except Exception as e:
            print(f"❌ Error populating vector store: {str(e)}")
    
    def _setup_rag_chain(self):
        """Setup the RAG chain for email processing"""
        
        # Template for email-related queries
        email_template = """
        You are an expert email assistant with access to a database of previous emails and communications.
        
        Here are some relevant emails from the database:
        {relevant_context}
        
        Based on the above emails and your general knowledge, help with this request: {question}
        
        Guidelines:
        - Reference specific emails when relevant (mention sender/date if helpful)
        - Maintain professional tone
        - Provide actionable advice
        - If drafting emails, make them contextually appropriate
        - If answering questions, be comprehensive but concise
        """
        
        self.email_prompt = ChatPromptTemplate.from_template(email_template)
        
        # Create RAG chain
        self.email_chain = (
            {
                "relevant_context": self.retriever | RunnableLambda(self._format_docs),
                "question": RunnablePassthrough()
            }
            | self.email_prompt
            | self.model
            | StrOutputParser()
        )

    def _format_docs(self, docs: List[Document]) -> str:
        if not docs:
            return "(no relevant emails found)"
        parts = []
        for d in docs[:6]:
            subj = d.metadata.get("subject", "")
            sender = d.metadata.get("sender", "")
            date = d.metadata.get("date", "")
            email_id = d.metadata.get("email_id", "")
            snippet = d.page_content
            snippet = snippet.replace("\n", " ").strip()
            if len(snippet) > 350:
                snippet = snippet[:350] + "..."
            parts.append(
                f"- email_id: {email_id}\n  subject: {subj}\n  sender: {sender}\n  date: {date}\n  excerpt: {snippet}"
            )
        return "\n\n".join(parts)

    def chat_with_emails(self, question: str) -> str:
        """Chat with the email database using RAG, with intent handling for list operations."""
        q_lower = (question or "").lower().strip()
        normalized = q_lower.replace("emials", "emails").replace("emial", "email")
        wants_list = ("list" in normalized) or ("show" in normalized)

        # list senders / sender emails (supports typos and count requests like "list me 2 emails senders")
        if wants_list and ("sender" in normalized or "senders" in normalized):
            m = re.search(r"\b(\d+)\b", normalized)
            if m:
                limit = max(1, min(int(m.group(1)), 50))
                emails = self.list_all_emails(limit=limit)
                response = f"🔁 **Routed to:** RAG `chat_with_emails(question={question!r})` → list {limit} email senders\n\n"
                response += f"📧 **Emails (showing {len(emails)}):\n\n"
                for i, e in enumerate(emails, 1):
                    sender_display = e.get("sender_display") or f"{e.get('sender_name', '')} ({e.get('sender_email', '')})"
                    body_text = e.get("body", "") or ""
                    body_out = body_text
                    if len(body_out) > 4000:
                        body_out = body_out[:4000] + "..."
                    response += (
                        f"{i}. **email_id:** {e.get('email_id', '')}\n"
                        f"   **From:** {sender_display}\n"
                        f"   **Sender Email:** {e.get('sender_email', '')}\n"
                        f"   **Subject:** {e.get('subject', '')}\n"
                        f"   **Timestamp:** {e.get('timestamp', '')}\n"
                        f"   **Thread:** {e.get('thread_id', '')}\n"
                        f"   **Has Attachment:** {e.get('has_attachment', '')}\n\n"
                        f"   **Body:**\n{body_out}\n\n"
                    )
                return response

            emails = self._load_emails_clean()
            sender_counts: Dict[str, int] = {}
            for e in emails:
                sender_display = e.get("sender_display") or f"{e.get('sender_name', '')} ({e.get('sender_email', '')})"
                sender_counts[sender_display] = sender_counts.get(sender_display, 0) + 1

            response = f"🔁 **Routed to:** RAG `chat_with_emails(question={question!r})` → list unique senders\n\n"
            response += f"📧 **All senders in database ({len(sender_counts)} unique):\n\n"
            for i, (sender, count) in enumerate(sorted(sender_counts.items(), key=lambda x: x[1], reverse=True), 1):
                response += f"{i}. {sender} — {count} email{'s' if count != 1 else ''}\n"
            return response

        # list all emails
        if any(phrase in normalized for phrase in [
            "list all emails", "show all emails", "list emails", "show emails"
        ]):
            emails = self.list_all_emails(limit=10)
            response = f"🔁 **Routed to:** RAG `chat_with_emails(question={question!r})` → list all\n\n"
            response += f"📧 **Recent emails (showing {len(emails)}):\n\n"
            for i, e in enumerate(emails, 1):
                sender_display = e.get("sender_display") or f"{e.get('sender_name', '')} ({e.get('sender_email', '')})"
                response += (
                    f"{i}. **{e.get('subject', '')}**\n"
                    f"   From: {sender_display}\n"
                    f"   Date: {e.get('timestamp', '')}\n"
                    f"   Preview: {e.get('body_preview', '')}\n\n"
                )
            return response

        # list emails from <sender>
        if "list from " in normalized or "show emails from " in normalized:
            sender = normalized.split("list from ")[1].strip() if "list from " in normalized else normalized.split("show emails from ")[1].strip()
            emails = self.list_emails_by_sender(sender, limit=10)
            response = f"🔁 **Routed to:** RAG `chat_with_emails(question={question!r})` → list from\n\n"
            response += f"📧 **Emails from {sender} (showing {len(emails)}):\n\n"
            for i, e in enumerate(emails, 1):
                response += (
                    f"{i}. **{e.get('subject', '')}**\n"
                    f"   Date: {e.get('timestamp', '')}\n"
                    f"   Preview: {e.get('body_preview', '')}\n\n"
                )
            return response

        # thread <id>
        if normalized.startswith("thread ") or " thread " in normalized:
            thread_id = normalized.split("thread ", 1)[1].strip() if "thread " in normalized else ""
            if thread_id:
                emails = self.filter_by_thread_id(thread_id)
                response = f"🔁 **Routed to:** RAG `chat_with_emails(question={question!r})` → thread\n\n"
                response += f"🧵 **Thread {thread_id} ({len(emails)} emails):\n\n"
                for i, e in enumerate(emails, 1):
                    sender_display = e.get("sender_display") or f"{e.get('sender_name', '')} ({e.get('sender_email', '')})"
                    response += (
                        f"{i}. **{e.get('subject', '')}**\n"
                        f"   From: {sender_display}\n"
                        f"   Date: {e.get('timestamp', '')}\n"
                        f"   Preview: {e.get('body_preview', '')}\n\n"
                    )
                return response

        # Default: RAG retrieval + LLM
        try:
            print(f"🔍 Searching emails for: {question}")
            relevant_emails = self.retriever.invoke(question)
            print(f"📧 Found {len(relevant_emails)} relevant emails")
            return self.email_chain.invoke(question)
        except Exception as e:
            return f"❌ Error processing your question: {str(e)}"
    
    def draft_comprehensive_email(self, 
                              recipient: str,
                              purpose: str,
                              key_points: List[str],
                              tone: str = "professional") -> str:
        """Draft a comprehensive email using RAG"""

        draft_prompt = """
        Draft a comprehensive email with these specifications:

        Recipient: {recipient}
        Purpose: {purpose}
        Key Points to Include: {key_points}
        Tone: {tone}

        Here are some relevant emails for context:
        {relevant_context}

        Please draft:
        1. A compelling subject line
        2. Professional greeting
        3. Well-structured body covering all key points
        4. Appropriate closing
        5. Any necessary follow-up actions

        Make it contextually relevant and professional.
        """

        try:
            # Get context from similar emails
            context_query = f"email to {recipient} about {purpose}"
            relevant_emails = self.retriever.invoke(context_query)
            formatted_context = self._format_docs(relevant_emails)

            draft_template = ChatPromptTemplate.from_template(draft_prompt)
            draft_chain = draft_template | self.model | StrOutputParser()
            
            response = draft_chain.invoke({
                "relevant_context": formatted_context,
                "recipient": recipient,
                "purpose": purpose,
                "key_points": ", ".join(key_points) if key_points else "",
                "tone": tone
            })
            
            return response
            
        except Exception as e:
            return f"❌ Error drafting email: {str(e)}"
    
    def search_emails(self, query: str, k: int = 5) -> List[Dict]:
        """Search for relevant emails"""
        try:
            # Chroma retriever's invoke() does not accept per-call search_kwargs in some versions.
            # Use the vector store search API directly for consistent behavior.
            results = self.vector_store.similarity_search(query, k=k)
            
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
            print(f"❌ Error searching emails: {str(e)}")
            return []
    
    def interactive_chat(self):
        """Interactive chat interface"""
        print("🤖 Email RAG Assistant")
        print("=" * 50)
        print("Commands:")
        print("- Type your question to chat with emails")
        print("- 'draft <recipient> <purpose>' to draft emails")
        print("- 'search <query>' to search emails")
        print("- 'quit' to exit")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\n💬 You: ").strip()
                
                if user_input.lower() == 'quit':
                    print("👋 Goodbye!")
                    break
                
                elif user_input.lower().startswith('draft '):
                    # Parse draft command
                    parts = user_input[6:].split(' ', 2)
                    if len(parts) >= 2:
                        recipient = parts[0]
                        purpose = parts[1]
                        key_points = parts[2].split(',') if len(parts) > 2 else []
                        
                        print(f"📝 Drafting email to {recipient} about {purpose}...")
                        draft = self.draft_comprehensive_email(recipient, purpose, key_points)
                        print("\n📧 Drafted Email:")
                        print("-" * 40)
                        print(draft)
                        print("-" * 40)
                    else:
                        print("❌ Usage: draft <recipient> <purpose> [key_point1, key_point2, ...]")
                
                elif user_input.lower().startswith('search '):
                    # Parse search command
                    query = user_input[7:]
                    print(f"🔍 Searching emails for: {query}")
                    results = self.search_emails(query)
                    
                    if results:
                        print(f"\n📧 Found {len(results)} relevant emails:")
                        for i, result in enumerate(results, 1):
                            print(f"\n{i}. Subject: {result['subject']}")
                            print(f"   From: {result['sender']}")
                            print(f"   Date: {result['date']}")
                            print(f"   Preview: {result['content'][:200]}...")
                    else:
                        print("📭 No relevant emails found.")
                
                else:
                    # Regular chat
                    print("🤖 Thinking...")
                    response = self.chat_with_emails(user_input)
                    print(f"\n🤖 Assistant: {response}")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")

# Main execution
if __name__ == "__main__":
    rag_system = EmailRAGSystem()
    rag_system.interactive_chat()
