# Command → Agent Mapping

## RAG Agent Commands
- `search [query]` → RAG search_emails
- `chat [topic]` → RAG chat_with_emails (with intent detection for list operations)
- `draft email to [recipient] about [topic]` → RAG draft_comprehensive_email
- `list all` → RAG list_all_emails
- `list from <sender>` → RAG list_emails_by_sender
- `list thread <thread_id>` → RAG filter_by_thread_id
- `emails <query>` → RAG query_emails

## Email Organizer Commands
- `categorize [emails]` → EmailOrganizerSystem.process_email / process_batch
- `organize all` → EmailOrganizerSystem.process_batch (full pipeline: spam → categorize → priority → action)
- `filter [type]` → EmailOrganizerSystem (filter by category/priority after processing)

## CrewAI Reply Commands
- `reply to [email content]` → EmailReplyOrchestrator (categorize → research → write workflow)

## Voice Agent Commands (via unified_chatbot)
- `show_urgent` → EmailVoiceAgent (mock urgent emails)
- `archive_newsletters` → EmailVoiceAgent (mock archive)
- `schedule_meeting` → EmailVoiceAgent (mock schedule)
- `search_emails` → EmailVoiceAgent (mock search)
- `categorize_all` → EmailVoiceAgent (mock categorize)
- `generate_reply` → EmailVoiceAgent (mock reply)
- `get_summary` → EmailVoiceAgent (mock summary)
- `set_priority` → EmailVoiceAgent (mock set priority)

## Natural Language Fallbacks
- Any unrecognized command falls back to `chat` → RAG chat_with_emails (with enhanced intent detection for list operations)

## Intent Detection in chat_with_emails
- "list me sender emails" / "list sender emails" → unique senders with counts
- "list all emails" / "show all emails" → list_all_emails
- "list from <sender>" / "show emails from <sender>" → list_emails_by_sender
- "thread <id>" / "show thread <id>" → filter_by_thread_id
- Otherwise → RAG search (retriever + LLM)
