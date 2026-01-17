from agents import EmailOrchestrator

# Test with different types of emails
test_emails = [
    {
        'email_id': 1,
        'sender_email': 'john.smith@techcorp.com',
        'sender_name': 'John Smith',
        'subject': 'Project Alpha - Budget Approval Needed',
        'body': 'Hi Team, I need urgent approval for Project Alpha budget increase. We are looking at an additional 50K for Q1 due to unexpected infrastructure costs. Can we schedule a meeting this week to discuss the budget requirements? This is critical for our Q1 deliverables. Thanks, John',
        'timestamp': '2024-01-15T09:23:00Z',
        'has_attachment': False,
        'thread_id': 'thread_001'
    },
    {
        'email_id': 2,
        'sender_email': 'spam@fake.com',
        'sender_name': 'Unknown Sender',
        'subject': 'WINNER!!! You have won 1,000,000!!!',
        'body': 'CONGRATULATIONS!!! You are the lucky winner of our lottery! Click here immediately to claim your 1,000,000 prize!!! This offer expires in 24 hours!!! ACT NOW!!!',
        'timestamp': '2024-01-15T10:15:00Z',
        'has_attachment': False,
        'thread_id': 'thread_002'
    }
]

# Initialize orchestrator with spam detection
orchestrator = EmailOrchestrator(use_langgraph=False)

print('🎯 Testing Integrated Spam Detection System:')
print('=' * 60)

for i, email in enumerate(test_emails, 1):
    print(f'\n📧 Email {i}: {email["subject"]}')
    print('-' * 40)
    
    try:
        result = orchestrator.process_email(email)
        
        # Spam detection results
        spam_info = result.spam_detection
        print(f'🚫 Spam Detection: {"SPAM" if spam_info.get("is_spam", False) else "NOT SPAM"}')
        print(f'📊 Spam Score: {spam_info.get("spam_score", 0):.1f}')
        print(f'🔍 Confidence: {spam_info.get("confidence", 0):.2%}')
        print(f'💭 Reasoning: {spam_info.get("reasoning", "No reasoning")}')
        
        # Category results
        print(f'📋 Category: {result.category.category}')
        print(f'🎯 Priority: {result.priority.priority}')
        print(f'⚡ Action: {result.action.action}')
        
    except Exception as e:
        print(f'❌ Error: {str(e)}')

print('=' * 60)
print('✅ Spam detection integration complete!')
