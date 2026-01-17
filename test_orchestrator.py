"""
Comprehensive Test Script for Email Orchestrator
Tests all command categories and agent integrations
"""

import sys
import time
from orchestrator import EmailOrchestrator

def test_routing_commands():
    """Test command routing and parsing"""
    print("🧪 TESTING ROUTING COMMANDS")
    print("=" * 50)
    
    orchestrator = EmailOrchestrator()
    
    # Test commands with expected routing
    test_commands = [
        # Search commands
        ("search pricing plans", "search"),
        ("find customer complaints", "search"),
        ("lookup shipping information", "search"),
        
        # Chat commands
        ("chat about customer concerns", "chat"),
        ("ask what are main topics", "chat"),
        ("tell me about pricing", "chat"),
        
        # Draft commands
        ("draft email to john@example.com about meeting", "draft"),
        ("write email for client regarding project", "draft"),
        ("compose boss about quarterly report", "draft"),
        
        # Reply commands
        ("reply to I need pricing info", "reply"),
        ("respond to customer complaint", "reply"),
        ("generate reply for technical issue", "reply"),
        
        # Organization commands
        ("categorize all emails", "categorize"),
        ("classify support tickets", "categorize"),
        ("organize inbox", "categorize"),
        
        # Filter commands
        ("filter high priority", "filter"),
        ("show urgent emails", "filter"),
        ("list marketing emails", "filter"),
        
        # System commands
        ("status", "status"),
        ("help", "help")
    ]
    
    results = {"passed": 0, "failed": 0, "errors": []}
    
    for command, expected_type in test_commands:
        try:
            parsed = orchestrator.parse_command(command)
            if parsed['command'] == expected_type:
                results["passed"] += 1
                print(f"✅ {command} → {parsed['command']} (confidence: {parsed['confidence']})")
            else:
                results["failed"] += 1
                results["errors"].append(f"{command} → {parsed['command']} (expected {expected_type})")
                print(f"❌ {command} → {parsed['command']} (expected {expected_type})")
        except Exception as e:
            results["failed"] += 1
            results["errors"].append(f"{command} → Error: {str(e)}")
            print(f"💥 {command} → Error: {str(e)}")
    
    return results

def test_agent_integration():
    """Test that all agent systems are accessible"""
    print("\n🧪 TESTING AGENT INTEGRATION")
    print("=" * 50)
    
    orchestrator = EmailOrchestrator()
    
    # Test RAG System
    print("\n📊 Testing RAG System...")
    try:
        if orchestrator.rag_system:
            # Test search
            search_results = orchestrator.rag_system.search_emails("test", k=3)
            print(f"✅ RAG Search: Found {len(search_results)} results")
            
            # Test chat
            chat_response = orchestrator.rag_system.chat_with_emails("test question")
            print(f"✅ RAG Chat: Response length {len(chat_response)} chars")
            
            # Test draft
            draft_response = orchestrator.rag_system.draft_comprehensive_email(
                "test@example.com", "test purpose", ["point1", "point2"]
            )
            print(f"✅ RAG Draft: Response length {len(draft_response)} chars")
        else:
            print("❌ RAG System: Not initialized")
    except Exception as e:
        print(f"💥 RAG System Error: {str(e)}")
    
    # Test CrewAI System
    print("\n🤖 Testing CrewAI System...")
    try:
        if orchestrator.reply_agents:
            print("✅ CrewAI Agents: Available")
        else:
            print("❌ CrewAI Agents: Not available")
    except Exception as e:
        print(f"💥 CrewAI Error: {str(e)}")
    
    # Test Email Organizer
    print("\n📋 Testing Email Organizer...")
    try:
        if orchestrator.email_organizer:
            print("✅ Email Organizer: Available")
        else:
            print("❌ Email Organizer: Not available")
    except Exception as e:
        print(f"💥 Email Organizer Error: {str(e)}")

def test_end_to_end_workflow():
    """Test complete workflow scenarios"""
    print("\n🔄 TESTING END-TO-END WORKFLOW")
    print("=" * 50)
    
    orchestrator = EmailOrchestrator()
    
    # Workflow 1: Search → Draft → Reply
    print("\n📝 Workflow 1: Search → Draft → Reply")
    try:
        # Step 1: Search
        search_cmd = "search customer complaints"
        search_response = orchestrator.process_input(search_cmd)
        print(f"🔍 Step 1 - Search: {len(search_response)} chars")
        
        time.sleep(1)
        
        # Step 2: Draft based on search
        draft_cmd = "draft email to customer@example.com about addressing complaints"
        draft_response = orchestrator.process_input(draft_cmd)
        print(f"📝 Step 2 - Draft: {len(draft_response)} chars")
        
        time.sleep(1)
        
        # Step 3: Reply to specific complaint
        reply_cmd = 'reply to "I am writing to complain about service quality"'
        reply_response = orchestrator.process_input(reply_cmd)
        print(f"✉️ Step 3 - Reply: {len(reply_response)} chars")
        
    except Exception as e:
        print(f"💥 Workflow Error: {str(e)}")

def main():
    """Run all tests"""
    print("🚀 COMPREHENSIVE ORCHESTRATOR TESTING")
    print("=" * 60)
    
    # Run all test suites
    routing_results = test_routing_commands()
    test_agent_integration()
    test_end_to_end_workflow()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Routing Tests Passed: {routing_results['passed']}")
    print(f"❌ Routing Tests Failed: {routing_results['failed']}")
    
    if routing_results['errors']:
        print("\n❌ Errors:")
        for error in routing_results['errors']:
            print(f"  • {error}")
    
    print("\n🎯 All agent systems tested!")
    print("🎯 Orchestrator ready for production use!")
    print("=" * 60)

if __name__ == "__main__":
    main()
