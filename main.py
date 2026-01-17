#!/usr/bin/env python3
"""
Email Inbox Organizer - Main Entry Point

This is the main entry point for the Email Inbox Organizer application.
Run this script to start the Streamlit application.

Usage:
    python main.py
    or
    streamlit run app.py
"""

import sys
import os
import subprocess

def check_requirements():
    """Check if all required packages are installed"""
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'python-dotenv', 
        'langchain', 'langchain-core', 'langchain-community', 
        'langchain-groq', 'langgraph', 'pydantic', 'tenacity', 
        'rich', 'plotly', 'scikit-learn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Install missing packages with:")
        print("   pip install -r requirements.txt")
        return False
    
    return True

def check_env_file():
    """Check if .env file exists or API key is configured"""
    if not os.path.exists('.env'):
        print("⚠️  No .env file found.")
        print("📝 Create a .env file with your Groq API key:")
        print("   cp .env.example .env")
        print("   Then edit .env and add your API key")
        return False
    
    return True

def main():
    """Main entry point"""
    print("🚀 Starting Email Inbox Organizer...")
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check environment
    if not check_env_file():
        print("\n⚠️  You can still run the app, but make sure to configure your API key in config.py")
    
    print("✅ All checks passed!")
    print("🌐 Starting Streamlit application...")
    print("📱 The app will open in your browser at http://localhost:8501")
    print("\n" + "="*50)
    
    # Run streamlit
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting Streamlit: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Shutting down Email Inbox Organizer...")
        sys.exit(0)

if __name__ == "__main__":
    main()
