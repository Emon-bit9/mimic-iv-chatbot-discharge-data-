#!/usr/bin/env python3
"""
Medical Chatbot Deployment Helper Script
Automates deployment to various platforms
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(command, check=True):
    """Run a shell command and return the result"""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if check and result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)
    
    return result

def check_requirements():
    """Check if all required files exist"""
    required_files = [
        'app.py',
        'simple_web_app.py', 
        'simple_chatbot.py',
        'requirements_deploy.txt',
        'README.md'
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"Error: Missing required files: {missing_files}")
        sys.exit(1)
    
    print("✅ All required files present")

def deploy_streamlit_cloud():
    """Deploy to Streamlit Cloud"""
    print("🚀 Deploying to Streamlit Cloud...")
    
    # Check if git repo exists
    if not Path('.git').exists():
        print("Initializing git repository...")
        run_command("git init")
        run_command("git add .")
        run_command('git commit -m "Initial commit for deployment"')
    
    # Check for remote
    result = run_command("git remote -v", check=False)
    if not result.stdout:
        repo_name = input("Enter your GitHub repository URL: ")
        run_command(f"git remote add origin {repo_name}")
    
    # Push to GitHub
    run_command("git add .")
    run_command('git commit -m "Deploy to Streamlit Cloud" || true', check=False)
    run_command("git push -u origin main")
    
    print("✅ Code pushed to GitHub")
    print("📋 Next steps:")
    print("1. Visit https://share.streamlit.io")
    print("2. Connect your GitHub account")
    print("3. Select your repository")
    print("4. Set main file as 'app.py'")
    print("5. Deploy!")

def deploy_heroku():
    """Deploy to Heroku"""
    print("🚀 Deploying to Heroku...")
    
    # Check if Heroku CLI is installed
    result = run_command("heroku --version", check=False)
    if result.returncode != 0:
        print("Error: Heroku CLI not installed")
        print("Install from: https://devcenter.heroku.com/articles/heroku-cli")
        sys.exit(1)
    
    # Login to Heroku
    print("Logging into Heroku...")
    run_command("heroku login")
    
    # Create app
    app_name = input("Enter Heroku app name (or press Enter for auto-generated): ")
    if app_name:
        run_command(f"heroku create {app_name}")
    else:
        run_command("heroku create")
    
    # Deploy
    run_command("git add .")
    run_command('git commit -m "Deploy to Heroku" || true', check=False)
    run_command("git push heroku main")
    
    print("✅ Deployed to Heroku!")
    run_command("heroku open")

def deploy_railway():
    """Deploy to Railway"""
    print("🚀 Deploying to Railway...")
    
    # Check if Railway CLI is installed
    result = run_command("railway --version", check=False)
    if result.returncode != 0:
        print("Installing Railway CLI...")
        run_command("npm install -g @railway/cli")
    
    # Login and deploy
    run_command("railway login")
    run_command("railway init")
    run_command("railway up")
    
    print("✅ Deployed to Railway!")

def test_local():
    """Test the application locally"""
    print("🧪 Testing application locally...")
    
    # Install requirements
    run_command("pip install -r requirements_deploy.txt")
    
    # Test import
    try:
        import streamlit
        print("✅ Streamlit installed successfully")
    except ImportError:
        print("❌ Error importing Streamlit")
        sys.exit(1)
    
    print("Starting local server...")
    print("Visit http://localhost:8501 to test")
    run_command("streamlit run app.py")

def create_env_template():
    """Create environment template file"""
    env_template = """# Environment Variables Template
# Copy this to .env and fill in your values

# Application Settings
DEBUG=False
APP_PASSWORD=your_secure_password_here

# Data Paths (for local deployment)
DATA_PATH=C:/Users/imran/Downloads/discharge.csv
MODEL_PATH=./trained_models

# Optional: Analytics and Monitoring
ANALYTICS_ENABLED=False
LOG_LEVEL=INFO

# Security Settings
RATE_LIMIT_ENABLED=True
MAX_REQUESTS_PER_MINUTE=10

# Deployment Platform (auto-detected)
PLATFORM=auto
"""
    
    with open('.env.template', 'w') as f:
        f.write(env_template)
    
    print("✅ Created .env.template file")
    print("📋 Copy this to .env and configure your settings")

def main():
    parser = argparse.ArgumentParser(description='Deploy Medical Chatbot')
    parser.add_argument('platform', choices=['streamlit', 'heroku', 'railway', 'test', 'env'], 
                       help='Deployment platform or action')
    parser.add_argument('--check', action='store_true', help='Check requirements only')
    
    args = parser.parse_args()
    
    print("🏥 Medical Chatbot Deployment Helper")
    print("=" * 40)
    
    # Always check requirements first
    check_requirements()
    
    if args.check:
        print("✅ Requirements check passed")
        return
    
    if args.platform == 'streamlit':
        deploy_streamlit_cloud()
    elif args.platform == 'heroku':
        deploy_heroku()
    elif args.platform == 'railway':
        deploy_railway()
    elif args.platform == 'test':
        test_local()
    elif args.platform == 'env':
        create_env_template()

if __name__ == '__main__':
    main() 