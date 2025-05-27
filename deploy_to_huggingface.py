#!/usr/bin/env python3
"""
Automated Deployment Script for Medical Chatbot to Hugging Face Spaces
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

def run_command(command, cwd=None):
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(command, shell=True, cwd=cwd, 
                              capture_output=True, text=True, check=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr

def check_prerequisites():
    """Check if all prerequisites are met."""
    print("🔍 Checking prerequisites...")
    
    # Check if git is installed
    success, _ = run_command("git --version")
    if not success:
        print("❌ Git is not installed. Please install Git first.")
        return False
    
    # Check if git lfs is installed
    success, _ = run_command("git lfs version")
    if not success:
        print("❌ Git LFS is not installed. Please install Git LFS first.")
        print("   Install: https://git-lfs.github.io/")
        return False
    
    # Check if required files exist
    required_files = [
        "simple_web_app.py",
        "simple_chatbot.py", 
        "requirements.txt",
        "trained_models/chatbot_data.pkl",
        "trained_models/tfidf_matrix.pkl"
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            print(f"❌ Required file missing: {file}")
            return False
    
    print("✅ All prerequisites met!")
    return True

def create_deployment_folder():
    """Create and populate deployment folder."""
    print("📁 Creating deployment folder...")
    
    # Create deployment folder
    deploy_dir = "huggingface-deployment"
    if os.path.exists(deploy_dir):
        shutil.rmtree(deploy_dir)
    os.makedirs(deploy_dir)
    
    # Copy files
    files_to_copy = [
        ("simple_web_app.py", "app.py"),  # Rename for Hugging Face
        ("simple_chatbot.py", "simple_chatbot.py"),
        ("requirements.txt", "requirements.txt"),
        ("README.md", "README.md")
    ]
    
    for src, dst in files_to_copy:
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(deploy_dir, dst))
            print(f"   ✅ Copied {src} → {dst}")
    
    # Copy trained_models folder
    if os.path.exists("trained_models"):
        shutil.copytree("trained_models", os.path.join(deploy_dir, "trained_models"))
        print("   ✅ Copied trained_models/ folder")
    
    print(f"✅ Deployment folder created: {deploy_dir}/")
    return deploy_dir

def get_user_input():
    """Get Hugging Face Space details from user."""
    print("\n🚀 Hugging Face Space Configuration")
    print("=" * 50)
    
    username = input("Enter your Hugging Face username: ").strip()
    space_name = input("Enter your Space name (e.g., medical-chatbot-mimic-iv): ").strip()
    
    if not username or not space_name:
        print("❌ Username and Space name are required!")
        return None, None
    
    space_url = f"https://huggingface.co/spaces/{username}/{space_name}"
    print(f"\n📍 Your Space URL will be: {space_url}")
    
    confirm = input("\nIs this correct? (y/n): ").strip().lower()
    if confirm != 'y':
        return None, None
    
    return username, space_name

def setup_git_lfs(repo_dir):
    """Setup Git LFS for large files."""
    print("📦 Setting up Git LFS...")
    
    # Initialize Git LFS
    success, output = run_command("git lfs install", cwd=repo_dir)
    if not success:
        print(f"❌ Failed to initialize Git LFS: {output}")
        return False
    
    # Track large files
    lfs_commands = [
        "git lfs track '*.pkl'",
        "git lfs track 'trained_models/*.pkl'"
    ]
    
    for cmd in lfs_commands:
        success, output = run_command(cmd, cwd=repo_dir)
        if not success:
            print(f"❌ Failed to track files: {output}")
            return False
    
    # Add .gitattributes
    success, output = run_command("git add .gitattributes", cwd=repo_dir)
    if success:
        run_command("git commit -m 'Add Git LFS tracking for model files'", cwd=repo_dir)
    
    print("✅ Git LFS setup complete!")
    return True

def deploy_to_huggingface(username, space_name, deploy_dir):
    """Deploy to Hugging Face Spaces."""
    print("🚀 Deploying to Hugging Face Spaces...")
    
    repo_url = f"https://huggingface.co/spaces/{username}/{space_name}"
    repo_dir = f"{space_name}-repo"
    
    # Clone the repository
    print(f"   📥 Cloning repository: {repo_url}")
    success, output = run_command(f"git clone {repo_url} {repo_dir}")
    if not success:
        print(f"❌ Failed to clone repository: {output}")
        print("   Make sure you've created the Space on Hugging Face first!")
        return False
    
    # Copy deployment files to repo
    print("   📋 Copying files to repository...")
    for item in os.listdir(deploy_dir):
        src = os.path.join(deploy_dir, item)
        dst = os.path.join(repo_dir, item)
        if os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    
    # Setup Git LFS
    if not setup_git_lfs(repo_dir):
        return False
    
    # Add, commit, and push
    print("   📤 Pushing to Hugging Face...")
    commands = [
        "git add .",
        "git commit -m 'Deploy full medical chatbot with MIMIC-IV search functionality'",
        "git push origin main"
    ]
    
    for cmd in commands:
        print(f"   Running: {cmd}")
        success, output = run_command(cmd, cwd=repo_dir)
        if not success:
            print(f"❌ Failed: {output}")
            return False
    
    print("✅ Deployment successful!")
    print(f"🌐 Your chatbot is now live at: https://{username}-{space_name}.hf.space")
    return True

def main():
    """Main deployment function."""
    print("🏥 Medical Chatbot - Hugging Face Spaces Deployment")
    print("=" * 60)
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please fix the issues above.")
        return
    
    # Create deployment folder
    deploy_dir = create_deployment_folder()
    
    # Get user input
    username, space_name = get_user_input()
    if not username or not space_name:
        print("❌ Deployment cancelled.")
        return
    
    print("\n📋 Pre-deployment Checklist:")
    print("1. ✅ Have you created a Hugging Face account?")
    print("2. ✅ Have you created a new Space on Hugging Face?")
    print("3. ✅ Did you select 'Streamlit' as the SDK?")
    print("4. ✅ Is your Space name correct?")
    
    proceed = input("\nAll items checked? Proceed with deployment? (y/n): ").strip().lower()
    if proceed != 'y':
        print("❌ Deployment cancelled.")
        return
    
    # Deploy to Hugging Face
    success = deploy_to_huggingface(username, space_name, deploy_dir)
    
    if success:
        print("\n🎉 DEPLOYMENT SUCCESSFUL!")
        print("=" * 40)
        print(f"🌐 Live URL: https://{username}-{space_name}.hf.space")
        print("⏱️  Build time: 5-10 minutes")
        print("📊 Features: Full medical search with 10,000 records")
        print("\n📝 Next Steps:")
        print("1. Wait for build to complete")
        print("2. Test your live chatbot")
        print("3. Share the URL with users")
        print("4. Monitor usage in Hugging Face dashboard")
    else:
        print("\n❌ Deployment failed. Check the errors above.")
        print("💡 Common solutions:")
        print("   - Ensure you've created the Space on Hugging Face first")
        print("   - Check your internet connection")
        print("   - Verify your Hugging Face credentials")

if __name__ == "__main__":
    main() 