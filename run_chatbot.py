#!/usr/bin/env python3
"""
Simple startup script for the MIMIC-ICU Medical Chatbot
"""
import subprocess
import sys
import os

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'pandas',
        'numpy', 
        'sklearn',
        'streamlit'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is missing")
    
    return missing_packages

def install_packages(packages):
    """Install missing packages"""
    if packages:
        print(f"\n📦 Installing missing packages: {', '.join(packages)}")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "-r", "requirements_simple.txt"
            ])
            print("✅ Packages installed successfully!")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install packages")
            return False
    return True

def check_data_file():
    """Check if the CSV data file exists"""
    csv_path = "C:/Users/imran/Downloads/discharge.csv"
    if os.path.exists(csv_path):
        print(f"✅ Data file found: {csv_path}")
        return True
    else:
        print(f"❌ Data file not found: {csv_path}")
        print("Please ensure the discharge.csv file is in the correct location")
        return False

def run_chatbot():
    """Launch the Streamlit chatbot"""
    print("\n🚀 Starting MIMIC-ICU Medical Chatbot...")
    print("The web interface will open in your browser automatically.")
    print("Press Ctrl+C to stop the server.\n")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "simple_web_chatbot.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Chatbot stopped. Thank you for using MIMIC-ICU Medical Chatbot!")
    except Exception as e:
        print(f"❌ Error running chatbot: {e}")

def main():
    print("🏥 MIMIC-ICU Medical Chatbot Launcher")
    print("=" * 50)
    
    # Check dependencies
    print("\n🔍 Checking dependencies...")
    missing = check_dependencies()
    
    if missing:
        install_success = install_packages(missing)
        if not install_success:
            print("\n❌ Cannot proceed without required packages.")
            print("Please install them manually using:")
            print("pip install -r requirements_simple.txt")
            return
    
    # Check data file
    print("\n📁 Checking data file...")
    if not check_data_file():
        print("\n❌ Cannot proceed without the data file.")
        print("Please ensure discharge.csv is in the correct location.")
        return
    
    # All checks passed
    print("\n✅ All checks passed!")
    
    # Ask user if they want to proceed
    response = input("\n🚀 Ready to launch the chatbot? (y/n): ").lower().strip()
    
    if response in ['y', 'yes', '']:
        run_chatbot()
    else:
        print("👋 Setup complete. Run this script again when ready to launch.")

if __name__ == "__main__":
    main() 