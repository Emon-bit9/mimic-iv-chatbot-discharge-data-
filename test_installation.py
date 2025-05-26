"""
Test script to verify the chatbot installation and data loading
"""
import sys

def test_imports():
    """Test if all required packages can be imported"""
    print("🧪 Testing package imports...")
    
    try:
        import pandas as pd
        print("✅ pandas imported successfully")
    except ImportError as e:
        print(f"❌ Error importing pandas: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ numpy imported successfully")
    except ImportError as e:
        print(f"❌ Error importing numpy: {e}")
        return False
    
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        print("✅ scikit-learn imported successfully")
    except ImportError as e:
        print(f"❌ Error importing scikit-learn: {e}")
        return False
    
    try:
        import nltk
        print("✅ nltk imported successfully")
    except ImportError as e:
        print(f"❌ Error importing nltk: {e}")
        return False
    
    return True

def test_data_loading():
    """Test if the CSV file can be loaded"""
    print("\n📁 Testing data loading...")
    
    try:
        import pandas as pd
        csv_path = "C:/Users/imran/Downloads/discharge.csv"
        
        # Try to read just the first few rows
        df = pd.read_csv(csv_path, nrows=5)
        print(f"✅ CSV file loaded successfully")
        print(f"   Shape (first 5 rows): {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        return True
        
    except FileNotFoundError:
        print(f"❌ CSV file not found at: {csv_path}")
        print("   Please check the file path and ensure the file exists")
        return False
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return False

def test_advanced_imports():
    """Test advanced AI packages (optional)"""
    print("\n🤖 Testing advanced AI packages...")
    
    # Test sentence transformers (for advanced features)
    try:
        from sentence_transformers import SentenceTransformer
        print("✅ sentence-transformers available (semantic search enabled)")
        has_transformers = True
    except ImportError:
        print("⚠️  sentence-transformers not available (semantic search disabled)")
        has_transformers = False
    
    # Test FAISS (for fast similarity search)
    try:
        import faiss
        print("✅ faiss available (fast search enabled)")
        has_faiss = True
    except ImportError:
        print("⚠️  faiss not available (using slower search)")
        has_faiss = False
    
    # Test Streamlit (for web interface)
    try:
        import streamlit as st
        print("✅ streamlit available (web interface enabled)")
        has_streamlit = True
    except ImportError:
        print("⚠️  streamlit not available (web interface disabled)")
        has_streamlit = False
    
    return has_transformers, has_faiss, has_streamlit

def main():
    print("🏥 MIMIC-ICU Chatbot Installation Test")
    print("=" * 50)
    
    # Test basic imports
    basic_ok = test_imports()
    
    # Test data loading
    data_ok = test_data_loading()
    
    # Test advanced imports
    has_transformers, has_faiss, has_streamlit = test_advanced_imports()
    
    # Summary
    print("\n📋 Summary:")
    print("=" * 30)
    
    if basic_ok and data_ok:
        print("✅ Basic functionality: READY")
        print("   You can run: python simple_chatbot.py")
    else:
        print("❌ Basic functionality: NOT READY")
        print("   Please fix the errors above")
    
    if basic_ok and data_ok and has_streamlit:
        print("✅ Web interface: READY")
        print("   You can run: streamlit run chatbot.py")
    else:
        print("⚠️  Web interface: NEEDS SETUP")
        if not has_streamlit:
            print("   Install streamlit: pip install streamlit")
    
    if has_transformers and has_faiss:
        print("✅ Advanced AI features: ENABLED")
    else:
        print("⚠️  Advanced AI features: LIMITED")
        if not has_transformers:
            print("   Install sentence-transformers: pip install sentence-transformers")
        if not has_faiss:
            print("   Install faiss: pip install faiss-cpu")
    
    print("\n🚀 Ready to start? Try:")
    print("   python simple_chatbot.py    (for command-line interface)")
    if has_streamlit:
        print("   streamlit run chatbot.py    (for web interface)")

if __name__ == "__main__":
    main() 