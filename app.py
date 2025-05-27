import streamlit as st
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Medical Information Retrieval - MIMIC-IV",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Check if we're in deployment environment
IS_DEPLOYMENT = not os.path.exists("trained_models")

if IS_DEPLOYMENT:
    st.error("⚠️ **Deployment Notice**: This is a demo version. The full trained models are not available in this deployment due to size constraints.")
    st.info("📋 **What you can do**: Explore the interface, read the documentation, and understand the system architecture.")
    st.info("🔧 **For full functionality**: Clone the repository and run locally with your own MIMIC-IV dataset.")
    
    # Create a simple demo interface
    st.title("🏥 Medical Information Retrieval System - MIMIC-IV")
    st.markdown("### Demo Interface")
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); padding: 2rem; border-radius: 15px; margin: 1rem 0;">
    <h3>🎯 System Overview</h3>
    <p>This is an AI-powered medical information retrieval system designed to search through MIMIC-IV discharge notes using advanced NLP techniques.</p>
    
    <h4>🔬 Technical Features:</h4>
    <ul>
    <li><strong>TF-IDF Vectorization</strong>: Advanced text analysis with 5,000+ features</li>
    <li><strong>Cosine Similarity</strong>: Mathematical similarity scoring (0-100%)</li>
    <li><strong>Medical Knowledge Base</strong>: Extracted medical entities and patterns</li>
    <li><strong>Real-time Processing</strong>: Sub-second query response times</li>
    <li><strong>Confidence Scoring</strong>: Transparent relevance assessment</li>
    </ul>
    
    <h4>📊 Dataset Information:</h4>
    <ul>
    <li><strong>Source</strong>: MIMIC-IV (Medical Information Mart for Intensive Care IV)</li>
    <li><strong>Records</strong>: 10,000 anonymized discharge summaries</li>
    <li><strong>Content</strong>: ICU patient diagnoses, treatments, medications, procedures</li>
    <li><strong>Privacy</strong>: Fully de-identified medical data</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Demo search interface
    st.markdown("### 🔍 Demo Search Interface")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        demo_query = st.text_input(
            "Try a sample medical search:",
            placeholder="e.g., diabetes treatment, heart failure, pneumonia...",
            help="This is a demo interface - actual search requires local deployment"
        )
        
        if st.button("🔍 Demo Search", type="primary"):
            if demo_query:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%); padding: 1.5rem; border-radius: 12px; margin: 1rem 0; border-left: 4px solid #28a745;">
                <h4>Demo Results for: "{demo_query}"</h4>
                <p><strong>Note:</strong> This is a simulated response. Actual results would show:</p>
                <ul>
                <li>📋 <strong>Medical Records</strong>: Relevant discharge summaries</li>
                <li>🎯 <strong>Confidence Scores</strong>: 60-100% (HIGH), 30-59% (MEDIUM), 0-29% (LOW)</li>
                <li>👤 <strong>Subject IDs</strong>: Patient identifiers (e.g., 10127480)</li>
                <li>📄 <strong>Note IDs</strong>: Specific record identifiers (e.g., 27006969)</li>
                <li>🔬 <strong>Medical Content</strong>: Diagnoses, treatments, medications, procedures</li>
                </ul>
                <p><strong>Example High-Confidence Result:</strong></p>
                <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; font-family: monospace; font-size: 0.9rem;">
                Subject ID: 10127480 | Note ID: 27006969 | Confidence: 87.3%<br>
                <em>Discharge Summary: Patient admitted with acute myocardial infarction, treated with percutaneous coronary intervention and dual antiplatelet therapy. Echocardiogram showed reduced ejection fraction...</em>
                </div>
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("**Search Options**")
        st.selectbox("Results", [5, 10, 15, 20], disabled=True)
        st.checkbox("Detailed Output", disabled=True)
    
    # Documentation sections
    st.markdown("---")
    
    # Confidence scoring explanation
    st.markdown("### 📈 Confidence Scoring System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Mathematical Foundation:**
        - TF-IDF vectorization
        - Cosine similarity calculation
        - Score = cosine_similarity × 100%
        
        **Classification Thresholds:**
        - **60-100% (HIGH)**: Strong semantic overlap
        - **30-59% (MEDIUM)**: Moderate relevance  
        - **0-29% (LOW)**: Minimal matches
        """)
    
    with col2:
        st.markdown("""
        **Key Factors:**
        - Term rarity (rare medical terms score higher)
        - Term frequency in document
        - Query coverage percentage
        - Medical terminology specificity
        - Document length normalization
        """)
    
    # Deployment instructions
    st.markdown("### 🚀 Local Deployment Instructions")
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #fff8f0 0%, #fef3e2 100%); padding: 1.5rem; border-radius: 12px; border-left: 4px solid #fd7e14;">
    <h4>To run the full system locally:</h4>
    <ol>
    <li><strong>Clone Repository:</strong> Download all project files</li>
    <li><strong>Install Dependencies:</strong> <code>pip install -r requirements.txt</code></li>
    <li><strong>Prepare Dataset:</strong> Place MIMIC-IV discharge.csv in the specified path</li>
    <li><strong>Train Models:</strong> <code>python simple_train.py</code> (30-60 minutes)</li>
    <li><strong>Launch App:</strong> <code>streamlit run simple_web_app.py</code></li>
    </ol>
    
    <p><strong>Requirements:</strong></p>
    <ul>
    <li>Python 3.8+</li>
    <li>8GB+ RAM</li>
    <li>2GB+ free disk space</li>
    <li>MIMIC-IV dataset access</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Contact and links
    st.markdown("### 📞 Contact & Resources")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **📚 Documentation:**
        - Complete README
        - Technical specifications
        - API documentation
        - Deployment guides
        """)
    
    with col2:
        st.markdown("""
        **🔬 Research:**
        - MIMIC-IV dataset info
        - TF-IDF methodology
        - Medical NLP techniques
        - Confidence scoring
        """)
    
    with col3:
        st.markdown("""
        **⚠️ Disclaimer:**
        - Educational use only
        - Not for medical diagnosis
        - Research purposes
        - Consult healthcare professionals
        """)

else:
    # If models exist, redirect to simple_web_app.py
    st.info("✅ **Models detected!** For full functionality, please run:")
    st.code("streamlit run simple_web_app.py")
    st.info("This will give you access to the complete medical search functionality with trained models.")
    
    st.markdown("---")
    st.markdown("### 🔄 Alternative: Demo Mode")
    st.markdown("You can still explore the demo interface below to understand the system capabilities.")
    
    # Show demo interface anyway
    st.title("🏥 Medical Information Retrieval System - MIMIC-IV")
    st.markdown("### Demo Interface")
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); padding: 2rem; border-radius: 15px; margin: 1rem 0;">
    <h3>🎯 System Overview</h3>
    <p>This is an AI-powered medical information retrieval system designed to search through MIMIC-IV discharge notes using advanced NLP techniques.</p>
    
    <h4>🔬 Technical Features:</h4>
    <ul>
    <li><strong>TF-IDF Vectorization</strong>: Advanced text analysis with 5,000+ features</li>
    <li><strong>Cosine Similarity</strong>: Mathematical similarity scoring (0-100%)</li>
    <li><strong>Medical Knowledge Base</strong>: Extracted medical entities and patterns</li>
    <li><strong>Real-time Processing</strong>: Sub-second query response times</li>
    <li><strong>Confidence Scoring</strong>: Transparent relevance assessment</li>
    </ul>
    </div>
    """, unsafe_allow_html=True) 