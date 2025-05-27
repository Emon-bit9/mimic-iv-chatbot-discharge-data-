import streamlit as st
import os
import sys
from datetime import datetime
import pandas as pd
from simple_chatbot import SimpleMedicalChatbot
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Medical Information Retrieval - MIMIC-IV",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, attractive styling
st.markdown("""
<style>
    /* Main styling - minimal header and reduced whitespace */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 100%;
    }
    
    /* Reduce gaps between elements */
    .element-container {
        margin-bottom: 0.5rem !important;
    }
    
    /* Remove unnecessary margins and padding */
    .stMarkdown {
        margin-bottom: 0.5rem !important;
    }
    
    /* Compact metric cards */
    .metric-card {
        margin: 0.5rem 0 !important;
    }
    
    /* Reduce space around forms */
    .stForm {
        margin-bottom: 1rem !important;
    }
    
    /* Cards and containers */
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        text-align: center;
        border: 1px solid rgba(255,255,255,0.2);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .response-container {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        color: #2c3e50;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border: 1px solid #dee2e6;
    }
    
    .result-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        border-left: 4px solid #28a745;
        border: 1px solid #e9ecef;
    }
    
    .patient-info {
        background: linear-gradient(90deg, #17a2b8, #138496);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .confidence-badge {
        background: #28a745;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    
    .confidence-low { background: #6c757d; color: white; }
    .confidence-medium { background: #fd7e14; color: white; }
    .confidence-high { background: #20c997; color: white; }
    
    /* Search styling */
    .search-container {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 2.5rem;
        border-radius: 15px;
        margin: 1rem 0 2rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 2px solid #e9ecef;
    }
    
    /* Sidebar styling */
    .sidebar-card {
        background: linear-gradient(135deg, #495057 0%, #343a40 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fff8f0 0%, #fef3e2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #fd7e14;
        margin: 1rem 0;
        border: 1px solid #fde2c7;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_chatbot():
    """Load the chatbot model (cached for performance)"""
    try:
        return SimpleMedicalChatbot()
    except Exception as e:
        st.error(f"Error loading chatbot: {e}")
        st.error("Please run 'python simple_train.py' first to train the models.")
        return None

def display_model_info(chatbot):
    """Display model information in sidebar"""
    if chatbot and chatbot.training_metadata:
        metadata = chatbot.training_metadata
        
        st.sidebar.markdown("""
        <div class="sidebar-card">
        <h3>Model Information</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.sidebar.markdown(f"**Training Date:** {metadata['training_date'][:10]}")
        st.sidebar.markdown(f"**Total Records:** {metadata['num_records']:,}")
        st.sidebar.markdown(f"**TF-IDF Features:** {metadata['tfidf_features']:,}")
        st.sidebar.markdown(f"**Sample Size:** {metadata.get('sample_size', 'N/A'):,}")
        
        # Model techniques
        st.sidebar.markdown("""
        <div class="sidebar-card">
        <h3>AI Techniques</h3>
        </div>
        """, unsafe_allow_html=True)
        
        techniques = [
            "TF-IDF Vectorization",
            "Medical Knowledge Base", 
            "Patient Record Lookup",
            "Text Preprocessing",
            "Similarity Analysis"
        ]
        for technique in techniques:
            st.sidebar.markdown(f"• {technique}")
        
        # Confidence explanation
        st.sidebar.markdown("""
        <div class="sidebar-card">
        <h3>Confidence Guide</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.sidebar.markdown("""
        **Confidence Score Methodology:**
        
        **Mathematical Basis:**
        - TF-IDF (Term Frequency-Inverse Document Frequency) vectorization
        - Cosine similarity calculation between query and medical records
        - Score = cosine_similarity × 100%
        
        **Classification Thresholds:**
        
        **60-100% (HIGH Confidence)**
        - Strong semantic overlap between search terms and medical content
        - Multiple exact or closely related medical terms found
        - High term frequency in the specific document
        - Low inverse document frequency (rare/specific terms)
        
        **30-59% (MEDIUM Confidence)**  
        - Moderate term overlap with medical record
        - Some relevant medical terminology present
        - Partial matches with search concepts
        - General medical relevance confirmed
        
        **0-29% (LOW Confidence)**
        - Minimal direct term matches
        - Broad medical context relevance
        - Common medical terms with high frequency across corpus
        - General healthcare content without specific matches
        
        **Key Factors Influencing Score:**
        - Term rarity (rare medical terms = higher weight)
        - Term frequency in specific document
        - Number of matching query terms
        - Medical terminology specificity
        - Document length normalization
        
        **Tip**: Use specific medical terms (e.g., "myocardial infarction" vs "heart problem") for higher confidence scores.
        """)

def display_statistics(chatbot):
    """Display dataset statistics"""
    if not chatbot or not chatbot.training_metadata:
        return
    
    metadata = chatbot.training_metadata
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #1f77b4; margin: 0;">Total Records</h3>
            <h2 style="margin: 0;">{:,}</h2>
        </div>
        """.format(metadata['num_records']), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #ff7f0e; margin: 0;">TF-IDF Features</h3>
            <h2 style="margin: 0;">{:,}</h2>
        </div>
        """.format(metadata['tfidf_features']), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #2ca02c; margin: 0;">Sample Size</h3>
            <h2 style="margin: 0;">{:,}</h2>
        </div>
        """.format(metadata.get('sample_size', 0)), unsafe_allow_html=True)

def display_search_examples_sidebar():
    """Display search examples in sidebar"""
    st.sidebar.markdown("""
    <div class="sidebar-card">
    <h3>Quick Search</h3>
    <p>Click any example to search instantly</p>
    </div>
    """, unsafe_allow_html=True)
    
    examples = {
        "Medical Conditions": [
            "diabetes mellitus treatment",
            "congestive heart failure", 
            "pneumonia diagnosis",
            "hypertension management",
            "sepsis treatment"
        ],
        "Procedures": [
            "cardiac catheterization",
            "mechanical ventilation",
            "hemodialysis",
            "coronary artery bypass",
            "endotracheal intubation"
        ],
        "Medications": [
            "insulin therapy",
            "antibiotic treatment",
            "pain management",
            "blood pressure medication",
            "vasopressor therapy"
        ],
        "Diagnostics": [
            "chest x-ray findings",
            "blood test results",
            "echocardiogram",
            "CT scan abdomen",
            "laboratory values"
        ]
    }
    
    for category, items in examples.items():
        st.sidebar.markdown(f"**{category}:**")
        for item in items:
            if st.sidebar.button(f"{item}", key=f"sidebar_{category}_{item}", use_container_width=True):
                st.session_state.search_query = item
                st.rerun()
        st.sidebar.markdown("")

def main():
    """Main application function"""
    
    # No header - direct to content
    
    # Load chatbot
    chatbot = load_chatbot()
    
    if not chatbot:
        st.error("Chatbot could not be loaded. Please check the model files.")
        st.info("To train the model, run: `python simple_train.py`")
        return
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page:", ["Search Interface", "About"])
    
    # Add cache clear button for debugging
    if st.sidebar.button("Reload Chatbot", help="Clear cache and reload the chatbot model"):
        st.cache_resource.clear()
        st.rerun()
    
    display_model_info(chatbot)
    display_search_examples_sidebar()
    
    if page == "Search Interface":
        display_chat_interface(chatbot)
    elif page == "About":
        display_about_page()

def display_chat_interface(chatbot):
    """Display the main chat interface"""
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'search_query' not in st.session_state:
        st.session_state.search_query = ""
    
    # Display statistics
    display_statistics(chatbot)
    
    # Search interface with modern styling
    st.markdown('<div class="search-container">', unsafe_allow_html=True)
    st.markdown("## Medical Search")
    
    # Search options
    col1, col2 = st.columns([3, 1])
    with col1:
        with st.form("search_form", clear_on_submit=True):
            query = st.text_input(
                "Enter your medical search query:",
                value=st.session_state.search_query,
                placeholder="e.g., diabetes treatment, heart failure, pneumonia, or subject ID...",
                help="Search through 10,000 medical records using AI-powered text analysis"
            )
            
            search_col1, search_col2, search_col3 = st.columns([2, 1, 1])
            with search_col1:
                search_button = st.form_submit_button("Search Medical Records", type="primary", use_container_width=True)
            with search_col2:
                clear_button = st.form_submit_button("Clear", use_container_width=True)
    
    with col2:
        st.markdown("**Search Options**")
        num_results = st.selectbox(
            "Number of Results",
            options=[3, 5, 10, 15, 20, 25, 30, 50],
            index=2,  # Default to 10
            help="Choose how many search results to display"
        )
        
        show_detailed = st.checkbox(
            "Show Detailed Output",
            value=False,
            help="Display full medical records instead of previews"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Clear search query after use
    if st.session_state.search_query:
        st.session_state.search_query = ""
    
    # Handle clear button
    if clear_button:
        st.session_state.chat_history = []
        st.rerun()
    
    # Handle search
    if search_button and query:
        with st.spinner("Analyzing medical records with AI..."):
            try:
                # Try advanced formatting first, fallback to basic if needed
                try:
                    # Get search results with custom parameters
                    results = chatbot.get_tfidf_similarity(chatbot.preprocess_query(query), num_results)
                    medical_knowledge = chatbot.search_medical_knowledge(query)
                    
                    # Format response with new options
                    if hasattr(chatbot, 'format_response_advanced'):
                        response = chatbot.format_response_advanced(query, results, medical_knowledge, show_detailed)
                    else:
                        # Fallback to basic format_response method
                        response = chatbot.format_response(query, results, medical_knowledge)
                except AttributeError:
                    # If advanced methods fail, use the basic generate_response
                    response = chatbot.generate_response(query)
                    results = []  # Set empty results for basic response
                
                # Add to chat history
                st.session_state.chat_history.append({
                    'timestamp': datetime.now(),
                    'query': query,
                    'response': response,
                    'num_results': num_results,
                    'detailed': show_detailed,
                    'results_data': results
                })
                
            except Exception as e:
                st.error(f"Error processing query: {e}")
                # Try to reload the chatbot if there's an attribute error
                if "has no attribute" in str(e):
                    st.info("Attempting to reload chatbot...")
                    st.cache_resource.clear()
                    st.rerun()
    
    # Display chat history with enhanced styling
    if st.session_state.chat_history:
        st.markdown("# Search Results")
        
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            # Create a more attractive header
            timestamp = chat['timestamp'].strftime('%H:%M:%S')
            num_results = chat.get('num_results', 3)
            detailed = chat.get('detailed', False)
            
            header = f"**{chat['query']}** | {num_results} results | {'Detailed' if detailed else 'Summary'} | {timestamp}"
            
            with st.expander(header, expanded=(i==0)):
                # Add explanation about the output
                if i == 0:  # Only show for the most recent search
                    st.markdown("""
                    <div class="info-box">
                    <h4>Understanding Your Results</h4>
                    <p><strong>What you're seeing:</strong> AI-powered search results from 10,000 real medical discharge notes from the MIMIC-IV database.</p>
                    <p><strong>Subject IDs:</strong> Patient identifiers and Note IDs for specific medical records in the database.</p>
                    <p><strong>Confidence Scores:</strong> Mathematical similarity scores (TF-IDF + cosine similarity) showing how well your search terms match the medical content. 60%+ = HIGH confidence, 30-59% = MEDIUM, 0-29% = LOW.</p>
                    <p><strong>Medical Records:</strong> Actual discharge summaries containing diagnoses, treatments, and patient information.</p>
                    <p><strong>Scoring Factors:</strong> Term rarity, frequency in document, medical specificity, and query coverage determine confidence levels.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Display the response
                st.markdown(f'<div class="response-container">{chat["response"]}</div>', unsafe_allow_html=True)
                
                # Add results summary
                if 'results_data' in chat and chat['results_data']:
                    results_data = chat['results_data']
                    st.markdown("### Results Summary")
                    
                    summary_col1, summary_col2, summary_col3 = st.columns(3)
                    with summary_col1:
                        avg_confidence = sum(r['similarity'] for r in results_data) / len(results_data) * 100
                        st.metric("Average Confidence", f"{avg_confidence:.1f}%")
                    with summary_col2:
                        st.metric("Total Results", len(results_data))
                    with summary_col3:
                        high_conf = sum(1 for r in results_data if r['similarity'] > 0.5)
                        st.metric("High Confidence Results", high_conf)

def display_about_page():
    """Display about page with system information"""
    
    st.markdown('<div style="text-align: center; margin-bottom: 2rem;"><h2 style="color: #2c3e50;">Medical AI System</h2><p style="color: #7f8c8d;">Advanced Medical Information Retrieval</p></div>', unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    <div class="info-box">
    <h3>What is this system?</h3>
    <p>This is an AI-powered medical information retrieval system that searches through <strong>10,000 real medical discharge notes</strong> 
    from the MIMIC-IV database. It uses advanced natural language processing to find relevant medical information based on your queries.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key capabilities
    st.markdown("## Key Capabilities")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
        <h3>Smart Search</h3>
        <p>TF-IDF powered text analysis finds the most relevant medical records for your query</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h3>Patient Lookup</h3>
        <p>Search specific patient records by ID or find patients with similar conditions</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="metric-card">
        <h3>Confidence Scoring</h3>
        <p>Every result includes a confidence score showing relevance to your search</p>
        </div>
        """, unsafe_allow_html=True)
    
    # How it works
    st.markdown("## How It Works")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-box">
        <h4>Data Processing Pipeline</h4>
        <ol>
        <li><strong>Text Preprocessing</strong>: Clean and normalize medical text</li>
        <li><strong>Feature Extraction</strong>: Extract medical keywords and concepts</li>
        <li><strong>TF-IDF Vectorization</strong>: Convert text to numerical vectors</li>
        <li><strong>Similarity Analysis</strong>: Find most relevant records</li>
        <li><strong>Response Generation</strong>: Format results with confidence scores</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="info-box">
        <h4>AI Techniques Used</h4>
        <ul>
        <li><strong>TF-IDF</strong>: Term frequency analysis for relevance</li>
        <li><strong>Cosine Similarity</strong>: Mathematical similarity measurement</li>
        <li><strong>Medical NLP</strong>: Healthcare-specific text processing</li>
        <li><strong>Knowledge Base</strong>: Extracted medical entities</li>
        <li><strong>Real-time Processing</strong>: Instant search results</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Dataset information
    st.markdown("## About the MIMIC-IV Dataset")
    
    st.markdown("""
    <div class="warning-box">
    <h4>What is MIMIC-IV?</h4>
    <p><strong>MIMIC-IV</strong> (Medical Information Mart for Intensive Care IV) is a large, freely-available database 
    comprising de-identified health-related data from patients who stayed in critical care units.</p>
    
    <p><strong>Our subset contains:</strong></p>
    <ul>
    <li><strong>10,000 discharge summaries</strong> from ICU patients</li>
    <li><strong>Fully anonymized</strong> patient data for privacy</li>
    <li><strong>Real medical cases</strong> with diagnoses, treatments, and outcomes</li>
    <li><strong>Comprehensive records</strong> including medications, procedures, and lab results</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Technical specifications
    st.markdown("## Technical Specifications")
    
    tech_col1, tech_col2 = st.columns(2)
    
    with tech_col1:
        st.markdown("""
        **Technology Stack:**
        - **Language**: Python 3.11+
        - **ML Framework**: scikit-learn
        - **Web Framework**: Streamlit
        - **Data Processing**: pandas, numpy
        - **Text Processing**: NLTK, regex
        """)
        
    with tech_col2:
        st.markdown("""
        **Performance Metrics:**
        - **Search Speed**: < 1 second per query
        - **Model Size**: 270MB total
        - **Memory Usage**: Optimized for efficiency
        - **Accuracy**: TF-IDF similarity matching
        - **Scalability**: Handles concurrent users
        """)
    
    # Confidence scoring explanation
    st.markdown("## Confidence Scoring System")
    
    st.markdown("""
    <div class="info-box">
    <h4>How Confidence Scores Work</h4>
    <p>Our system uses <strong>TF-IDF (Term Frequency-Inverse Document Frequency)</strong> combined with <strong>cosine similarity</strong> 
    to calculate mathematically precise confidence scores:</p>
    
    <p><strong>Formula:</strong> <code>Confidence = cosine_similarity(query_vector, document_vector) × 100%</code></p>
    
    <h5>Classification Thresholds:</h5>
    <ul>
    <li><strong>60-100% (HIGH):</strong> Strong semantic overlap, rare medical terms, multiple exact matches</li>
    <li><strong>30-59% (MEDIUM):</strong> Moderate overlap, some relevant terminology, partial matches</li>
    <li><strong>0-29% (LOW):</strong> Minimal matches, broad context, common medical terms</li>
    </ul>
    
    <h5>Key Factors Affecting Scores:</h5>
    <ol>
    <li><strong>Term Rarity:</strong> Rare medical terms (e.g., "pneumothorax") score higher than common ones (e.g., "patient")</li>
    <li><strong>Term Frequency:</strong> How often your search terms appear in the specific medical record</li>
    <li><strong>Query Coverage:</strong> Percentage of your search terms found in the document</li>
    <li><strong>Medical Specificity:</strong> Clinical terminology receives priority over general language</li>
    <li><strong>Document Normalization:</strong> Longer documents are normalized to prevent bias</li>
    </ol>
    
    <h5>Examples:</h5>
    <ul>
    <li><strong>HIGH Confidence:</strong> "myocardial infarction treatment" → exact medical terminology</li>
    <li><strong>MEDIUM Confidence:</strong> "heart problems" → general medical concept</li>
    <li><strong>LOW Confidence:</strong> "patient care" → very common, non-specific terms</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Usage tips
    st.markdown("## Usage Tips")
    
    st.markdown("""
    <div class="info-box">
    <h4>Getting Better Results</h4>
    <ul>
    <li><strong>Be Specific</strong>: Use medical terms like "myocardial infarction" instead of "heart attack"</li>
    <li><strong>Try Variations</strong>: Search for both "diabetes" and "diabetes mellitus"</li>
    <li><strong>Use Multiple Terms</strong>: "pneumonia treatment antibiotics" gives better results</li>
            <li><strong>Subject IDs</strong>: Search by specific subject ID for complete patient records</li>
    <li><strong>Adjust Results</strong>: Use 5-10 results for comprehensive searches</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("## Important Disclaimer")
    
    st.markdown("""
    <div class="warning-box">
    <h4>Medical Disclaimer</h4>
    <p><strong>This system is for educational and research purposes only.</strong></p>
    <ul>
    <li><strong>Not for medical diagnosis</strong> or treatment decisions</li>
    <li><strong>Not a substitute</strong> for professional medical advice</li>
    <li><strong>Historical data only</strong> - not current medical guidelines</li>
    <li><strong>Educational tool</strong> for understanding medical records</li>
    <li><strong>Research purposes</strong> and medical informatics study</li>
    </ul>
    <p><em>Always consult qualified healthcare professionals for medical advice.</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 