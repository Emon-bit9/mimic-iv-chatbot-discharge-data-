import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import time

# Configure the page
st.set_page_config(
    page_title="MIMIC-ICU Medical Chatbot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

class SimpleMedicalChatbot:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None
        self.search_df = None
        self.vectorizer = None
        self.tfidf_matrix = None
        
    def load_data(self, sample_size=15000):
        """Load and preprocess the discharge data"""
        try:
            # Load data with progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Loading CSV file...")
            progress_bar.progress(20)
            
            # Read the CSV file
            self.df = pd.read_csv(self.csv_path)
            progress_bar.progress(40)
            
            status_text.text("Processing data...")
            
            # Convert datetime columns
            self.df['charttime'] = pd.to_datetime(self.df['charttime'])
            self.df['storetime'] = pd.to_datetime(self.df['storetime'], errors='coerce')
            progress_bar.progress(60)
            
            # Use a sample for better performance
            if len(self.df) > sample_size:
                self.search_df = self.df.sample(n=sample_size, random_state=42).reset_index(drop=True)
                status_text.text(f"Using sample of {sample_size:,} notes for faster search...")
            else:
                self.search_df = self.df.copy()
            
            # Clean text data
            status_text.text("Cleaning text data...")
            self.search_df['cleaned_text'] = self.search_df['text'].apply(self.clean_text)
            progress_bar.progress(80)
            
            # Add text statistics
            self.df['text_length'] = self.df['text'].str.len()
            self.df['word_count'] = self.df['text'].str.split().str.len()
            
            progress_bar.progress(100)
            status_text.text("✅ Data loaded successfully!")
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            
            return True
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return False
    
    def clean_text(self, text):
        """Clean and preprocess medical text"""
        if pd.isna(text):
            return ""
        
        # Remove excessive whitespace and special characters
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-]', ' ', text)
        text = text.strip()
        
        return text
    
    def build_search_index(self):
        """Build TF-IDF search index"""
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Building search index...")
            progress_bar.progress(20)
            
            # Build TF-IDF vectorizer
            texts = self.search_df['cleaned_text'].tolist()
            
            self.vectorizer = TfidfVectorizer(
                max_features=8000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )
            
            progress_bar.progress(60)
            status_text.text("Creating search vectors...")
            
            self.tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            progress_bar.progress(100)
            status_text.text("✅ Search index ready!")
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            
            return True
            
        except Exception as e:
            st.error(f"Error building search index: {e}")
            return False
    
    def search(self, query, top_k=5):
        """Search for relevant discharge notes"""
        try:
            # Transform the query
            query_tfidf = self.vectorizer.transform([query])
            
            # Calculate cosine similarity
            similarities = cosine_similarity(query_tfidf, self.tfidf_matrix).flatten()
            
            # Get top results
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            results = []
            for i, idx in enumerate(top_indices):
                if similarities[idx] > 0:  # Only include results with some similarity
                    row = self.search_df.iloc[idx]
                    results.append({
                        'rank': i + 1,
                        'score': float(similarities[idx]),
                        'note_id': row['note_id'],
                        'subject_id': row['subject_id'],
                        'hadm_id': row['hadm_id'],
                        'charttime': row['charttime'],
                        'text': row['text'],
                        'preview': row['text'][:400] + "..." if len(row['text']) > 400 else row['text']
                    })
            
            return results
            
        except Exception as e:
            st.error(f"Error in search: {e}")
            return []
    
    def search_by_keywords(self, keywords, top_k=5):
        """Simple keyword-based search"""
        try:
            # Create a regex pattern for keywords
            pattern = '|'.join([re.escape(keyword.lower()) for keyword in keywords])
            
            # Search in text
            mask = self.search_df['cleaned_text'].str.lower().str.contains(pattern, na=False)
            results_df = self.search_df[mask].copy()
            
            if len(results_df) == 0:
                return []
            
            # Calculate relevance score based on keyword frequency
            results_df['relevance'] = results_df['cleaned_text'].str.lower().str.count(pattern)
            results_df = results_df.sort_values('relevance', ascending=False).head(top_k)
            
            results = []
            for i, (_, row) in enumerate(results_df.iterrows()):
                results.append({
                    'rank': i + 1,
                    'score': row['relevance'],
                    'note_id': row['note_id'],
                    'subject_id': row['subject_id'],
                    'hadm_id': row['hadm_id'],
                    'charttime': row['charttime'],
                    'text': row['text'],
                    'preview': row['text'][:400] + "..." if len(row['text']) > 400 else row['text']
                })
            
            return results
            
        except Exception as e:
            st.error(f"Error in keyword search: {e}")
            return []
    
    def search_by_patient(self, subject_id):
        """Get all notes for a specific patient"""
        try:
            patient_notes = self.df[self.df['subject_id'] == subject_id].copy()
            patient_notes = patient_notes.sort_values('charttime')
            
            results = []
            for _, row in patient_notes.iterrows():
                results.append({
                    'note_id': row['note_id'],
                    'hadm_id': row['hadm_id'],
                    'charttime': row['charttime'],
                    'note_type': row['note_type'],
                    'text': row['text'],
                    'preview': row['text'][:300] + "..." if len(row['text']) > 300 else row['text']
                })
            
            return results
            
        except Exception as e:
            st.error(f"Error in patient search: {e}")
            return []
    
    def get_statistics(self):
        """Get basic statistics about the dataset"""
        if self.df is None:
            return {}
        
        stats = {
            'total_notes': len(self.df),
            'search_notes': len(self.search_df) if self.search_df is not None else 0,
            'unique_patients': self.df['subject_id'].nunique(),
            'unique_admissions': self.df['hadm_id'].nunique(),
            'date_range': f"{self.df['charttime'].min().strftime('%Y-%m-%d')} to {self.df['charttime'].max().strftime('%Y-%m-%d')}",
            'avg_text_length': self.df['text_length'].mean(),
            'note_types': self.df['note_type'].value_counts().to_dict()
        }
        
        return stats

# Initialize session state
if 'chatbot' not in st.session_state:
    csv_path = "C:/Users/imran/Downloads/discharge.csv"
    st.session_state.chatbot = SimpleMedicalChatbot(csv_path)
    st.session_state.data_loaded = False
    st.session_state.index_built = False

def main():
    # Header
    st.title("🏥 MIMIC-ICU Medical Chatbot")
    st.markdown("### Search and analyze patient discharge notes")
    
    chatbot = st.session_state.chatbot
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Setup")
        
        # Step 1: Load Data
        if not st.session_state.data_loaded:
            st.markdown("**Step 1: Load Data**")
            if st.button("📁 Load Discharge Data", type="primary"):
                if chatbot.load_data():
                    st.session_state.data_loaded = True
                    st.rerun()
        else:
            st.success("✅ Data loaded")
        
        # Step 2: Build Index
        if st.session_state.data_loaded and not st.session_state.index_built:
            st.markdown("**Step 2: Build Search Index**")
            if st.button("🔍 Build Search Index", type="primary"):
                if chatbot.build_search_index():
                    st.session_state.index_built = True
                    st.rerun()
        elif st.session_state.index_built:
            st.success("✅ Search ready")
        
        # Statistics
        if st.session_state.data_loaded:
            st.markdown("---")
            st.subheader("📊 Dataset Info")
            stats = chatbot.get_statistics()
            st.metric("Total Notes", f"{stats['total_notes']:,}")
            st.metric("Unique Patients", f"{stats['unique_patients']:,}")
            st.metric("Search Index Size", f"{stats['search_notes']:,}")
            st.caption(f"Date Range: {stats['date_range']}")
    
    # Main content
    if st.session_state.index_built:
        # Search interface
        st.subheader("🔍 Search Medical Records")
        
        # Search type selection
        search_type = st.radio(
            "Choose search method:",
            ["💬 Text Search", "🔤 Keyword Search", "👤 Patient Search"],
            horizontal=True
        )
        
        if search_type == "👤 Patient Search":
            # Patient search
            col1, col2 = st.columns([3, 1])
            with col1:
                patient_id = st.number_input(
                    "Enter Patient ID:", 
                    min_value=0, 
                    step=1,
                    help="Enter a numeric patient ID (subject_id)"
                )
            with col2:
                search_btn = st.button("🔍 Search", type="primary")
            
            if search_btn and patient_id > 0:
                with st.spinner("Searching patient records..."):
                    results = chatbot.search_by_patient(patient_id)
                
                if results:
                    st.success(f"Found {len(results)} notes for patient {patient_id}")
                    
                    for i, result in enumerate(results):
                        with st.expander(f"📄 Note {i+1}: {result['note_type']} ({result['charttime'].strftime('%Y-%m-%d')})"):
                            col1, col2 = st.columns([1, 3])
                            with col1:
                                st.write(f"**Note ID:** {result['note_id']}")
                                st.write(f"**Admission:** {result['hadm_id']}")
                                st.write(f"**Date:** {result['charttime'].strftime('%Y-%m-%d %H:%M')}")
                            with col2:
                                st.text_area("Content:", result['text'], height=200, key=f"patient_{i}")
                else:
                    st.warning(f"No records found for patient {patient_id}")
        
        else:
            # Text/Keyword search
            query = st.text_area(
                "Enter your search query:",
                height=100,
                placeholder="e.g., diabetes treatment, heart failure, surgical procedure..."
            )
            
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                num_results = st.selectbox("Results to show:", [3, 5, 10], index=1)
            with col2:
                search_btn = st.button("🔍 Search", type="primary")
            
            if search_btn and query:
                with st.spinner("Searching medical records..."):
                    if search_type == "💬 Text Search":
                        results = chatbot.search(query, num_results)
                    else:  # Keyword search
                        keywords = [k.strip() for k in query.replace(',', ' ').split()]
                        results = chatbot.search_by_keywords(keywords, num_results)
                
                if results:
                    st.success(f"Found {len(results)} relevant results")
                    
                    for result in results:
                        with st.expander(f"📄 Rank {result['rank']} - Score: {result['score']:.3f} - Patient {result['subject_id']}"):
                            col1, col2 = st.columns([1, 2])
                            
                            with col1:
                                st.write(f"**Patient ID:** {result['subject_id']}")
                                st.write(f"**Note ID:** {result['note_id']}")
                                st.write(f"**Admission:** {result['hadm_id']}")
                                st.write(f"**Date:** {result['charttime'].strftime('%Y-%m-%d')}")
                                st.write(f"**Relevance:** {result['score']:.3f}")
                            
                            with col2:
                                st.write("**Content Preview:**")
                                st.text_area("", result['preview'], height=150, key=f"preview_{result['note_id']}")
                                
                                if st.button(f"📖 Show Full Note", key=f"full_{result['note_id']}"):
                                    st.text_area("Full Content:", result['text'], height=400, key=f"fulltext_{result['note_id']}")
                else:
                    st.warning("No relevant results found. Try different search terms.")
            elif search_btn:
                st.warning("Please enter a search query.")
    
    elif st.session_state.data_loaded:
        st.info("👈 Please build the search index using the sidebar to start searching.")
    else:
        st.info("👈 Please load the data using the sidebar to get started.")
    
    # Help section
    with st.expander("ℹ️ How to use this chatbot"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Getting Started:**
            1. Click "Load Discharge Data" in sidebar
            2. Click "Build Search Index" when data is loaded
            3. Choose your search method and start exploring!
            
            **Search Types:**
            - **Text Search**: AI-powered semantic search
            - **Keyword Search**: Find exact keyword matches
            - **Patient Search**: View all notes for a patient
            """)
        
        with col2:
            st.markdown("""
            **Example Queries:**
            - "diabetes complications"
            - "heart failure treatment"
            - "post-surgical care"
            - "medication allergies"
            - "discharge instructions"
            
            **Tips:**
            - Use medical terminology for better results
            - Try different search methods for comprehensive results
            - Patient IDs are numeric values from the dataset
            """)

if __name__ == "__main__":
    main() 