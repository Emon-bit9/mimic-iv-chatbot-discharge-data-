import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import re
import pickle
import os
from datetime import datetime
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class MedicalChatbot:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None
        self.embeddings = None
        self.index = None
        self.model = None
        self.vectorizer = None
        self.tfidf_matrix = None
        
    def load_data(self):
        """Load and preprocess the discharge data"""
        try:
            self.df = pd.read_csv(self.csv_path)
            
            # Convert datetime columns
            self.df['charttime'] = pd.to_datetime(self.df['charttime'])
            self.df['storetime'] = pd.to_datetime(self.df['storetime'])
            
            # Clean and preprocess text
            self.df['cleaned_text'] = self.df['text'].apply(self.clean_text)
            
            # Create summary info for each note
            self.df['text_length'] = self.df['text'].str.len()
            self.df['word_count'] = self.df['text'].str.split().str.len()
            
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
    
    def build_search_index(self, sample_size=10000):
        """Build search index using sentence transformers"""
        try:
            # Use a sample for faster processing during development
            if len(self.df) > sample_size:
                sample_df = self.df.sample(n=sample_size, random_state=42)
            else:
                sample_df = self.df
            
            # Load sentence transformer model
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Create embeddings
            texts = sample_df['cleaned_text'].tolist()
            self.embeddings = self.model.encode(texts)
            
            # Build FAISS index
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(self.embeddings)
            self.index.add(self.embeddings)
            
            # Store the sample dataframe
            self.search_df = sample_df.reset_index(drop=True)
            
            return True
        except Exception as e:
            st.error(f"Error building search index: {e}")
            return False
    
    def build_tfidf_index(self):
        """Build TF-IDF index for keyword-based search"""
        try:
            # Use cleaned text for TF-IDF
            texts = self.search_df['cleaned_text'].tolist()
            
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )
            
            self.tfidf_matrix = self.vectorizer.fit_transform(texts)
            return True
        except Exception as e:
            st.error(f"Error building TF-IDF index: {e}")
            return False
    
    def semantic_search(self, query, top_k=5):
        """Perform semantic search using sentence transformers"""
        try:
            # Encode the query
            query_embedding = self.model.encode([query])
            faiss.normalize_L2(query_embedding)
            
            # Search in the index
            scores, indices = self.index.search(query_embedding, top_k)
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.search_df):
                    row = self.search_df.iloc[idx]
                    results.append({
                        'rank': i + 1,
                        'score': float(score),
                        'note_id': row['note_id'],
                        'subject_id': row['subject_id'],
                        'hadm_id': row['hadm_id'],
                        'charttime': row['charttime'],
                        'text': row['text'][:500] + "..." if len(row['text']) > 500 else row['text'],
                        'full_text': row['text']
                    })
            
            return results
        except Exception as e:
            st.error(f"Error in semantic search: {e}")
            return []
    
    def keyword_search(self, query, top_k=5):
        """Perform keyword-based search using TF-IDF"""
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
                        'text': row['text'][:500] + "..." if len(row['text']) > 500 else row['text'],
                        'full_text': row['text']
                    })
            
            return results
        except Exception as e:
            st.error(f"Error in keyword search: {e}")
            return []
    
    def get_statistics(self):
        """Get basic statistics about the dataset"""
        if self.df is None:
            return {}
        
        stats = {
            'total_notes': len(self.df),
            'unique_patients': self.df['subject_id'].nunique(),
            'unique_admissions': self.df['hadm_id'].nunique(),
            'date_range': f"{self.df['charttime'].min()} to {self.df['charttime'].max()}",
            'avg_text_length': self.df['text_length'].mean(),
            'note_types': self.df['note_type'].value_counts().to_dict()
        }
        
        return stats
    
    def search_by_patient(self, subject_id):
        """Get all notes for a specific patient"""
        if self.df is None:
            return []
        
        patient_notes = self.df[self.df['subject_id'] == subject_id].copy()
        patient_notes = patient_notes.sort_values('charttime')
        
        results = []
        for _, row in patient_notes.iterrows():
            results.append({
                'note_id': row['note_id'],
                'hadm_id': row['hadm_id'],
                'charttime': row['charttime'],
                'note_type': row['note_type'],
                'text': row['text'][:300] + "..." if len(row['text']) > 300 else row['text'],
                'full_text': row['text']
            })
        
        return results

def main():
    st.set_page_config(
        page_title="MIMIC-ICU Discharge Notes Chatbot",
        page_icon="🏥",
        layout="wide"
    )
    
    st.title("🏥 MIMIC-ICU Discharge Notes Chatbot")
    st.markdown("Ask questions about patient discharge notes from the MIMIC-ICU dataset")
    
    # Initialize the chatbot
    if 'chatbot' not in st.session_state:
        csv_path = "C:/Users/imran/Downloads/discharge.csv"
        st.session_state.chatbot = MedicalChatbot(csv_path)
        st.session_state.data_loaded = False
        st.session_state.index_built = False
    
    chatbot = st.session_state.chatbot
    
    # Sidebar for controls
    st.sidebar.header("Controls")
    
    # Data loading
    if not st.session_state.data_loaded:
        if st.sidebar.button("Load Data"):
            with st.spinner("Loading discharge data..."):
                if chatbot.load_data():
                    st.session_state.data_loaded = True
                    st.success("Data loaded successfully!")
                    st.experimental_rerun()
    
    # Index building
    if st.session_state.data_loaded and not st.session_state.index_built:
        if st.sidebar.button("Build Search Index"):
            with st.spinner("Building search index (this may take a few minutes)..."):
                if chatbot.build_search_index() and chatbot.build_tfidf_index():
                    st.session_state.index_built = True
                    st.success("Search index built successfully!")
                    st.experimental_rerun()
    
    # Display statistics
    if st.session_state.data_loaded:
        stats = chatbot.get_statistics()
        st.sidebar.subheader("Dataset Statistics")
        st.sidebar.write(f"📊 Total Notes: {stats['total_notes']:,}")
        st.sidebar.write(f"👥 Unique Patients: {stats['unique_patients']:,}")
        st.sidebar.write(f"🏥 Unique Admissions: {stats['unique_admissions']:,}")
        st.sidebar.write(f"📅 Date Range: {stats['date_range']}")
        st.sidebar.write(f"📝 Avg Text Length: {stats['avg_text_length']:.0f} chars")
    
    # Main interface
    if st.session_state.index_built:
        # Search options
        search_type = st.radio(
            "Choose search method:",
            ["Semantic Search", "Keyword Search", "Patient Search"]
        )
        
        if search_type == "Patient Search":
            # Patient-specific search
            col1, col2 = st.columns([3, 1])
            with col1:
                patient_id = st.number_input("Enter Patient ID (subject_id):", min_value=0, step=1)
            with col2:
                search_patient = st.button("Search Patient")
            
            if search_patient and patient_id > 0:
                with st.spinner("Searching for patient notes..."):
                    results = chatbot.search_by_patient(patient_id)
                    
                    if results:
                        st.success(f"Found {len(results)} notes for patient {patient_id}")
                        
                        for i, result in enumerate(results):
                            with st.expander(f"Note {i+1}: {result['note_type']} - {result['charttime']}"):
                                st.write(f"**Note ID:** {result['note_id']}")
                                st.write(f"**Admission ID:** {result['hadm_id']}")
                                st.write(f"**Chart Time:** {result['charttime']}")
                                st.write("**Content:**")
                                st.text_area("", result['full_text'], height=200, key=f"patient_note_{i}")
                    else:
                        st.warning(f"No notes found for patient {patient_id}")
        
        else:
            # Text-based search
            query = st.text_area("Enter your question or search query:", height=100)
            
            col1, col2 = st.columns([1, 4])
            with col1:
                num_results = st.selectbox("Number of results:", [3, 5, 10], index=1)
            
            if st.button("Search", type="primary"):
                if query:
                    with st.spinner(f"Performing {search_type.lower()}..."):
                        if search_type == "Semantic Search":
                            results = chatbot.semantic_search(query, num_results)
                        else:
                            results = chatbot.keyword_search(query, num_results)
                        
                        if results:
                            st.success(f"Found {len(results)} relevant results")
                            
                            for result in results:
                                with st.expander(f"Rank {result['rank']} (Score: {result['score']:.3f}) - Note {result['note_id']}"):
                                    col1, col2 = st.columns([1, 2])
                                    
                                    with col1:
                                        st.write(f"**Patient ID:** {result['subject_id']}")
                                        st.write(f"**Admission ID:** {result['hadm_id']}")
                                        st.write(f"**Chart Time:** {result['charttime']}")
                                        st.write(f"**Similarity Score:** {result['score']:.3f}")
                                    
                                    with col2:
                                        st.write("**Relevant Content:**")
                                        st.text_area("", result['text'], height=150, key=f"result_{result['note_id']}")
                                        
                                        if st.button(f"Show Full Note {result['rank']}", key=f"full_{result['note_id']}"):
                                            st.text_area("Full Note Content:", result['full_text'], height=400, key=f"full_text_{result['note_id']}")
                        else:
                            st.warning("No relevant results found. Try different keywords or search terms.")
                else:
                    st.warning("Please enter a search query.")
    
    elif st.session_state.data_loaded:
        st.info("👆 Please build the search index using the button in the sidebar to start searching.")
    else:
        st.info("👆 Please load the data using the button in the sidebar to get started.")
    
    # Instructions
    with st.expander("ℹ️ How to use this chatbot"):
        st.markdown("""
        ### Getting Started:
        1. **Load Data**: Click "Load Data" in the sidebar to load the discharge notes
        2. **Build Index**: Click "Build Search Index" to prepare the search functionality
        
        ### Search Methods:
        - **Semantic Search**: Uses AI to understand the meaning of your query
        - **Keyword Search**: Traditional text matching based on keywords
        - **Patient Search**: Find all notes for a specific patient ID
        
        ### Example Queries:
        - "diabetes complications"
        - "heart failure treatment"
        - "medication allergies"
        - "surgical procedures"
        - "discharge planning"
        
        ### Tips:
        - Use specific medical terms for better results
        - Try both semantic and keyword search for comprehensive results
        - Patient IDs are numerical values from the dataset
        """)

if __name__ == "__main__":
    main() 