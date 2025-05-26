import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page config
st.set_page_config(
    page_title="MIMIC-ICU Chatbot",
    page_icon="🏥",
    layout="wide"
)

@st.cache_data
def load_data():
    """Load and cache the CSV data"""
    try:
        csv_path = "C:/Users/imran/Downloads/discharge.csv"
        # Load a sample for faster performance
        df = pd.read_csv(csv_path, nrows=10000)  # Start with 10K rows
        
        # Basic preprocessing
        df['charttime'] = pd.to_datetime(df['charttime'], errors='coerce')
        df['text_clean'] = df['text'].fillna('').astype(str)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def build_search_index(df):
    """Build and cache the search index"""
    try:
        # Clean text
        texts = df['text_clean'].tolist()
        
        # Build TF-IDF
        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.9
        )
        
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        return vectorizer, tfidf_matrix
    except Exception as e:
        st.error(f"Error building index: {e}")
        return None, None

def search_notes(query, df, vectorizer, tfidf_matrix, top_k=5):
    """Search for relevant notes"""
    try:
        # Transform query
        query_vec = vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
        
        # Get top results
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for i, idx in enumerate(top_indices):
            if similarities[idx] > 0:
                row = df.iloc[idx]
                results.append({
                    'rank': i + 1,
                    'score': similarities[idx],
                    'patient_id': row['subject_id'],
                    'note_id': row['note_id'],
                    'date': row['charttime'],
                    'text': row['text'][:500] + "..." if len(str(row['text'])) > 500 else str(row['text'])
                })
        
        return results
    except Exception as e:
        st.error(f"Search error: {e}")
        return []

def main():
    st.title("🏥 MIMIC-ICU Medical Chatbot")
    st.markdown("Search through discharge notes from the MIMIC-ICU dataset")
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
    
    if df is None:
        st.error("Failed to load data. Please check the file path.")
        return
    
    st.success(f"✅ Loaded {len(df):,} discharge notes")
    
    # Build search index
    with st.spinner("Building search index..."):
        vectorizer, tfidf_matrix = build_search_index(df)
    
    if vectorizer is None:
        st.error("Failed to build search index.")
        return
    
    st.success("✅ Search index ready!")
    
    # Sidebar stats
    with st.sidebar:
        st.header("📊 Dataset Info")
        st.metric("Total Notes", f"{len(df):,}")
        st.metric("Unique Patients", f"{df['subject_id'].nunique():,}")
        st.metric("Date Range", f"{df['charttime'].min().strftime('%Y-%m-%d')} to {df['charttime'].max().strftime('%Y-%m-%d')}")
    
    # Search interface
    st.header("🔍 Search Medical Records")
    
    # Search type
    search_type = st.radio("Search Method:", ["Text Search", "Patient Lookup"], horizontal=True)
    
    if search_type == "Text Search":
        # Text search
        query = st.text_area(
            "Enter your search query:",
            placeholder="e.g., diabetes, heart failure, surgery...",
            height=100
        )
        
        col1, col2 = st.columns([1, 3])
        with col1:
            num_results = st.selectbox("Results:", [3, 5, 10], index=1)
        
        if st.button("🔍 Search", type="primary") and query:
            with st.spinner("Searching..."):
                results = search_notes(query, df, vectorizer, tfidf_matrix, num_results)
            
            if results:
                st.success(f"Found {len(results)} relevant results")
                
                for result in results:
                    with st.expander(f"📄 Rank {result['rank']} - Patient {result['patient_id']} (Score: {result['score']:.3f})"):
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.write(f"**Patient ID:** {result['patient_id']}")
                            st.write(f"**Note ID:** {result['note_id']}")
                            st.write(f"**Date:** {result['date'].strftime('%Y-%m-%d') if pd.notna(result['date']) else 'N/A'}")
                            st.write(f"**Relevance:** {result['score']:.3f}")
                        
                        with col2:
                            st.text_area("Content:", result['text'], height=200, key=f"result_{result['note_id']}")
            else:
                st.warning("No results found. Try different search terms.")
    
    else:
        # Patient lookup
        patient_id = st.number_input("Enter Patient ID:", min_value=0, step=1)
        
        if st.button("🔍 Search Patient", type="primary") and patient_id > 0:
            patient_notes = df[df['subject_id'] == patient_id]
            
            if len(patient_notes) > 0:
                st.success(f"Found {len(patient_notes)} notes for patient {patient_id}")
                
                for i, (_, note) in enumerate(patient_notes.iterrows()):
                    with st.expander(f"📄 Note {i+1} - {note['charttime'].strftime('%Y-%m-%d') if pd.notna(note['charttime']) else 'No date'}"):
                        st.write(f"**Note ID:** {note['note_id']}")
                        st.write(f"**Date:** {note['charttime']}")
                        st.text_area("Content:", str(note['text'])[:1000] + "..." if len(str(note['text'])) > 1000 else str(note['text']), 
                                   height=300, key=f"patient_{i}")
            else:
                st.warning(f"No notes found for patient {patient_id}")
    
    # Help
    with st.expander("ℹ️ How to use"):
        st.markdown("""
        **Text Search:**
        - Enter medical terms like "diabetes", "heart failure", "surgery"
        - The system will find the most relevant discharge notes
        
        **Patient Lookup:**
        - Enter a patient ID number to see all their notes
        - Notes are sorted by date
        
        **Tips:**
        - Use specific medical terminology for better results
        - Try different search terms if you don't find what you're looking for
        """)

if __name__ == "__main__":
    main() 