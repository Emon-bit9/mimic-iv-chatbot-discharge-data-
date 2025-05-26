import pandas as pd
import re
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SimpleMedicalChatbot:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None
        self.vectorizer = None
        self.tfidf_matrix = None
        
    def load_data(self):
        """Load and preprocess the discharge data"""
        print("Loading discharge data...")
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
            
            print(f"✅ Successfully loaded {len(self.df):,} discharge notes")
            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
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
    
    def build_index(self, sample_size=20000):
        """Build TF-IDF search index"""
        print("Building search index...")
        try:
            # Use a sample for faster processing
            if len(self.df) > sample_size:
                print(f"Using a sample of {sample_size:,} notes for faster search")
                self.search_df = self.df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            else:
                self.search_df = self.df.copy()
            
            # Build TF-IDF vectorizer
            texts = self.search_df['cleaned_text'].tolist()
            
            self.vectorizer = TfidfVectorizer(
                max_features=10000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )
            
            self.tfidf_matrix = self.vectorizer.fit_transform(texts)
            print("✅ Search index built successfully")
            return True
        except Exception as e:
            print(f"❌ Error building index: {e}")
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
                        'relevance': similarities[idx]
                    })
            
            return results
        except Exception as e:
            print(f"❌ Error in search: {e}")
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
                    'relevance': row['relevance']
                })
            
            return results
        except Exception as e:
            print(f"❌ Error in keyword search: {e}")
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
                    'text': row['text']
                })
            
            return results
        except Exception as e:
            print(f"❌ Error in patient search: {e}")
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
    
    def display_results(self, results, max_text_length=300):
        """Display search results in a formatted way"""
        if not results:
            print("❌ No results found.")
            return
        
        print(f"\n🔍 Found {len(results)} relevant results:\n")
        print("=" * 80)
        
        for result in results:
            print(f"📄 Rank {result['rank']} | Score: {result['score']:.3f}")
            print(f"   Patient ID: {result['subject_id']} | Note ID: {result['note_id']}")
            print(f"   Admission: {result['hadm_id']} | Date: {result['charttime']}")
            print(f"   Content Preview:")
            
            # Show preview of text
            text = result['text']
            if len(text) > max_text_length:
                text = text[:max_text_length] + "..."
            
            # Indent the text
            for line in text.split('\n')[:5]:  # Show first 5 lines
                print(f"   {line}")
            
            print("-" * 80)
    
    def interactive_mode(self):
        """Run the chatbot in interactive mode"""
        print("\n🏥 MIMIC-ICU Discharge Notes Chatbot")
        print("=" * 50)
        
        while True:
            print("\nChoose an option:")
            print("1. Search by text query")
            print("2. Search by keywords")
            print("3. Search by patient ID")
            print("4. Show dataset statistics")
            print("5. Exit")
            
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == '1':
                query = input("\nEnter your search query: ").strip()
                if query:
                    num_results = input("Number of results (default 5): ").strip()
                    num_results = int(num_results) if num_results.isdigit() else 5
                    
                    print(f"\n🔍 Searching for: '{query}'...")
                    results = self.search(query, num_results)
                    self.display_results(results)
                else:
                    print("❌ Please enter a valid query.")
            
            elif choice == '2':
                keywords_input = input("\nEnter keywords (separated by commas): ").strip()
                if keywords_input:
                    keywords = [k.strip() for k in keywords_input.split(',')]
                    num_results = input("Number of results (default 5): ").strip()
                    num_results = int(num_results) if num_results.isdigit() else 5
                    
                    print(f"\n🔍 Searching for keywords: {keywords}...")
                    results = self.search_by_keywords(keywords, num_results)
                    self.display_results(results)
                else:
                    print("❌ Please enter valid keywords.")
            
            elif choice == '3':
                patient_id = input("\nEnter patient ID (subject_id): ").strip()
                if patient_id.isdigit():
                    patient_id = int(patient_id)
                    print(f"\n🔍 Searching for patient {patient_id}...")
                    results = self.search_by_patient(patient_id)
                    
                    if results:
                        print(f"\n📋 Found {len(results)} notes for patient {patient_id}:")
                        print("=" * 80)
                        
                        for i, result in enumerate(results):
                            print(f"\n📄 Note {i+1}: {result['note_type']}")
                            print(f"   Note ID: {result['note_id']}")
                            print(f"   Admission: {result['hadm_id']}")
                            print(f"   Date: {result['charttime']}")
                            print(f"   Content Preview:")
                            
                            # Show preview
                            text = result['text'][:300] + "..." if len(result['text']) > 300 else result['text']
                            for line in text.split('\n')[:3]:
                                print(f"   {line}")
                            print("-" * 80)
                    else:
                        print(f"❌ No notes found for patient {patient_id}")
                else:
                    print("❌ Please enter a valid patient ID (number).")
            
            elif choice == '4':
                print("\n📊 Dataset Statistics:")
                stats = self.get_statistics()
                print("=" * 50)
                print(f"📄 Total Notes: {stats['total_notes']:,}")
                print(f"👥 Unique Patients: {stats['unique_patients']:,}")
                print(f"🏥 Unique Admissions: {stats['unique_admissions']:,}")
                print(f"📅 Date Range: {stats['date_range']}")
                print(f"📝 Average Text Length: {stats['avg_text_length']:.0f} characters")
                print(f"\n📋 Note Types:")
                for note_type, count in stats['note_types'].items():
                    print(f"   {note_type}: {count:,}")
            
            elif choice == '5':
                print("\n👋 Thank you for using the MIMIC-ICU Chatbot!")
                break
            
            else:
                print("❌ Invalid choice. Please enter 1-5.")

def main():
    # Initialize the chatbot
    csv_path = "C:/Users/imran/Downloads/discharge.csv"
    chatbot = SimpleMedicalChatbot(csv_path)
    
    # Load data
    if not chatbot.load_data():
        print("❌ Failed to load data. Please check the file path.")
        return
    
    # Build search index
    if not chatbot.build_index():
        print("❌ Failed to build search index.")
        return
    
    # Show basic statistics
    stats = chatbot.get_statistics()
    print(f"\n📊 Dataset loaded successfully!")
    print(f"📄 Total discharge notes: {stats['total_notes']:,}")
    print(f"👥 Unique patients: {stats['unique_patients']:,}")
    
    # Start interactive mode
    chatbot.interactive_mode()

if __name__ == "__main__":
    main() 