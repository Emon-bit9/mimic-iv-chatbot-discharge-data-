import pandas as pd
import numpy as np
import pickle
import os
import re
import gc
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
import warnings
warnings.filterwarnings('ignore')

class OptimizedMedicalChatbotTrainer:
    def __init__(self, data_path="C:/Users/imran/Downloads/discharge.csv", sample_size=50000):
        self.data_path = data_path
        self.sample_size = sample_size  # Process subset for faster training
        self.models_dir = "trained_models"
        self.vectorizer = None
        self.tfidf_matrix = None
        self.word2vec_model = None
        self.svd_model = None
        self.kmeans_model = None
        self.scaler = None
        self.processed_data = None
        self.medical_entities = None
        
        # Create models directory
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Download required NLTK data
        self._download_nltk_data()
        
        # Initialize NLP tools
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def _download_nltk_data(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
            
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
    
    def load_and_preprocess_data(self):
        """Load and preprocess the discharge data with memory optimization"""
        print("Loading discharge data...")
        
        try:
            # Load data in chunks to manage memory
            print(f"Loading first {self.sample_size:,} records for efficient training...")
            df = pd.read_csv(self.data_path, nrows=self.sample_size)
            print(f"Loaded {len(df)} discharge records")
            
            # Clean and preprocess text
            print("Cleaning text data...")
            df['text_clean'] = df['text'].apply(self._clean_text)
            
            print("Processing text for NLP...")
            df['text_processed'] = df['text_clean'].apply(self._preprocess_text)
            
            # Extract medical entities and keywords
            print("Extracting medical keywords...")
            df['medical_keywords'] = df['text_clean'].apply(self._extract_medical_keywords)
            
            # Remove empty records
            df = df[df['text_processed'].str.len() > 10]
            
            # Force garbage collection
            gc.collect()
            
            self.processed_data = df
            print(f"Preprocessed {len(df)} valid records")
            
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def _clean_text(self, text):
        """Clean medical text data"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters but keep medical abbreviations
        text = re.sub(r'[^\w\s\-\.]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove very short words (less than 2 characters)
        text = ' '.join([word for word in text.split() if len(word) >= 2])
        
        return text.strip()
    
    def _preprocess_text(self, text):
        """Advanced text preprocessing for NLP"""
        if not text:
            return ""
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        processed_tokens = []
        for token in tokens:
            if token not in self.stop_words and len(token) > 2:
                lemmatized = self.lemmatizer.lemmatize(token)
                processed_tokens.append(lemmatized)
        
        return ' '.join(processed_tokens)
    
    def _extract_medical_keywords(self, text):
        """Extract medical keywords and entities"""
        medical_terms = [
            'diagnosis', 'treatment', 'medication', 'procedure', 'surgery',
            'patient', 'hospital', 'discharge', 'admission', 'condition',
            'symptoms', 'vital', 'blood', 'pressure', 'heart', 'lung',
            'kidney', 'liver', 'brain', 'infection', 'diabetes', 'hypertension'
        ]
        
        found_terms = []
        for term in medical_terms:
            if term in text:
                found_terms.append(term)
        
        return ' '.join(found_terms)
    
    def train_tfidf_model(self):
        """Train TF-IDF vectorizer with memory optimization"""
        print("Training TF-IDF model...")
        
        # Configure TF-IDF with reduced features for memory efficiency
        self.vectorizer = TfidfVectorizer(
            max_features=10000,  # Reduced from 15000
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2),  # Reduced from (1, 3)
            stop_words='english',
            sublinear_tf=True
        )
        
        # Fit and transform the processed text
        texts = self.processed_data['text_processed'].tolist()
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        
        print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        
        # Save the model
        with open(f"{self.models_dir}/tfidf_vectorizer.pkl", 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        # Save the matrix (sparse format)
        with open(f"{self.models_dir}/tfidf_matrix.pkl", 'wb') as f:
            pickle.dump(self.tfidf_matrix, f)
        
        # Force garbage collection
        gc.collect()
    
    def train_word2vec_model(self):
        """Train Word2Vec model for semantic understanding"""
        print("Training Word2Vec model...")
        
        # Prepare sentences for Word2Vec
        sentences = []
        for text in self.processed_data['text_processed']:
            if text:
                sentences.append(text.split())
        
        # Train Word2Vec model with optimized parameters
        self.word2vec_model = Word2Vec(
            sentences=sentences,
            vector_size=100,  # Reduced from 200
            window=5,         # Reduced from 10
            min_count=2,      # Reduced from 3
            workers=2,        # Reduced workers
            epochs=5,         # Reduced from 10
            sg=1  # Skip-gram model
        )
        
        # Save the model
        self.word2vec_model.save(f"{self.models_dir}/word2vec_model.model")
        print(f"Word2Vec vocabulary size: {len(self.word2vec_model.wv.key_to_index)}")
        
        # Force garbage collection
        gc.collect()
    
    def train_dimensionality_reduction(self):
        """Train SVD for dimensionality reduction"""
        print("Training dimensionality reduction model...")
        
        # Apply SVD to reduce TF-IDF dimensions
        self.svd_model = TruncatedSVD(n_components=200, random_state=42)  # Reduced from 300
        tfidf_reduced = self.svd_model.fit_transform(self.tfidf_matrix)
        
        # Normalize the reduced features
        self.scaler = StandardScaler()
        tfidf_normalized = self.scaler.fit_transform(tfidf_reduced)
        
        # Save models
        with open(f"{self.models_dir}/svd_model.pkl", 'wb') as f:
            pickle.dump(self.svd_model, f)
        
        with open(f"{self.models_dir}/scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
        
        with open(f"{self.models_dir}/tfidf_reduced.pkl", 'wb') as f:
            pickle.dump(tfidf_normalized, f)
        
        print(f"Reduced dimensions: {tfidf_normalized.shape}")
        
        # Force garbage collection
        gc.collect()
        
        return tfidf_normalized
    
    def train_clustering_model(self, tfidf_reduced):
        """Train clustering model for document grouping"""
        print("Training clustering model...")
        
        # Train K-means clustering with reduced clusters
        self.kmeans_model = KMeans(n_clusters=25, random_state=42, n_init=5)  # Reduced from 50
        clusters = self.kmeans_model.fit_predict(tfidf_reduced)
        
        # Add cluster labels to data
        self.processed_data['cluster'] = clusters
        
        # Save clustering model
        with open(f"{self.models_dir}/kmeans_model.pkl", 'wb') as f:
            pickle.dump(self.kmeans_model, f)
        
        print(f"Created {len(set(clusters))} clusters")
        
        # Force garbage collection
        gc.collect()
    
    def create_medical_knowledge_base(self):
        """Create a medical knowledge base from the data"""
        print("Creating medical knowledge base...")
        
        # Extract common medical patterns
        medical_patterns = {
            'diagnoses': [],
            'medications': [],
            'procedures': [],
            'symptoms': []
        }
        
        # Common medical terms patterns
        diagnosis_patterns = r'\b(diagnosis|diagnosed|condition|disease|disorder)\b'
        medication_patterns = r'\b(medication|drug|prescribed|treatment|therapy)\b'
        procedure_patterns = r'\b(procedure|surgery|operation|intervention)\b'
        symptom_patterns = r'\b(symptom|complaint|pain|fever|nausea)\b'
        
        # Process in smaller batches
        for i, text in enumerate(self.processed_data['text_clean']):
            if i > 1000:  # Limit for memory efficiency
                break
                
            if re.search(diagnosis_patterns, text):
                medical_patterns['diagnoses'].append(text[:200])
            if re.search(medication_patterns, text):
                medical_patterns['medications'].append(text[:200])
            if re.search(procedure_patterns, text):
                medical_patterns['procedures'].append(text[:200])
            if re.search(symptom_patterns, text):
                medical_patterns['symptoms'].append(text[:200])
        
        # Save medical knowledge base
        with open(f"{self.models_dir}/medical_knowledge_base.pkl", 'wb') as f:
            pickle.dump(medical_patterns, f)
        
        self.medical_entities = medical_patterns
        print("Medical knowledge base created")
    
    def save_processed_data(self):
        """Save processed data for the chatbot"""
        print("Saving processed data...")
        
        # Save essential data for chatbot
        chatbot_data = {
            'texts': self.processed_data['text'].tolist(),
            'processed_texts': self.processed_data['text_processed'].tolist(),
            'subject_ids': self.processed_data['subject_id'].tolist() if 'subject_id' in self.processed_data.columns else None,
            'hadm_ids': self.processed_data['hadm_id'].tolist() if 'hadm_id' in self.processed_data.columns else None,
            'clusters': self.processed_data['cluster'].tolist(),
            'medical_keywords': self.processed_data['medical_keywords'].tolist()
        }
        
        with open(f"{self.models_dir}/chatbot_data.pkl", 'wb') as f:
            pickle.dump(chatbot_data, f)
        
        print(f"Saved {len(chatbot_data['texts'])} records for chatbot")
    
    def save_training_metadata(self):
        """Save training metadata"""
        metadata = {
            'training_date': datetime.now().isoformat(),
            'data_path': self.data_path,
            'sample_size': self.sample_size,
            'num_records': len(self.processed_data),
            'tfidf_features': self.tfidf_matrix.shape[1] if self.tfidf_matrix is not None else 0,
            'word2vec_vocab_size': len(self.word2vec_model.wv.key_to_index) if self.word2vec_model else 0,
            'svd_components': 200,
            'num_clusters': 25
        }
        
        with open(f"{self.models_dir}/training_metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        
        print("Training metadata saved")
    
    def train_complete_model(self):
        """Train the complete chatbot model with memory optimization"""
        print("Starting optimized model training...")
        print("=" * 50)
        print(f"Processing {self.sample_size:,} records for efficient training")
        print("=" * 50)
        
        # Load and preprocess data
        if self.load_and_preprocess_data() is None:
            print("Failed to load data. Exiting...")
            return False
        
        # Train all models
        self.train_tfidf_model()
        self.train_word2vec_model()
        tfidf_reduced = self.train_dimensionality_reduction()
        self.train_clustering_model(tfidf_reduced)
        self.create_medical_knowledge_base()
        
        # Save data and metadata
        self.save_processed_data()
        self.save_training_metadata()
        
        print("=" * 50)
        print("Model training completed successfully!")
        print(f"Models saved in: {self.models_dir}/")
        print("The chatbot is now ready to use without reloading the dataset.")
        print("=" * 50)
        
        return True

if __name__ == "__main__":
    # Use optimized trainer with sample size
    trainer = OptimizedMedicalChatbotTrainer(sample_size=50000)
    trainer.train_complete_model() 