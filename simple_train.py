import pandas as pd
import numpy as np
import pickle
import os
import re
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

class SimpleMedicalChatbotTrainer:
    def __init__(self, data_path="C:/Users/imran/Downloads/discharge.csv", sample_size=10000):
        self.data_path = data_path
        self.sample_size = sample_size
        self.models_dir = "trained_models"
        self.vectorizer = None
        self.tfidf_matrix = None
        self.processed_data = None
        
        # Create models directory
        os.makedirs(self.models_dir, exist_ok=True)
    
    def load_and_preprocess_data(self):
        """Load and preprocess the discharge data"""
        print("Loading discharge data...")
        
        try:
            # Load a sample of the data
            print(f"Loading first {self.sample_size:,} records...")
            df = pd.read_csv(self.data_path, nrows=self.sample_size)
            print(f"Loaded {len(df)} discharge records")
            
            # Basic text cleaning
            print("Cleaning text data...")
            df['text_clean'] = df['text'].apply(self._clean_text)
            
            # Remove empty records
            df = df[df['text_clean'].str.len() > 50]
            
            self.processed_data = df
            print(f"Preprocessed {len(df)} valid records")
            
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def _clean_text(self, text):
        """Basic text cleaning"""
        if pd.isna(text):
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def train_tfidf_model(self):
        """Train TF-IDF vectorizer"""
        print("Training TF-IDF model...")
        
        # Configure TF-IDF
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.8,
            stop_words='english'
        )
        
        # Fit and transform the text
        texts = self.processed_data['text_clean'].tolist()
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        
        print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        
        # Save the model
        with open(f"{self.models_dir}/tfidf_vectorizer.pkl", 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        # Save the matrix
        with open(f"{self.models_dir}/tfidf_matrix.pkl", 'wb') as f:
            pickle.dump(self.tfidf_matrix, f)
        
        print("TF-IDF model saved successfully")
    
    def create_medical_knowledge_base(self):
        """Create a simple medical knowledge base"""
        print("Creating medical knowledge base...")
        
        medical_patterns = {
            'diagnoses': [],
            'medications': [],
            'procedures': [],
            'symptoms': []
        }
        
        # Simple keyword matching
        for i, text in enumerate(self.processed_data['text_clean']):
            if i > 500:  # Limit for efficiency
                break
                
            if any(word in text for word in ['diagnosis', 'diagnosed', 'condition']):
                medical_patterns['diagnoses'].append(text[:200])
            if any(word in text for word in ['medication', 'drug', 'prescribed']):
                medical_patterns['medications'].append(text[:200])
            if any(word in text for word in ['procedure', 'surgery', 'operation']):
                medical_patterns['procedures'].append(text[:200])
            if any(word in text for word in ['symptom', 'pain', 'fever']):
                medical_patterns['symptoms'].append(text[:200])
        
        # Save medical knowledge base
        with open(f"{self.models_dir}/medical_knowledge_base.pkl", 'wb') as f:
            pickle.dump(medical_patterns, f)
        
        print("Medical knowledge base created")
    
    def save_processed_data(self):
        """Save processed data for the chatbot"""
        print("Saving processed data...")
        
        # Save essential data
        chatbot_data = {
            'texts': self.processed_data['text'].tolist(),
            'processed_texts': self.processed_data['text_clean'].tolist(),
            'subject_ids': self.processed_data['subject_id'].tolist() if 'subject_id' in self.processed_data.columns else None,
            'hadm_ids': self.processed_data['hadm_id'].tolist() if 'hadm_id' in self.processed_data.columns else None,
            'clusters': [0] * len(self.processed_data),  # Simple clustering
            'medical_keywords': ['medical'] * len(self.processed_data)  # Placeholder
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
            'word2vec_vocab_size': 0,  # Not using Word2Vec in simple version
            'svd_components': 0,
            'num_clusters': 1
        }
        
        with open(f"{self.models_dir}/training_metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        
        print("Training metadata saved")
    
    def create_dummy_models(self):
        """Create dummy models for compatibility"""
        print("Creating compatibility models...")
        
        # Create dummy Word2Vec model directory
        word2vec_dir = f"{self.models_dir}/word2vec_model.model"
        os.makedirs(word2vec_dir, exist_ok=True)
        
        # Create dummy files
        dummy_data = {'dummy': True}
        
        with open(f"{self.models_dir}/svd_model.pkl", 'wb') as f:
            pickle.dump(dummy_data, f)
        
        with open(f"{self.models_dir}/scaler.pkl", 'wb') as f:
            pickle.dump(dummy_data, f)
        
        with open(f"{self.models_dir}/tfidf_reduced.pkl", 'wb') as f:
            pickle.dump(np.random.rand(len(self.processed_data), 100), f)
        
        with open(f"{self.models_dir}/kmeans_model.pkl", 'wb') as f:
            pickle.dump(dummy_data, f)
        
        print("Compatibility models created")
    
    def train_complete_model(self):
        """Train the complete simple chatbot model"""
        print("Starting simple model training...")
        print("=" * 50)
        print(f"Processing {self.sample_size:,} records")
        print("=" * 50)
        
        # Load and preprocess data
        if self.load_and_preprocess_data() is None:
            print("Failed to load data. Exiting...")
            return False
        
        # Train models
        self.train_tfidf_model()
        self.create_medical_knowledge_base()
        self.create_dummy_models()
        
        # Save data and metadata
        self.save_processed_data()
        self.save_training_metadata()
        
        print("=" * 50)
        print("Simple model training completed successfully!")
        print(f"Models saved in: {self.models_dir}/")
        print("The chatbot is now ready to use!")
        print("=" * 50)
        
        return True

if __name__ == "__main__":
    trainer = SimpleMedicalChatbotTrainer(sample_size=10000)
    trainer.train_complete_model() 