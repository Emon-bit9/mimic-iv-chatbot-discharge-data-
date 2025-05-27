import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import re
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SimpleMedicalChatbot:
    def __init__(self, models_dir="trained_models"):
        self.models_dir = models_dir
        self.vectorizer = None
        self.tfidf_matrix = None
        self.chatbot_data = None
        self.medical_knowledge_base = None
        self.training_metadata = None
        
        # Load all trained models
        self.load_models()
    
    def load_models(self):
        """Load all pre-trained models"""
        try:
            print("Loading pre-trained models...")
            
            # Load TF-IDF vectorizer and matrix
            with open(f"{self.models_dir}/tfidf_vectorizer.pkl", 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            with open(f"{self.models_dir}/tfidf_matrix.pkl", 'rb') as f:
                self.tfidf_matrix = pickle.load(f)
            
            # Load processed data
            with open(f"{self.models_dir}/chatbot_data.pkl", 'rb') as f:
                self.chatbot_data = pickle.load(f)
            
            # Load medical knowledge base
            with open(f"{self.models_dir}/medical_knowledge_base.pkl", 'rb') as f:
                self.medical_knowledge_base = pickle.load(f)
            
            # Load training metadata
            with open(f"{self.models_dir}/training_metadata.pkl", 'rb') as f:
                self.training_metadata = pickle.load(f)
            
            print("All models loaded successfully!")
            print(f"Dataset contains {self.training_metadata['num_records']} medical records")
            print(f"Model trained on: {self.training_metadata['training_date']}")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Please run simple_train.py first to train the models.")
            raise
    
    def preprocess_query(self, query):
        """Preprocess user query"""
        if not query:
            return ""
        
        # Clean the query
        query = query.lower()
        query = re.sub(r'[^\w\s\-\.]', ' ', query)
        query = re.sub(r'\s+', ' ', query)
        
        return query.strip()
    
    def get_tfidf_similarity(self, query, top_k=5):
        """Get similar documents using TF-IDF cosine similarity"""
        # Transform query using trained vectorizer
        query_vector = self.vectorizer.transform([query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get valid indices only (within the range of available texts)
        max_valid_idx = len(self.chatbot_data['texts']) - 1
        valid_similarities = similarities[:max_valid_idx + 1]  # Only consider valid range
        
        # Get top k similar documents from valid range
        top_indices = valid_similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            # All indices are now guaranteed to be valid
            results.append({
                'index': idx,
                'similarity': similarities[idx],
                'text': self.chatbot_data['texts'][idx],
                'cluster': 0  # Simple clustering
            })
        
        return results
    
    def search_medical_knowledge(self, query):
        """Search in medical knowledge base"""
        results = []
        query_lower = query.lower()
        
        for category, texts in self.medical_knowledge_base.items():
            for text in texts[:10]:  # Limit search
                if any(word in text.lower() for word in query_lower.split()):
                    results.append({
                        'category': category,
                        'text': text,
                        'relevance': 'medical_knowledge'
                    })
        
        return results[:3]  # Return top 3 medical knowledge results
    
    def search_by_patient_id(self, patient_id):
        """Search for specific patient records by subject_id"""
        results = []
        
        if self.chatbot_data['subject_ids']:
            for i, subject_id in enumerate(self.chatbot_data['subject_ids']):
                if str(subject_id) == str(patient_id):
                    # Check if index is valid for texts
                    if 0 <= i < len(self.chatbot_data['texts']):
                        # Get note_id as well
                        note_id = "N/A"
                        if (self.chatbot_data['hadm_ids'] and 
                            i < len(self.chatbot_data['hadm_ids'])):
                            note_id = self.chatbot_data['hadm_ids'][i]
                        
                        results.append({
                            'index': i,
                            'similarity': 1.0,
                            'text': self.chatbot_data['texts'][i],
                            'subject_id': subject_id,
                            'note_id': note_id,
                            'type': 'patient_record'
                        })
        
        return results
    
    def generate_response(self, query):
        """Generate response using TF-IDF similarity"""
        if not query.strip():
            return "Please provide a medical question or search term."
        
        # Preprocess query
        processed_query = self.preprocess_query(query)
        
        # Check if it's a patient ID search
        if re.match(r'^\d+$', query.strip()):
            patient_results = self.search_by_patient_id(query.strip())
            if patient_results:
                return self.format_patient_response(patient_results)
        
        # Get results using TF-IDF
        tfidf_results = self.get_tfidf_similarity(processed_query, 5)
        medical_knowledge = self.search_medical_knowledge(processed_query)
        
        return self.format_response(query, tfidf_results, medical_knowledge)
    
    def format_response(self, query, results, medical_knowledge):
        """Format the final response"""
        if not results and not medical_knowledge:
            return f"I couldn't find specific information about '{query}' in the medical records. Please try rephrasing your question or use more specific medical terms."
        
        response = f"**Search Results for '{query}':**\n\n"
        
        # Add main results
        for i, result in enumerate(results, 1):
            text_preview = result['text'][:300] + "..." if len(result['text']) > 300 else result['text']
            confidence = result['similarity'] * 100
            
            # Get subject ID and note ID if available
            subject_id = "N/A"
            note_id = "N/A"
            try:
                idx = int(result['index'])  # Convert numpy int to regular int
                if (self.chatbot_data['subject_ids'] and 
                    0 <= idx < len(self.chatbot_data['subject_ids'])):
                    if self.chatbot_data['subject_ids'][idx]:
                        subject_id = self.chatbot_data['subject_ids'][idx]
                
                if (self.chatbot_data['hadm_ids'] and 
                    0 <= idx < len(self.chatbot_data['hadm_ids'])):
                    if self.chatbot_data['hadm_ids'][idx]:
                        note_id = self.chatbot_data['hadm_ids'][idx]
            except (KeyError, IndexError, TypeError, ValueError):
                subject_id = "N/A"
                note_id = "N/A"
            
            response += f"**Result {i}** | Subject ID: {subject_id} | Note ID: {note_id} | Confidence: {confidence:.1f}%\n"
            response += f"{text_preview}\n\n"
        
        # Add medical knowledge if available
        if medical_knowledge:
            response += "**Related Medical Information:**\n"
            for knowledge in medical_knowledge:
                response += f"• {knowledge['category'].title()}: {knowledge['text'][:200]}...\n"
        
        # Add confidence explanation
        response += "\n---\n"
        response += "**About Confidence Scores:** The confidence percentage shows how well your search terms match the medical record content. Higher percentages indicate stronger matches.\n\n"
        response += f"*Search completed using TF-IDF analysis on {self.training_metadata['num_records']:,} medical records.*"
        
        return response
    
    def format_response_advanced(self, query, results, medical_knowledge, show_detailed=False):
        """Format response with advanced options"""
        if not results and not medical_knowledge:
            return f"No specific information found for '{query}' in the medical records. Try different search terms or check spelling."
        
        response = f"## Search Results for '{query}'\n\n"
        
        # Add main results with enhanced formatting
        for i, result in enumerate(results, 1):
            confidence = result['similarity'] * 100
            
            # Get subject ID and note ID
            subject_id = "N/A"
            note_id = "N/A"
            try:
                idx = int(result['index'])  # Convert numpy int to regular int
                if (self.chatbot_data['subject_ids'] and 
                    0 <= idx < len(self.chatbot_data['subject_ids'])):
                    if self.chatbot_data['subject_ids'][idx]:
                        subject_id = self.chatbot_data['subject_ids'][idx]
                
                if (self.chatbot_data['hadm_ids'] and 
                    0 <= idx < len(self.chatbot_data['hadm_ids'])):
                    if self.chatbot_data['hadm_ids'][idx]:
                        note_id = self.chatbot_data['hadm_ids'][idx]
            except (KeyError, IndexError, TypeError, ValueError):
                subject_id = "N/A"
                note_id = "N/A"
            
            # Determine confidence level for styling
            if confidence >= 60:
                conf_class = "confidence-high"
                conf_indicator = "HIGH"
            elif confidence >= 30:
                conf_class = "confidence-medium" 
                conf_indicator = "MEDIUM"
            else:
                conf_class = "confidence-low"
                conf_indicator = "LOW"
            
            # Format text based on detailed option
            if show_detailed:
                text_content = result['text']  # Full text
                preview_note = ""
            else:
                text_content = result['text'][:400] + "..." if len(result['text']) > 400 else result['text']
                preview_note = "\n*Note: This is a preview. Enable 'Show Detailed Output' for full records.*"
            
            response += f"""
<div class="result-card">
<div class="patient-info">
Subject ID: {subject_id} | Note ID: {note_id} | Result {i} of {len(results)}
</div>
<div class="{conf_class} confidence-badge">
{conf_indicator} Confidence: {confidence:.1f}%
</div>

**Medical Record Content:**
{text_content}{preview_note}
</div>

"""
        
        # Add medical knowledge if available
        if medical_knowledge:
            response += """
### Related Medical Knowledge
"""
            for knowledge in medical_knowledge:
                response += f"• **{knowledge['category'].title()}**: {knowledge['text'][:150]}...\n"
        
        # Add comprehensive explanation
        response += f"""

---

### About These Results

**Search Method**: TF-IDF (Term Frequency-Inverse Document Frequency) analysis
**Database**: {self.training_metadata['num_records']:,} MIMIC-IV discharge notes
**Results Shown**: Top {len(results)} most relevant matches
**Processing**: Real-time AI-powered text analysis

**What are MIMIC-IV Records?**
These are real (anonymized) medical discharge summaries from intensive care unit patients. They contain:
- Patient demographics and admission details
- Medical diagnoses and conditions  
- Treatment procedures and medications
- Discharge instructions and follow-up care

**Confidence Score Methodology:**

**Mathematical Foundation:**
- **TF-IDF Analysis**: Term Frequency-Inverse Document Frequency vectorization
- **Cosine Similarity**: Mathematical similarity measurement (0-1 scale, converted to %)
- **Score Calculation**: cosine_similarity(query_vector, document_vector) × 100%

**Classification Thresholds:**
- **60-100% (HIGH)**: Strong semantic overlap, rare/specific medical terms, multiple exact matches
- **30-59% (MEDIUM)**: Moderate term overlap, some relevant medical terminology, partial concept matches  
- **0-29% (LOW)**: Minimal direct matches, broad medical context, common terms with high corpus frequency

**Key Scoring Factors:**
1. **Term Rarity**: Rare medical terms receive higher weights (lower IDF = higher importance)
2. **Term Frequency**: How often search terms appear in the specific medical record
3. **Query Coverage**: Percentage of search terms found in the document
4. **Medical Specificity**: Specialized terminology scores higher than general terms
5. **Document Normalization**: Longer documents are normalized to prevent length bias

**Examples of High vs Low Confidence:**
- HIGH: "pneumothorax chest tube" → exact medical procedure match
- MEDIUM: "breathing problems" → general respiratory terminology  
- LOW: "patient care" → very common, non-specific medical terms

*Tip: Use precise medical terminology (e.g., "myocardial infarction" vs "heart attack") for optimal results*
"""
        
        return response
    
    def format_patient_response(self, patient_results):
        """Format response for patient-specific searches"""
        if not patient_results:
            return "No records found for this subject ID."
        
        subject_id = patient_results[0]['subject_id']
        response = f"Found {len(patient_results)} record(s) for Subject ID {subject_id}:\n\n"
        
        for i, result in enumerate(patient_results, 1):
            text_preview = result['text'][:400] + "..." if len(result['text']) > 400 else result['text']
            note_id = result.get('note_id', 'N/A')
            response += f"**Record {i}** | Subject ID: {subject_id} | Note ID: {note_id}\n"
            response += f"{text_preview}\n\n"
        
        return response
    
    def get_model_info(self):
        """Get information about the loaded models"""
        if not self.training_metadata:
            return "Model information not available."
        
        info = f"""
Medical Chatbot Model Information:
- Training Date: {self.training_metadata['training_date']}
- Total Records: {self.training_metadata['num_records']:,}
- TF-IDF Features: {self.training_metadata['tfidf_features']:,}
- Sample Size: {self.training_metadata.get('sample_size', 'N/A'):,}

Techniques Implemented:
- TF-IDF Vectorization with cosine similarity
- Medical knowledge base extraction
- Patient-specific record lookup
- Text preprocessing and cleaning
        """
        return info.strip()

# Example usage and testing
if __name__ == "__main__":
    try:
        chatbot = SimpleMedicalChatbot()
        
        print("Medical Chatbot loaded successfully!")
        print(chatbot.get_model_info())
        print("\n" + "="*50)
        
        # Test queries
        test_queries = [
            "diabetes treatment",
            "heart failure symptoms",
            "blood pressure medication",
            "pneumonia diagnosis"
        ]
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            print("-" * 30)
            response = chatbot.generate_response(query)
            print(response[:200] + "..." if len(response) > 200 else response)
            print()
            
    except Exception as e:
        print(f"Error: {e}")
        print("Please run 'python simple_train.py' first to train the models.") 