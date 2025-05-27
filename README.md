# Medical Chatbot - MIMIC-IV Dataset

## Overview

This Medical Chatbot is an intelligent information retrieval system designed to analyze and search through MIMIC-IV discharge notes. The system employs advanced Natural Language Processing (NLP) and Machine Learning techniques to provide accurate and relevant medical information from a comprehensive dataset of intensive care unit discharge summaries.

## What is it

The Medical Chatbot is a sophisticated search and analysis tool that processes medical discharge notes to answer queries about patient conditions, treatments, medications, and procedures. It uses multiple machine learning approaches to understand medical terminology and provide contextually relevant responses from real clinical data.

## How it Works

### Data Processing Pipeline

1. **Text Preprocessing**: Raw medical text is cleaned, normalized, and prepared for analysis
2. **Medical Entity Extraction**: Identification and extraction of medical keywords, conditions, and terminology
3. **Feature Engineering**: Conversion of text data into numerical representations suitable for machine learning
4. **Model Training**: Multiple ML models are trained on the processed data
5. **Knowledge Base Creation**: Medical patterns and entities are organized into a searchable knowledge base

### Machine Learning Architecture

The system implements a multi-technique approach combining several state-of-the-art NLP methods:

#### 1. TF-IDF Vectorization
- **Purpose**: Convert text documents into numerical feature vectors
- **Configuration**: 15,000 features, 1-3 gram analysis, sublinear term frequency scaling
- **Function**: Identifies important terms and their relevance across the document corpus

#### 2. Word2Vec Embeddings
- **Purpose**: Capture semantic relationships between medical terms
- **Configuration**: 200-dimensional vectors, skip-gram model, 10-word context window
- **Function**: Enables understanding of medical terminology relationships and synonyms

#### 3. Truncated SVD (Singular Value Decomposition)
- **Purpose**: Dimensionality reduction and noise reduction
- **Configuration**: 300 components with standardization
- **Function**: Reduces computational complexity while preserving important information

#### 4. K-means Clustering
- **Purpose**: Group similar medical documents for improved search relevance
- **Configuration**: 50 clusters using reduced feature space
- **Function**: Organizes documents by medical similarity for targeted search

#### 5. Cosine Similarity Matching
- **Purpose**: Measure document similarity for query matching
- **Function**: Ranks search results by relevance to user queries

#### 6. Medical Knowledge Base
- **Purpose**: Structured extraction of medical patterns
- **Categories**: Diagnoses, medications, procedures, symptoms
- **Function**: Provides categorized medical information for enhanced responses

## Technical Implementation

### Core Components

1. **MedicalChatbotTrainer** (`train_model.py`)
   - Handles data loading and preprocessing
   - Trains all machine learning models
   - Saves trained models for persistent use
   - Creates medical knowledge base

2. **MedicalChatbot** (`medical_chatbot.py`)
   - Loads pre-trained models
   - Processes user queries
   - Generates intelligent responses
   - Combines multiple search techniques

3. **Web Interface** (`web_app.py`)
   - Streamlit-based user interface
   - Interactive search functionality
   - Model statistics and visualization
   - Professional design with responsive layout

### Search Capabilities

- **Medical Term Search**: Query medical conditions, treatments, and procedures
- **Patient-Specific Search**: Look up records by patient ID
- **Semantic Search**: Understanding of medical terminology relationships
- **Multi-method Ranking**: Combines results from different ML techniques
- **Confidence Scoring**: Provides relevance scores for search results

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- MIMIC-IV discharge.csv dataset
- Minimum 8GB RAM recommended
- 2GB free disk space for models

### Installation Steps

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the dataset**:
   - Place your `discharge.csv` file at `C:/Users/imran/Downloads/discharge.csv`
   - Or modify the path in `train_model.py` line 21

4. **Train the models** (one-time setup):
   ```bash
   python train_model.py
   ```
   This process may take 30-60 minutes depending on dataset size and hardware.

5. **Launch the web application**:
   ```bash
   streamlit run web_app.py
   ```

## Usage

### Web Interface

1. **Access the application** at `http://localhost:8501`
2. **Navigate between pages**:
   - Chat Interface: Main search functionality
   - Model Statistics: Performance metrics and visualizations
   - About: System information and documentation

### Search Examples

- **Medical Conditions**: "diabetes treatment", "heart failure symptoms"
- **Procedures**: "cardiac catheterization", "mechanical ventilation"
- **Medications**: "insulin therapy", "antibiotic treatment"
- **Patient Records**: Enter numeric patient ID (e.g., "12345")

### Response Features

- **Confidence Scores**: Percentage relevance for each result
- **Multiple Results**: Top 3 most relevant matches
- **Medical Knowledge**: Related medical information from knowledge base
- **Search History**: Previous queries and responses

## Model Performance

### Training Metrics

- **Dataset Size**: 300,000+ discharge notes
- **TF-IDF Features**: 15,000 optimized features
- **Word2Vec Vocabulary**: 20,000+ medical terms
- **Clustering**: 50 document clusters
- **Processing Time**: Sub-second query response

### Accuracy Features

- **Multi-technique Validation**: Cross-validation using multiple ML approaches
- **Medical Term Recognition**: Specialized medical vocabulary processing
- **Context Awareness**: Understanding of medical terminology relationships
- **Relevance Ranking**: Weighted scoring system for result prioritization

## Confidence Scoring System

### Mathematical Foundation

The system uses **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorization combined with **cosine similarity** to calculate confidence scores:

```
Confidence Score = cosine_similarity(query_vector, document_vector) × 100%
```

### Classification Thresholds

**60-100% (HIGH Confidence)**
- Strong semantic overlap between search terms and medical content
- Multiple exact or closely related medical terms found
- High term frequency in the specific document
- Low inverse document frequency (rare/specific terms)
- Examples: "pneumothorax chest tube", "myocardial infarction treatment"

**30-59% (MEDIUM Confidence)**
- Moderate term overlap with medical record
- Some relevant medical terminology present
- Partial matches with search concepts
- General medical relevance confirmed
- Examples: "breathing problems", "heart condition management"

**0-29% (LOW Confidence)**
- Minimal direct term matches
- Broad medical context relevance
- Common medical terms with high frequency across corpus
- General healthcare content without specific matches
- Examples: "patient care", "medical treatment"

### Key Scoring Factors

1. **Term Rarity (IDF Component)**
   - Rare medical terms receive exponentially higher weights
   - Specialized terminology (e.g., "pneumothorax") vs common terms (e.g., "patient")
   - Formula: `log(total_documents / documents_containing_term)`

2. **Term Frequency (TF Component)**
   - How often search terms appear in the specific medical record
   - Normalized by document length to prevent bias
   - Multiple mentions increase relevance score

3. **Query Coverage**
   - Percentage of search terms found in the document
   - Partial matches receive proportional scoring
   - Multi-word queries require comprehensive term matching

4. **Medical Specificity**
   - Medical terminology receives priority over general language
   - Clinical abbreviations and Latin terms weighted higher
   - Procedure codes and drug names given maximum specificity

5. **Document Normalization**
   - Longer documents normalized to prevent length bias
   - Cosine similarity inherently accounts for document magnitude
   - Ensures fair comparison across varying record lengths

### Optimization for Medical Searches

**High Confidence Strategies:**
- Use precise medical terminology: "myocardial infarction" vs "heart attack"
- Include specific procedures: "cardiac catheterization with stent placement"
- Combine condition + treatment: "diabetes mellitus insulin therapy"
- Use clinical abbreviations: "COPD exacerbation treatment"

**Understanding Low Scores:**
- Very specific searches may yield lower scores if terminology doesn't match exactly
- Common medical terms compete with high-frequency corpus words
- General searches return broader, less specific matches

### Technical Implementation

The confidence calculation process:

1. **Query Preprocessing**: Tokenization, stemming, medical term normalization
2. **Vectorization**: Convert query and documents to TF-IDF feature vectors
3. **Similarity Calculation**: Compute cosine similarity between normalized vectors
4. **Score Conversion**: Scale similarity (0-1) to percentage (0-100%)
5. **Ranking**: Sort results by confidence score in descending order

This methodology ensures that users receive quantitatively accurate confidence assessments based on mathematical similarity rather than subjective interpretation.

## System Requirements

### Minimum Requirements

- **CPU**: Dual-core processor
- **RAM**: 8GB
- **Storage**: 2GB free space
- **Python**: 3.8+

### Recommended Requirements

- **CPU**: Quad-core processor or better
- **RAM**: 16GB or more
- **Storage**: 5GB free space (for large datasets)
- **Python**: 3.9+

## File Structure

```
medical-chatbot/
├── train_model.py          # Model training script
├── medical_chatbot.py      # Core chatbot class
├── web_app.py             # Streamlit web interface
├── requirements.txt       # Python dependencies
├── README.md             # Documentation
├── trained_models/       # Saved ML models (created after training)
└── .streamlit/           # Streamlit configuration
```

## Deployment

### Local Deployment

The system is designed for local deployment and can be run on any machine with Python support.

### Cloud Deployment

For cloud deployment, ensure:
- Sufficient memory allocation (minimum 2GB)
- Persistent storage for model files
- Proper security configuration for medical data

## Troubleshooting

### Common Issues

1. **Model Loading Errors**: Ensure `train_model.py` has been run successfully
2. **Memory Issues**: Increase system RAM or reduce dataset size
3. **NLTK Data Errors**: The system automatically downloads required NLTK data
4. **Path Issues**: Verify the discharge.csv file path in `train_model.py`

### Performance Optimization

- **Model Caching**: Pre-trained models are cached for faster loading
- **Efficient Search**: Multiple optimization techniques reduce query time
- **Memory Management**: Optimized data structures minimize memory usage

## Technical Support

For technical issues or questions about implementation, refer to the code documentation and comments within each module. The system is designed to be self-contained and includes comprehensive error handling and user feedback.

## Data Privacy

This system is designed to work with de-identified medical data. Ensure compliance with relevant healthcare data regulations (HIPAA, GDPR) when deploying in production environments. 