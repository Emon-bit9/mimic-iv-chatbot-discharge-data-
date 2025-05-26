"""
Quick test of the basic chatbot functionality
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

def test_basic_search():
    print("🏥 MIMIC-ICU Chatbot Quick Test")
    print("=" * 40)
    
    # Load a small sample of data
    csv_path = "C:/Users/imran/Downloads/discharge.csv"
    print("📁 Loading sample data...")
    
    try:
        # Read only first 1000 rows for quick testing
        df = pd.read_csv(csv_path, nrows=1000)
        print(f"✅ Loaded {len(df)} sample records")
        
        # Clean text data
        print("🧹 Cleaning text data...")
        df['cleaned_text'] = df['text'].apply(lambda x: re.sub(r'[^\w\s]', ' ', str(x)) if pd.notna(x) else "")
        
        # Build TF-IDF index
        print("🔧 Building search index...")
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.8
        )
        
        tfidf_matrix = vectorizer.fit_transform(df['cleaned_text'])
        print("✅ Search index ready")
        
        # Test search
        test_queries = [
            "diabetes",
            "heart failure",
            "medication",
            "discharge planning",
            "surgery"
        ]
        
        print("\n🔍 Testing search functionality:")
        print("-" * 40)
        
        for query in test_queries:
            # Transform query
            query_tfidf = vectorizer.transform([query])
            
            # Calculate similarity
            similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
            
            # Get top result
            top_idx = similarities.argmax()
            top_score = similarities[top_idx]
            
            if top_score > 0:
                result = df.iloc[top_idx]
                print(f"\n📄 Query: '{query}'")
                print(f"   Score: {top_score:.3f}")
                print(f"   Patient: {result['subject_id']}")
                print(f"   Note ID: {result['note_id']}")
                print(f"   Preview: {result['text'][:100]}...")
            else:
                print(f"\n❌ No results for: '{query}'")
        
        # Show statistics
        print(f"\n📊 Sample Statistics:")
        print(f"   Total notes: {len(df)}")
        print(f"   Unique patients: {df['subject_id'].nunique()}")
        print(f"   Date range: {df['charttime'].min()} to {df['charttime'].max()}")
        
        print("\n✅ Basic functionality test PASSED!")
        print("\n🚀 Ready to use the full chatbot:")
        print("   python simple_chatbot.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_basic_search() 