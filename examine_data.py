import pandas as pd
import os

# Load the CSV file
csv_path = "C:/Users/imran/Downloads/discharge.csv"

try:
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    print("=== DISCHARGE CSV DATA ANALYSIS ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")
    print("\nColumn names:")
    for i, col in enumerate(df.columns):
        print(f"{i+1}. {col}")
    
    print("\nFirst 3 rows:")
    print(df.head(3))
    
    print("\nData types:")
    print(df.dtypes)
    
    print("\nNull values:")
    print(df.isnull().sum())
    
    # Sample some text data if available
    text_columns = df.select_dtypes(include=['object']).columns
    print(f"\nText columns: {list(text_columns)}")
    
    if len(text_columns) > 0:
        print("\nSample text data:")
        for col in text_columns[:3]:  # Show first 3 text columns
            print(f"\n{col}:")
            sample_text = df[col].dropna().iloc[0] if not df[col].dropna().empty else "No data"
            print(f"Sample: {str(sample_text)[:200]}...")
            
except Exception as e:
    print(f"Error reading CSV: {e}") 