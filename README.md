# MIMIC-ICU Discharge Notes Chatbot

A Python-based chatbot for searching and analyzing MIMIC-ICU discharge notes. This project provides two interfaces:
1. **Web interface** using Streamlit with advanced semantic search
2. **Command-line interface** for simple text-based interactions

## Features

- 🔍 **Multiple Search Methods**: Semantic search, keyword search, and patient-specific search
- 📊 **Dataset Analytics**: Comprehensive statistics about the discharge notes
- 🎯 **Smart Filtering**: Search by patient ID, admission ID, or date ranges
- 💻 **Dual Interface**: Web UI (Streamlit) and CLI versions
- 🧠 **AI-Powered**: Uses sentence transformers for semantic understanding
- ⚡ **Fast Search**: TF-IDF and FAISS indexing for quick results

## Dataset Information

This chatbot works with the MIMIC-ICU discharge notes dataset containing:
- **331,793** discharge notes
- **Patient information** with unique IDs
- **Hospital admission details**
- **Full medical text** from discharge summaries
- **Timestamps** for chronological analysis

## Installation

### Prerequisites
- Python 3.8 or higher
- At least 4GB RAM (recommended 8GB for full dataset)
- MIMIC-ICU discharge.csv file

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Verify Data Path
Make sure your `discharge.csv` file is located at:
```
C:\Users\imran\Downloads\discharge.csv
```

Or update the file path in the code if your file is located elsewhere.

## Usage

### Option 1: Web Interface (Recommended)

Launch the Streamlit web application:

```bash
streamlit run chatbot.py
```

This will open a web browser with an interactive interface where you can:
- Load and analyze the discharge data
- Perform semantic searches using AI
- Browse patient records
- View comprehensive statistics

### Option 2: Command Line Interface

For a simpler, terminal-based experience:

```bash
python simple_chatbot.py
```

This provides a menu-driven interface with the following options:
1. Search by text query
2. Search by keywords
3. Search by patient ID
4. Show dataset statistics
5. Exit

## Search Examples

### Semantic Search Queries
```
"diabetes complications and treatment"
"heart failure medication management"
"post-surgical recovery instructions"
"patient discharge planning"
"antibiotic allergies and reactions"
```

### Keyword Search
```
Keywords: diabetes, insulin, glucose
Keywords: surgery, procedure, operation
Keywords: medication, dosage, prescription
```

### Patient Search
```
Patient ID: 10000032
Patient ID: 10001217
```

## File Structure

```
chatbot/
├── chatbot.py              # Main Streamlit web application
├── simple_chatbot.py       # Command-line interface
├── examine_data.py         # Data exploration script
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Key Features Explained

### 1. Semantic Search
Uses advanced AI models (Sentence Transformers) to understand the meaning behind your queries, not just keyword matching.

### 2. TF-IDF Search
Traditional text search using Term Frequency-Inverse Document Frequency for fast keyword-based results.

### 3. Patient Timeline
View all discharge notes for a specific patient chronologically to understand their medical journey.

### 4. Smart Sampling
For performance, the system can work with a subset of the data while maintaining search accuracy.

## Performance Considerations

- **Full Dataset**: ~331K notes (requires more memory and processing time)
- **Sample Mode**: 10K-20K notes (faster, good for testing and development)
- **Search Index**: Built once, then cached for fast subsequent searches

## Troubleshooting

### Common Issues

1. **Memory Error**: Reduce the sample size in the code or ensure you have enough RAM
2. **File Path Error**: Update the CSV path in the code to match your file location
3. **Slow Performance**: Use the sample mode for faster results during development

### Dependencies Issues
If you encounter package installation issues:

```bash
pip install --upgrade pip
pip install pandas numpy scikit-learn nltk streamlit sentence-transformers faiss-cpu
```

## Advanced Usage

### Customizing Search Parameters

In `chatbot.py` or `simple_chatbot.py`, you can modify:
- `sample_size`: Number of notes to index (default: 10K-20K)
- `max_features`: TF-IDF vocabulary size
- `top_k`: Number of search results returned

### Adding New Search Methods

The codebase is modular and allows easy addition of new search algorithms or filters.

## Data Privacy and Ethics

⚠️ **Important**: MIMIC-ICU data contains sensitive medical information. Please ensure:
- You have proper authorization to access this data
- Follow all applicable privacy regulations (HIPAA, etc.)
- Do not share or distribute the data
- Use only for research and educational purposes

## Example Output

```
🔍 Found 5 relevant results:

📄 Rank 1 | Score: 0.756
   Patient ID: 10001884 | Note ID: 10001884-DS-15
   Admission: 29827561 | Date: 2180-04-23 00:00:00
   Content Preview:
   Discharge Summary for patient with diabetes mellitus type 2
   Patient was admitted for hyperglycemic crisis...
```

## Contributing

This is a research/educational tool. Feel free to:
- Add new search algorithms
- Improve the user interface
- Optimize performance
- Add new analysis features

## License

This code is provided for educational and research purposes. Please ensure you have proper authorization for the MIMIC-ICU dataset before use.

## Support

For issues with:
- **Code**: Check the troubleshooting section or review the error messages
- **MIMIC-ICU Data**: Refer to the official MIMIC documentation
- **Dependencies**: Ensure all packages are correctly installed

---

*Built with ❤️ for medical research and education* 