# 🏥 MIMIC-ICU Web Chatbot - Quick Start

## ✅ Your chatbot is READY TO RUN!

### 🚀 How to Start (Choose one method):

#### Method 1: Double-click the batch file
```
start_chatbot.bat
```

#### Method 2: Command line
```bash
streamlit run web_chatbot.py
```

#### Method 3: Python launcher
```bash
python run_chatbot.py
```

### 🌐 Access the Web Interface

Once started, open your browser and go to:
**http://localhost:8501**

The web interface will automatically open in your default browser.

### 🎯 What You Can Do

#### **Text Search**
- Search for medical conditions: "diabetes", "heart failure"
- Find procedures: "surgery", "catheter", "intubation"
- Look for medications: "insulin", "antibiotics", "morphine"

#### **Patient Lookup**
- Enter any patient ID (e.g., 10000032, 10001217)
- View all discharge notes for that patient
- See chronological medical history

### 📊 Features Available

✅ **Fast Search** - TF-IDF powered text search  
✅ **10,000 Notes** - Sample of discharge records for quick performance  
✅ **Real-time Results** - Instant search with relevance scoring  
✅ **Patient Timeline** - Complete medical history per patient  
✅ **Clean Interface** - Modern, medical-focused design  
✅ **Mobile Friendly** - Works on phones and tablets  

### 🔧 Technical Details

- **Data Source**: C:/Users/imran/Downloads/discharge.csv
- **Sample Size**: 10,000 notes (for fast performance)
- **Search Method**: TF-IDF vectorization with cosine similarity
- **Web Framework**: Streamlit
- **Port**: 8501 (configurable)

### 🆘 Troubleshooting

**If the chatbot won't start:**
1. Make sure you're in the correct directory: `C:\Users\imran\Downloads\chatbot`
2. Check if packages are installed: `pip install streamlit pandas numpy scikit-learn`
3. Verify data file exists: `C:\Users\imran\Downloads\discharge.csv`

**If search returns no results:**
- Try simpler terms like "diabetes" or "surgery"
- Use medical terminology
- Check spelling

**If the web page won't load:**
- Wait 30 seconds for the server to start
- Try refreshing the browser
- Check the terminal for error messages

### 🎉 Ready to Explore!

Your MIMIC-ICU medical chatbot is now running and ready to help you explore discharge notes and patient data. The web interface provides an intuitive way to search through medical records and gain insights from the data.

**Happy searching! 🔍** 