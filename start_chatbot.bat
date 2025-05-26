@echo off
echo 🏥 Starting MIMIC-ICU Medical Chatbot...
echo.
echo The web interface will open at: http://localhost:8501
echo Press Ctrl+C to stop the server
echo.
streamlit run web_chatbot.py --server.port 8501 --server.address localhost
pause 