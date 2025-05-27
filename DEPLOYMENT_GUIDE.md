# 🚀 Medical Chatbot Deployment Guide

## Overview

This guide provides multiple deployment options for your Medical Information Retrieval System, from simple cloud deployments to enterprise-grade solutions.

## ⚠️ Important Considerations

### Data Privacy & Compliance
- **MIMIC-IV Data**: Contains de-identified medical records - ensure compliance with local regulations
- **HIPAA Compliance**: Required for US healthcare applications
- **GDPR Compliance**: Required for EU deployments
- **Access Control**: Implement proper authentication for sensitive medical data

### Model Size Limitations
- **Trained Models**: ~270MB total size
- **Memory Requirements**: 2-4GB RAM minimum
- **Storage**: 2GB+ free space required

---

## 🌟 Recommended Deployment Options

### 1. Streamlit Cloud (Easiest - Demo Only)

**Best for**: Public demos, portfolios, proof-of-concept

**Limitations**: 
- Cannot upload large model files (>100MB limit)
- Limited to demo interface without actual search functionality
- 1GB memory limit

**Steps**:
1. **Create GitHub Repository**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/yourusername/medical-chatbot.git
   git push -u origin main
   ```

2. **Deploy to Streamlit Cloud**:
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select your repository
   - Set main file as `app.py`
   - Deploy

3. **Configuration**:
   - Uses `requirements_deploy.txt`
   - Automatically shows demo interface
   - No model training required

**Result**: Public demo at `https://yourusername-medical-chatbot-app-xyz123.streamlit.app`

---

### 2. Heroku (Recommended for Full Functionality)

**Best for**: Full-featured deployment with actual search capabilities

**Requirements**: 
- Heroku account (free tier available)
- Git repository
- Pre-trained models

**Steps**:

1. **Install Heroku CLI**:
   ```bash
   # Windows
   # Download from https://devcenter.heroku.com/articles/heroku-cli
   
   # Verify installation
   heroku --version
   ```

2. **Create Heroku Configuration Files**:

   **Procfile**:
   ```
   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```

   **runtime.txt**:
   ```
   python-3.11.6
   ```

   **heroku.yml** (for container deployment):
   ```yaml
   build:
     docker:
       web: Dockerfile
   ```

3. **Create Dockerfile** (for larger models):
   ```dockerfile
   FROM python:3.11-slim
   
   WORKDIR /app
   
   COPY requirements_deploy.txt .
   RUN pip install -r requirements_deploy.txt
   
   COPY . .
   
   EXPOSE 8501
   
   CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```

4. **Deploy**:
   ```bash
   heroku login
   heroku create your-medical-chatbot
   
   # For regular deployment
   git push heroku main
   
   # For container deployment (larger models)
   heroku stack:set container
   git push heroku main
   ```

**Cost**: Free tier (512MB RAM) or Hobby ($7/month, 1GB RAM)

---

### 3. Railway (Modern Alternative)

**Best for**: Easy deployment with better resource limits

**Steps**:
1. Visit [railway.app](https://railway.app)
2. Connect GitHub repository
3. Deploy automatically
4. Configure environment variables if needed

**Advantages**:
- 1GB RAM on free tier
- Automatic deployments
- Better performance than Heroku free tier

---

### 4. Google Cloud Platform (Enterprise)

**Best for**: Production deployments, enterprise use

**Steps**:

1. **Create Cloud Run Service**:
   ```bash
   gcloud run deploy medical-chatbot \
     --source . \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated
   ```

2. **Configure Resources**:
   ```yaml
   # cloudbuild.yaml
   steps:
   - name: 'gcr.io/cloud-builders/docker'
     args: ['build', '-t', 'gcr.io/$PROJECT_ID/medical-chatbot', '.']
   - name: 'gcr.io/cloud-builders/docker'
     args: ['push', 'gcr.io/$PROJECT_ID/medical-chatbot']
   - name: 'gcr.io/cloud-builders/gcloud'
     args: ['run', 'deploy', 'medical-chatbot', '--image', 'gcr.io/$PROJECT_ID/medical-chatbot', '--region', 'us-central1', '--platform', 'managed', '--allow-unauthenticated']
   ```

**Cost**: Pay-per-use, typically $10-50/month for moderate usage

---

### 5. AWS (Scalable Solution)

**Best for**: High-traffic applications, enterprise deployments

**Options**:
- **AWS App Runner**: Easiest container deployment
- **ECS Fargate**: Container orchestration
- **EC2**: Full control virtual machines

**Example App Runner Deployment**:
```yaml
# apprunner.yaml
version: 1.0
runtime: python3
build:
  commands:
    build:
      - pip install -r requirements_deploy.txt
run:
  runtime-version: 3.11.6
  command: streamlit run app.py --server.port=8080 --server.address=0.0.0.0
  network:
    port: 8080
    env: PORT
```

---

## 🔧 Pre-Deployment Checklist

### 1. Model Preparation
```bash
# Ensure models are trained
python simple_train.py

# Verify model files exist
ls -la trained_models/
```

### 2. Dependencies
```bash
# Test requirements
pip install -r requirements_deploy.txt

# Test application locally
streamlit run app.py
```

### 3. Environment Variables
Create `.env` file for sensitive configurations:
```env
DATA_PATH=/path/to/discharge.csv
MODEL_PATH=./trained_models
DEBUG=False
```

### 4. Security Configuration
```python
# Add to app.py for production
if not st.session_state.get('authenticated', False):
    password = st.text_input("Enter access password:", type="password")
    if password == os.getenv('APP_PASSWORD'):
        st.session_state.authenticated = True
        st.rerun()
    else:
        st.stop()
```

---

## 📊 Deployment Comparison

| Platform | Cost | RAM | Storage | Model Support | Difficulty |
|----------|------|-----|---------|---------------|------------|
| Streamlit Cloud | Free | 1GB | 1GB | Demo Only | ⭐ |
| Railway | Free/Paid | 1GB/8GB | 1GB/100GB | Full | ⭐⭐ |
| Heroku | $7/month | 1GB | 1GB | Full | ⭐⭐ |
| Google Cloud | Pay-per-use | 2GB+ | Unlimited | Full | ⭐⭐⭐ |
| AWS | Pay-per-use | 2GB+ | Unlimited | Full | ⭐⭐⭐⭐ |

---

## 🚀 Quick Start Deployment

### Option 1: Demo Deployment (5 minutes)
```bash
# 1. Push to GitHub
git add .
git commit -m "Deploy medical chatbot"
git push origin main

# 2. Deploy to Streamlit Cloud
# Visit share.streamlit.io and connect repository
```

### Option 2: Full Deployment (15 minutes)
```bash
# 1. Install Heroku CLI
# 2. Login and create app
heroku login
heroku create your-medical-chatbot

# 3. Configure for larger models
echo "web: streamlit run app.py --server.port=\$PORT --server.address=0.0.0.0" > Procfile
echo "python-3.11.6" > runtime.txt

# 4. Deploy
git add .
git commit -m "Heroku deployment"
git push heroku main
```

---

## 🔒 Security Best Practices

### 1. Authentication
```python
# Simple password protection
def check_password():
    def password_entered():
        if st.session_state["password"] == "your_secure_password":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("Password incorrect")
        return False
    else:
        return True

if check_password():
    # Your app content here
    pass
```

### 2. Rate Limiting
```python
import time
from collections import defaultdict

# Simple rate limiting
if 'last_request' not in st.session_state:
    st.session_state.last_request = defaultdict(float)

def rate_limit(user_id, max_requests_per_minute=10):
    current_time = time.time()
    if current_time - st.session_state.last_request[user_id] < 60/max_requests_per_minute:
        st.error("Too many requests. Please wait.")
        return False
    st.session_state.last_request[user_id] = current_time
    return True
```

### 3. Data Privacy
```python
# Anonymize sensitive data
def anonymize_results(results):
    for result in results:
        # Remove or hash sensitive identifiers
        result['subject_id'] = f"ANON_{hash(result['subject_id']) % 10000}"
    return results
```

---

## 📈 Monitoring & Analytics

### 1. Usage Tracking
```python
import streamlit.analytics as analytics

# Track user interactions
analytics.track_event("search_performed", {
    "query": query,
    "results_count": len(results),
    "timestamp": datetime.now()
})
```

### 2. Performance Monitoring
```python
import time

def monitor_performance(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        st.sidebar.metric("Response Time", f"{end_time - start_time:.2f}s")
        return result
    return wrapper
```

---

## 🆘 Troubleshooting

### Common Issues

1. **Memory Errors**:
   - Reduce model size or use model compression
   - Implement lazy loading
   - Use cloud storage for models

2. **Slow Performance**:
   - Implement caching with `@st.cache_data`
   - Use CDN for static assets
   - Optimize TF-IDF parameters

3. **Deployment Failures**:
   - Check requirements.txt compatibility
   - Verify Python version
   - Review deployment logs

### Debug Mode
```python
# Add debug information
if os.getenv('DEBUG', 'False').lower() == 'true':
    st.sidebar.write("Debug Info:")
    st.sidebar.json({
        "memory_usage": psutil.virtual_memory().percent,
        "model_loaded": chatbot is not None,
        "session_state": dict(st.session_state)
    })
```

---

## 📞 Support & Resources

- **Documentation**: Complete technical documentation in README.md
- **Issues**: Report bugs and feature requests on GitHub
- **Community**: Join discussions about medical NLP and deployment
- **Updates**: Follow repository for latest improvements

---

## ⚖️ Legal Disclaimer

This system is for educational and research purposes only. Not intended for medical diagnosis or treatment decisions. Always consult qualified healthcare professionals for medical advice. Ensure compliance with local healthcare data regulations before deployment. 