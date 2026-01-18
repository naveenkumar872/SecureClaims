# 🛡️ SecureClaims

**AI-powered insurance fraud detection system** that uses machine learning to analyze car insurance claims and predict fraud probability. It combines a trained Random Forest model with a RAG-based policy assistant (using Qdrant + Groq) to provide fraud scores and answer policy-related questions through a simple web interface.

## 🚀 Features

- **Fraud Detection** - ML model analyzes claim details to predict fraud probability
- **Policy Q&A** - RAG-powered assistant answers questions about insurance policies
- **Web Interface** - Clean UI for submitting claims and viewing results

## 🛠️ Tech Stack

- **Backend**: FastAPI
- **ML**: Scikit-learn, Joblib
- **Vector DB**: Qdrant
- **Embeddings**: Sentence Transformers (BGE)
- **LLM**: Groq API
- **Frontend**: HTML/CSS

## 📁 Project Structure

```
SecureClaims/
├── api.py                 # FastAPI backend
├── frontend/              # Web interface
├── models/                # Trained ML model
├── data/                  # Dataset & policy documents
└── scripts/               # Training & data processing scripts
```

## ⚙️ Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set environment variables in `.env`:
   ```
   QUADRANT_URL=your_qdrant_url
   QUADRANT_API_KEY=your_qdrant_api_key
   GROQ_API_KEY=your_groq_api_key
   ```

3. Run the server:
   ```bash
   uvicorn api:app --reload
   ```

4. Open `http://localhost:8000` in your browser
