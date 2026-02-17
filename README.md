# Financial AI Extractor (FastAPI + Streamlit + Groq LLM)

An AI-powered web app that extracts structured financial metrics (Actual vs Estimated) from unstructured earnings text (and text-based PDFs).  
It provides:
- A **FastAPI backend** with **OpenAPI/Swagger docs** and **API-key auth**
- A **Streamlit frontend** for paste/upload, table view, and **CSV/JSON downloads**
- **Schema validation** using **Pydantic** to keep outputs consistent

---

## Architecture

Streamlit UI (port 8501)
→ calls FastAPI backend (port 8000)
→ backend calls Groq LLM
→ returns validated JSON
→ UI renders a table + allows downloads

---

## Features

- Text input + (optional) PDF upload (text-based PDFs)
- Batch processing (multiple paragraphs separated by blank lines)
- JSON output with schema validation (Pydantic)
- Table view (Measure / Estimated / Actual)
- Download results as CSV + JSON
- Run history (last 5 extractions)
- FastAPI docs at `/docs` with **Authorize** button (X-API-Key)
- Docker-ready setup

---

## Project Structure
ai-projects/
├─ app.py # Streamlit UI
├─ api.py # FastAPI backend
├─ requirements.txt # Python deps
├─ .env # local secrets 
├─ .env.docker # docker env 
├─ Dockerfile # docker build 
└─ start.sh # docker startup 


---

## Requirements

- Python 3.12 recommended (works on newer versions, but you may see warnings)
- A Groq API key

---

## Setup (Local)

### 1) Create `.env` in the project folder

Create a file named `.env` inside your project directory and add:

```env
GROQ_API_KEY=your_groq_key_here
API_KEY=supersecret123

Install dependencies
pip install -r requirements.txt

Run FastAPI backend
cd C:\Users\basil\OneDrive\Desktop\ai-projects
python -m uvicorn api:app --reload

Open Swagger docs:
http://127.0.0.1:8000/docs

Run Streamlit frontend (new terminal)
cd C:\Users\basil\OneDrive\Desktop\ai-projects
streamlit run app.py

Open the app:
http://localhost:8501/

Using Swagger (Authorize)
Open http://127.0.0.1:8000/docs

Click Authorize

Enter your API key (value of API_KEY)

Run POST /extract

Example request
{
  "text": "Earnings per share: $1.64, adjusted, versus $1.60 estimated\nRevenue: $94.93 billion vs. $94.58 billion estimated"
}
Example Output
{
  "rows": [
    {"measure": "EPS", "estimated": "$1.60", "actual": "$1.64"},
    {"measure": "Revenue", "estimated": "$94.58 billion", "actual": "$94.93 billion"}
  ]
}

Notes / Limitations

Scanned PDFs (image-only) may not extract text without OCR.

This project focuses on structured extraction + validation rather than perfect financial normalization.

Roadmap (Optional)

OCR support for scanned PDFs

Evaluation set + accuracy metrics

Multi-document schemas (invoice, bank statements, 10-K sections)

Hosted deployment (Railway/Render)

Author
Basel Atiyire

---







