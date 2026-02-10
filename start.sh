#!/usr/bin/env bash
set -e

# Start FastAPI in background
uvicorn api:app --host 0.0.0.0 --port 8000 &

# Start Streamlit in foreground
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
