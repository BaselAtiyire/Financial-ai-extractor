import os
import time
import requests
import json
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# PDF reading
from pypdf import PdfReader

# Pydantic validation
from pydantic import BaseModel, ValidationError
from typing import List, Optional

# -----------------------
# Config (local + Railway)
# -----------------------
load_dotenv()

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/extract")
API_KEY = os.getenv("API_KEY", "")

# --- History (last 5 extractions) ---
if "history" not in st.session_state:
    st.session_state.history = []

if "last_latency" not in st.session_state:
    st.session_state.last_latency = None

# -----------------------
# Pydantic schemas
# -----------------------
class MetricRow(BaseModel):
    measure: str
    estimated: Optional[str] = None
    actual: Optional[str] = None

class ExtractedResult(BaseModel):
    rows: List[MetricRow]

# -----------------------
# Helpers
# -----------------------
def read_pdf_to_text(file) -> str:
    reader = PdfReader(file)
    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts).strip()

def split_into_batches(text: str) -> List[str]:
    chunks = [t.strip() for t in text.split("\n\n") if t.strip()]
    return chunks if chunks else [text.strip()]

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Financial Data Extractor", layout="centered")

st.title("📊 Financial Data Extractor")
st.caption("Upload TXT/PDF or paste financial text. Extract structured metrics via FastAPI.")

with st.sidebar:
    st.markdown("### 🔌 API Config")
    st.write("API_URL:", API_URL)
    st.write("API_KEY loaded?", bool(API_KEY))

uploaded = st.file_uploader("Upload a file (TXT or PDF)", type=["txt", "pdf"])

default_text = """Earnings per share: $1.64, adjusted, versus $1.60 estimated
Revenue: $94.93 billion vs. $94.58 billion estimated
"""

if "input_text" not in st.session_state:
    st.session_state.input_text = default_text

if uploaded is not None:
    try:
        if uploaded.name.lower().endswith(".txt"):
            st.session_state.input_text = uploaded.read().decode("utf-8", errors="ignore")
        elif uploaded.name.lower().endswith(".pdf"):
            st.session_state.input_text = read_pdf_to_text(uploaded)
    except Exception as e:
        st.error("Failed to read file")
        st.code(str(e))

text = st.text_area("Enter financial text:", value=st.session_state.input_text, height=220)
st.session_state.input_text = text

colA, colB = st.columns([1, 1])
with colA:
    extract_clicked = st.button("Extract", type="primary")
with colB:
    if st.button("Load sample"):
        st.session_state.input_text = default_text
        st.rerun()

# -----------------------
# Extract
# -----------------------
if extract_clicked:
    if not text.strip():
        st.warning("Paste some text first.")
    else:
        with st.spinner("Calling API + extracting..."):
            try:
                batches = split_into_batches(text)
                all_rows = []
                total_start = time.perf_counter()

                headers = {"X-API-Key": API_KEY} if API_KEY else {}

                for i, batch in enumerate(batches, start=1):
                    start = time.perf_counter()
                    resp = requests.post(API_URL, json={"text": batch}, headers=headers, timeout=60)
                    elapsed = time.perf_counter() - start

                    if resp.status_code != 200:
                        st.error(f"API Error {resp.status_code} on batch {i}")
                        st.code(resp.text)
                        st.stop()

                    validated = ExtractedResult(**resp.json())
                    all_rows.extend([r.model_dump() for r in validated.rows])
                    st.caption(f"Batch {i}/{len(batches)} done in {elapsed:.2f}s")

                total_elapsed = time.perf_counter() - total_start
                st.session_state.last_latency = total_elapsed

                df = pd.DataFrame(all_rows)
                if not df.empty:
                    df = df.rename(columns={"measure": "Measure", "estimated": "Estimated", "actual": "Actual"})
                    df = df[["Measure", "Estimated", "Actual"]]

                st.success(f"Done in {total_elapsed:.2f}s")
                st.dataframe(df, use_container_width=True, hide_index=True)

                if not df.empty:
                    st.download_button("Download CSV", df.to_csv(index=False), "financial_extract.csv", "text/csv")

                st.download_button("Download JSON", json.dumps({"rows": all_rows}, indent=2), "financial_extract.json", "application/json")

                preview = text.replace("\n", " ")[:120]
                st.session_state.history.insert(0, {"input_preview": preview, "result": {"rows": all_rows}})
                st.session_state.history = st.session_state.history[:5]

            except requests.exceptions.ConnectionError:
                st.error("Could not connect to API. Is FastAPI running?")
            except ValidationError as ve:
                st.error("Schema validation failed.")
                st.code(str(ve))
            except Exception as e:
                st.error("Extraction failed.")
                st.code(str(e))

# -----------------------
# History
# -----------------------
st.subheader("🕘 Recent Extractions")
if not st.session_state.history:
    st.info("No history yet.")
else:
    for i, item in enumerate(st.session_state.history, start=1):
        with st.expander(f"Run #{i}: {item['input_preview']}"):
            st.json(item["result"])

if st.session_state.last_latency is not None:
    st.caption(f"⏱️ Last runtime: {st.session_state.last_latency:.2f}s")
