import time
import os
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
# Config
# -----------------------
load_dotenv()

# ✅ Use Railway FastAPI in production (NOT localhost)
DEFAULT_API_BASE = "https://financial-ai-extractor-production.up.railway.app"

# You can override this from Streamlit Secrets or env vars if needed
API_BASE = os.getenv("API_BASE", DEFAULT_API_BASE).strip()
API_URL = f"{API_BASE.rstrip('/')}/extract"  # ensures no double slashes

# Optional API key (only if your FastAPI checks it)
API_KEY = os.getenv("API_KEY", "").strip()
HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}

# --- History (last 5 extractions) ---
if "history" not in st.session_state:
    st.session_state.history = []  # each item: {"input_preview": str, "result": dict}

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

def test_api_connectivity() -> tuple[bool, str]:
    """
    Tests if API is reachable by hitting /docs (FastAPI should serve it).
    Returns (ok, message).
    """
    try:
        r = requests.get(f"{API_BASE.rstrip('/')}/docs", timeout=15)
        if r.status_code == 200:
            return True, "Connected ✅ (/docs reachable)"
        return False, f"Reached API but got status {r.status_code} at /docs"
    except Exception as e:
        return False, f"Could not reach API: {e}"

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Financial Data Extractor", layout="centered")

st.markdown(
    """
    <style>
      .stApp { background: #0b0f17; color: #e6e9ef; }
      h1, h2, h3, p, label, div { color: #e6e9ef !important; }
      .block-container { padding-top: 2.25rem; max-width: 900px; }
      .stTextArea textarea {
        background: #141b2d !important;
        color: #e6e9ef !important;
        border: 1px solid #2a3552 !important;
        border-radius: 12px !important;
      }
      .stButton>button {
        background: #111827 !important;
        color: #e6e9ef !important;
        border: 1px solid #2a3552 !important;
        border-radius: 12px !important;
        padding: 0.6rem 1.1rem !important;
      }
      .stButton>button:hover { border-color: #3b82f6 !important; }
      div[data-testid="stDataFrame"] {
        background: #0f1626 !important;
        border: 1px solid #2a3552 !important;
        border-radius: 12px !important;
        padding: 0.25rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h1 style='font-weight:800;'>Financial Data Extractor</h1>", unsafe_allow_html=True)
st.caption("Upload TXT/PDF or paste text. Calls FastAPI on Railway and returns validated results.")

# -----------------------
# Sidebar debug
# -----------------------
st.sidebar.subheader("Debug")
st.sidebar.write("API_BASE:", API_BASE)
st.sidebar.write("API_URL:", API_URL)
st.sidebar.write("API_KEY loaded?", bool(API_KEY))
st.sidebar.write("Sending X-API-Key header?", bool(HEADERS))

ok, msg = test_api_connectivity()
if ok:
    st.sidebar.success(msg)
else:
    st.sidebar.error(msg)

st.sidebar.markdown("---")
st.sidebar.caption("Tip: If your API uses auth, be sure API_KEY is set in Streamlit Secrets or env vars.")

# -----------------------
# File Upload UI
# -----------------------
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
        st.error("Could not read uploaded file")
        st.code(str(e))

text = st.text_area("Enter financial text:", value=st.session_state.input_text, height=200)
st.session_state.input_text = text

colA, colB = st.columns([1, 1])
with colA:
    extract_clicked = st.button("Extract", type="primary")
with colB:
    if st.button("Load sample"):
        st.session_state.input_text = default_text
        st.rerun()

# -----------------------
# Extract Button
# -----------------------
if extract_clicked:
    if not text.strip():
        st.warning("Paste or upload text first.")
    else:
        with st.spinner("Calling API..."):
            try:
                batches = split_into_batches(text)
                all_rows = []
                total_start = time.perf_counter()

                for i, batch in enumerate(batches, start=1):
                    resp = requests.post(
                        API_URL,
                        json={"text": batch},
                        headers=HEADERS,
                        timeout=90
                    )

                    # Helpful debugging output
                    if resp.status_code != 200:
                        st.error(f"API Error {resp.status_code} on batch {i}")
                        st.code(resp.text[:2000])
                        st.stop()

                    raw_result = resp.json()

                    # Validate shape
                    try:
                        validated = ExtractedResult(**raw_result)
                        all_rows.extend([r.model_dump() for r in validated.rows])
                    except ValidationError as ve:
                        st.error("Response validation failed (API returned unexpected format)")
                        st.code(str(ve))
                        st.json(raw_result)
                        st.stop()

                total_elapsed = time.perf_counter() - total_start
                st.session_state.last_latency = total_elapsed

                df = pd.DataFrame(all_rows)
                if not df.empty:
                    df = df.rename(columns={"measure": "Measure", "estimated": "Estimated", "actual": "Actual"})
                    df = df[["Measure", "Estimated", "Actual"]]

                st.success("Extraction successful!")
                st.dataframe(df, use_container_width=True, hide_index=True)

                if not df.empty:
                    st.download_button(
                        "Download CSV",
                        df.to_csv(index=False).encode("utf-8"),
                        "financial_extract.csv",
                        "text/csv",
                    )

                st.download_button(
                    "Download JSON",
                    json.dumps({"rows": all_rows}, indent=2),
                    "financial_extract.json",
                    "application/json",
                )

                # Save history
                preview = text.strip().replace("\n", " ")[:120]
                st.session_state.history.insert(0, {"input_preview": preview, "result": {"rows": all_rows}})
                st.session_state.history = st.session_state.history[:5]

            except Exception as e:
                st.error("Extraction failed")
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

if st.session_state.last_latency:
    st.caption(f"⏱️ Last runtime: {st.session_state.last_latency:.2f}s")