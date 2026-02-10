import os
import fitz  # PyMuPDF
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()


class FinancialMetric(BaseModel):
    metric: str = Field(description="Metric name, e.g., Revenue, Net Income, EBITDA.")
    value: str | None = Field(description="Exact value with units/currency if present, else null.")
    period: str | None = Field(description="FY2024, Q3 2023, etc., else null.")
    currency: str | None = Field(description="USD, $, etc., else null.")
    page: int | None = Field(description="1-based page number where evidence appears, else null.")
    evidence: str | None = Field(description="Short supporting quote (max ~25 words), else null.")
    confidence: float | None = Field(description="0 to 1 confidence score for this extraction, else null.")


class ExtractionResult(BaseModel):
    company_name: str | None = Field(description="Company name if found, else null.")
    metrics: list[FinancialMetric] = Field(description="Extracted metrics list.")


def extract_pages_text(pdf_bytes: bytes, max_pages: int = 8) -> list[dict]:
    """Return list of {page: 1-based, text: '...'} for first max_pages."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    pages_to_read = min(len(doc), max_pages)

    for i in range(pages_to_read):
        page = doc.load_page(i)
        pages.append({"page": i + 1, "text": page.get_text("text")})

    doc.close()
    return pages


def extract_financials_llm(pages: list[dict], temperature: float = 0.0) -> dict:
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("Missing GROQ_API_KEY. Put it in your .env file.")

    parser = JsonOutputParser(pydantic_object=ExtractionResult)

    # Combine pages but keep page markers so model can cite pages
    combined = "\n\n".join([f"[PAGE {p['page']}]\n{p['text']}" for p in pages])
    combined = combined[:24000]  # cap for cost/speed

    prompt = ChatPromptTemplate.from_template(
        """
You are a financial document extraction assistant.

Return ONLY valid JSON that matches the schema exactly:
{format_instructions}

Extraction rules:
- Only extract metrics that appear explicitly in the text.
- For each metric, include:
  - page: the 1-based page number (from [PAGE X]) where evidence appears
  - evidence: short quote (~25 words) from that page supporting the value
  - confidence: a number 0.0–1.0 (higher only if evidence is clear)
- Use null when unknown.
- Do not invent values.

Document text:
{text}
"""
    )

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=temperature)
    chain = prompt | llm | parser

    return chain.invoke(
        {"text": combined, "format_instructions": parser.get_format_instructions()}
    )
import json
import pandas as pd
import streamlit as st
from extractor import extract_pages_text, extract_financials_llm

st.set_page_config(page_title="Financial PDF Extractor", page_icon="📄", layout="wide")
st.title("📄 Financial Document Extractor (PDF → Structured JSON)")
st.caption("Upload a financial PDF and extract key metrics with evidence + page numbers.")

with st.sidebar:
    st.header("Settings")
    max_pages = st.slider("Max pages to read", 1, 30, 8)
    temperature = st.slider("LLM temperature", 0.0, 1.0, 0.0, 0.1)
    show_raw_text = st.checkbox("Show extracted raw text (debug)", value=False)

uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded:
    pdf_bytes = uploaded.read()

    with st.status("Reading PDF pages...", expanded=False) as status:
        pages = extract_pages_text(pdf_bytes, max_pages=max_pages)
        status.update(label=f"Loaded {len(pages)} pages ✅", state="complete")

    if show_raw_text:
        st.subheader("Extracted Pages Text (debug)")
        for p in pages:
            with st.expander(f"Page {p['page']} text"):
                st.text(p["text"][:4000])

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("Run extraction")
        if st.button("Extract Financial Metrics", type="primary"):
            with st.spinner("Extracting…"):
                try:
                    result = extract_financials_llm(pages, temperature=temperature)
                    st.session_state["result"] = result
                    st.success("Extraction complete ✅")
                except Exception as e:
                    st.error(f"Extraction failed: {e}")

    with col2:
        st.subheader("Results")
        result = st.session_state.get("result")

        if result:
            st.write("**Company:**", result.get("company_name"))

            metrics = result.get("metrics", [])
            if metrics:
                df = pd.DataFrame(metrics)

                # Nice table for recruiters: sort by confidence
                if "confidence" in df.columns:
                    df = df.sort_values(by="confidence", ascending=False)

                st.dataframe(df, use_container_width=True)

                # Evidence cards
                st.markdown("### Evidence")
                for m in metrics:
                    title = f"{m.get('metric')} — {m.get('value')} (p.{m.get('page')})"
                    conf = m.get("confidence")
                    with st.expander(f"{title} | confidence: {conf}"):
                        st.write(m.get("evidence"))

                st.markdown("### Downloads")

                # Download JSON
                st.download_button(
                    "Download JSON",
                    data=json.dumps(result, indent=2),
                    file_name="extracted_financials.json",
                    mime="application/json",
                )

                # Download CSV
                st.download_button(
                    "Download CSV",
                    data=df.to_csv(index=False),
                    file_name="extracted_financials.csv",
                    mime="text/csv",
                )
            else:
                st.info("No metrics found. Try increasing max pages or upload a clearer PDF.")
else:
    st.info("Upload a PDF to begin.")
