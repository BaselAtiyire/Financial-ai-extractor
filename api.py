import os
import time
import json
import sqlite3
from datetime import datetime
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, ValidationError

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

app = FastAPI(title="Financial Extractor API")

# -----------------------
# Auth (Swagger Authorize)
# -----------------------
API_KEY = os.getenv("API_KEY", "").strip()
api_key_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)

def require_api_key(api_key: Optional[str]) -> None:
    if API_KEY and api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

# -----------------------
# DB
# -----------------------
DB_PATH = os.getenv("DB_PATH", "runs.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            input_text TEXT,
            output_json TEXT,
            latency_seconds REAL
        )
    """)
    conn.commit()
    conn.close()

def save_run(input_text: str, output: dict, latency: float):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO runs (created_at, input_text, output_json, latency_seconds) VALUES (?, ?, ?, ?)",
        (datetime.utcnow().isoformat(), input_text, json.dumps(output), latency)
    )
    conn.commit()
    conn.close()

init_db()

# -----------------------
# Schemas
# -----------------------
class ExtractRequest(BaseModel):
    text: str

class MetricRow(BaseModel):
    measure: str
    estimated: Optional[str] = None
    actual: Optional[str] = None

class ExtractedResult(BaseModel):
    rows: List[MetricRow]

# -----------------------
# LLM Chain
# -----------------------
MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

llm = ChatGroq(model=MODEL, temperature=0)
parser = JsonOutputParser()

prompt = ChatPromptTemplate.from_template(
    """
You extract financial metrics comparing estimates vs actuals.

Return ONLY valid JSON in EXACTLY this format:
{
  "rows": [
    {"measure": "Revenue" | "EPS" | "Other", "estimated": string | null, "actual": string | null}
  ]
}

Text:
{text}

{format_instructions}
"""
).partial(format_instructions=parser.get_format_instructions())

chain = prompt | llm | parser

# -----------------------
# Routes
# -----------------------
@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "auth_enabled": bool(API_KEY),
        "groq_key_loaded": bool(os.getenv("GROQ_API_KEY")),
        "model": MODEL
    }

@app.post("/extract", response_model=ExtractedResult)
def extract(req: ExtractRequest, api_key: Optional[str] = Security(api_key_scheme)):
    require_api_key(api_key)

    start = time.perf_counter()
    try:
        raw = chain.invoke({"text": req.text})
        validated = ExtractedResult(**raw)
    except ValidationError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {str(e)}")

    latency = time.perf_counter() - start
    save_run(req.text, validated.model_dump(), latency)
    return validated

@app.get("/runs")
def runs(limit: int = 10, api_key: Optional[str] = Security(api_key_scheme)):
    require_api_key(api_key)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, created_at, substr(input_text,1,120), latency_seconds FROM runs ORDER BY id DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    return [{"id": r[0], "created_at": r[1], "input_preview": r[2], "latency_seconds": r[3]} for r in rows]
