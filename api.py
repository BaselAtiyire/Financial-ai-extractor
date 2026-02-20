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

# -----------------------
# Load env + create app
# -----------------------
load_dotenv()

app = FastAPI(
    title="Financial Extractor API",
    description="Extracts simple financial metrics (actual vs estimated) from text.",
    version="1.0.0",
)

# -----------------------
# 1) Swagger API key scheme (shows Authorize button)
# -----------------------
API_KEY = os.getenv("API_KEY", "").strip()
api_key_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)

# -----------------------
# 2) Auth checker
# -----------------------
def require_api_key(api_key: Optional[str]) -> None:
    """
    If API_KEY is set, require the request header X-API-Key to match it.
    If API_KEY is empty, auth is disabled.
    """
    if API_KEY:
        if not api_key or api_key != API_KEY:
            raise HTTPException(status_code=401, detail="Unauthorized")

# -----------------------
# DB (SQLite)
# -----------------------
DB_PATH = os.getenv("DB_PATH", "runs.db")

def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            input_text TEXT NOT NULL,
            output_json TEXT NOT NULL,
            latency_seconds REAL NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()

def save_run(input_text: str, output_obj: dict, latency: float) -> int:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    created_at = datetime.utcnow().isoformat()
    output_json = json.dumps(output_obj, ensure_ascii=False)
    cur.execute(
        "INSERT INTO runs (created_at, input_text, output_json, latency_seconds) VALUES (?, ?, ?, ?)",
        (created_at, input_text, output_json, float(latency)),
    )
    conn.commit()
    run_id = cur.lastrowid
    conn.close()
    return int(run_id)

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
# LLM chain
# -----------------------
MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# IMPORTANT: GROQ_API_KEY must exist in env (.env or container env)
llm = ChatGroq(model=MODEL, temperature=0)
parser = JsonOutputParser()

prompt = ChatPromptTemplate.from_template(
    """
You extract financial metrics comparing estimates vs actuals.

Return ONLY valid JSON in EXACTLY this format (no extra keys):
{{
  "rows": [
    {{
      "measure": "Revenue" | "EPS" | "Other",
      "estimated": string | null,
      "actual": string | null
    }}
  ]
}}

Rules:
- If text says "Revenue: $94.93 billion vs. $94.58 billion estimated":
  - measure="Revenue", actual="$94.93 billion", estimated="$94.58 billion"
- If text says "Earnings per share: $1.64, adjusted, versus $1.60 estimated":
  - measure="EPS", actual="$1.64", estimated="$1.60"
- If you can’t find estimated or actual for a measure, use null.
- Keep values as strings exactly as shown (include $ and units).

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
        "model": MODEL,
        "db_path": DB_PATH,
    }

@app.post("/extract", response_model=ExtractedResult)
def extract(
    req: ExtractRequest,
    api_key: Optional[str] = Security(api_key_scheme),  # ✅ makes Swagger show Authorize
):
    require_api_key(api_key)

    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty text")

    start = time.perf_counter()

    # 1) LLM call (catch Groq/LLM failures)
    try:
        raw = chain.invoke({"text": req.text})
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {str(e)}")

    latency = time.perf_counter() - start

    # 2) Validate JSON structure
    try:
        validated = ExtractedResult(**raw)
    except ValidationError as ve:
        raise HTTPException(status_code=422, detail=f"Schema validation failed: {str(ve)}")

    # 3) Save run (catch DB issues)
    try:
        save_run(req.text, validated.model_dump(), latency)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB save failed: {str(e)}")

    return validated

@app.get("/runs")
def list_runs(
    limit: int = 10,
    api_key: Optional[str] = Security(api_key_scheme),
):
    require_api_key(api_key)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT id, created_at, substr(input_text,1,120), latency_seconds "
        "FROM runs ORDER BY id DESC LIMIT ?",
        (int(limit),),
    )
    rows = cur.fetchall()
    conn.close()

    return [
        {
            "id": r[0],
            "created_at": r[1],
            "input_preview": r[2],
            "latency_seconds": r[3],
        }
        for r in rows
    ]

@app.get("/runs/{run_id}")
def get_run(
    run_id: int,
    api_key: Optional[str] = Security(api_key_scheme),
):
    require_api_key(api_key)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT id, created_at, input_text, output_json, latency_seconds "
        "FROM runs WHERE id=?",
        (int(run_id),),
    )
    row = cur.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Run not found")

    return {
        "id": row[0],
        "created_at": row[1],
        "input_text": row[2],
        "output": json.loads(row[3]),
        "latency_seconds": row[4],
    }
