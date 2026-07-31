from typing import Optional
from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel

from service.engine.vector_store import search_chunks
from service.engine.response_generator import generate_response

router = APIRouter()

class Question(BaseModel):
    query: str
    session_id: Optional[str] = "default"


@router.post("/ask")
def ask_question(
    payload: Question,
    x_session_id: Optional[str] = Header(default=None, alias="X-Session-ID")
):
    query = payload.query
    session_id = payload.session_id if payload.session_id != "default" else (x_session_id or "default")

    # Step 1: retrieve relevant chunks from session's isolated vector store
    retrieved_chunks = search_chunks(query, session_id=session_id)

    # Step 2: generate response + risk score
    try:
        result = generate_response(query, retrieved_chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "question": query,
        "session_id": session_id,
        "executive_summary": result.get("executive_summary"),
        "key_risks": result.get("key_risks", []),
        "recommendation": result.get("recommendation"),
        "risk_score": result.get("risk_score"),
        "risk_level": result.get("risk_level"),
        "confidence": result.get("confidence"),
        "sources": retrieved_chunks,
    }