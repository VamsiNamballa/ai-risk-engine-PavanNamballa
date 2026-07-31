from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Header

from service.engine.text_extractor import extract_text
from service.engine.chunker import chunk_text
from service.engine.vector_store import store_chunks

router = APIRouter()

@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    x_session_id: Optional[str] = Header(default="default", alias="X-Session-ID")
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    content = await file.read()

    text = extract_text(content)
    if not text.strip():
        raise HTTPException(status_code=422, detail="Could not extract text from PDF.")

    chunks = chunk_text(text)
    session_id = x_session_id or "default"
    stored_count = store_chunks(chunks, session_id=session_id, clear_existing=True)

    return {
        "message": "uploaded successfully",
        "session_id": session_id,
        "chunks_stored": stored_count
    }