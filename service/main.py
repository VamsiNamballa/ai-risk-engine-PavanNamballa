from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os

from service.api.upload_api import router as upload_router
from service.api.ask_api import router as ask_router
from service.api.pipeline_api import router as pipeline_router
from service.engine.response_generator import active_model
from service.engine.db_store import get_assessment_history

app = FastAPI(title="AI Risk Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router)
app.include_router(ask_router)
app.include_router(pipeline_router)

static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/config")
def config():
    return {"model": active_model()}

@app.get("/history")
def history(limit: int = 20):
    return {"history": get_assessment_history(limit=limit)}

@app.get("/")
def home():
    return FileResponse(os.path.join(static_dir, "index.html"))