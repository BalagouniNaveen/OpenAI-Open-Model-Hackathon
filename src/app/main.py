from fastapi import FastAPI
from . import db
from .ml import detector
from .api import router
from fastapi.responses import HTMLResponse
from .dashboard import templates
from pathlib import Path
import os

app = FastAPI(title="IoT + ML + RPA (gpt-oss demo)")

@app.on_event("startup")
def startup():
    db.init_db()
    detector.init_detector()

app.include_router(router, prefix="")

# simple page
@app.get("/", response_class=HTMLResponse)
def homepage():
    p = Path(__file__).parent / "dashboard" / "templates" / "index.html"
    return p.read_text()
