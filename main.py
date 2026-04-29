import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from api.routes import router

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

_BASE = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(_BASE / "templates"))

app = FastAPI(title="AI Interview Simulator", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(_BASE / "static")), name="static")
app.include_router(router)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {"request": request},
    )
