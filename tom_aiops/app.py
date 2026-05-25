from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any, Dict

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import OPENAI_API_KEY, OPENAI_MODEL
from .model_service import TomatoAIOpsService, available_targets
from .report_service import TomatoReportService


BASE_DIR = Path(__file__).resolve().parent
DATA_BASE = BASE_DIR.parent / "data" / "분석.csv"
UPLOAD_DIR = BASE_DIR / "uploaded"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

service = TomatoAIOpsService(base_dir=BASE_DIR, base_analysis_csv=DATA_BASE)
report_service = TomatoReportService(api_key=OPENAI_API_KEY, model=OPENAI_MODEL)

app = FastAPI(title="Tomato AIOps")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(BASE_DIR)), name="static")


class ReportRequest(BaseModel):
    analysis_result: Dict[str, Any]


@app.get("/")
def root():
    return FileResponse(str(BASE_DIR / "index_farm_improved.html"))


@app.get("/api/targets")
def get_targets():
    if not DATA_BASE.exists():
        raise HTTPException(status_code=404, detail=f"분석.csv not found: {DATA_BASE}")
    all_targets = available_targets(DATA_BASE)
    supported = service.get_supported_targets()
    return {"targets": all_targets, "supported_targets": supported}


async def _analyze_core(file: UploadFile, target_col: str):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="CSV 파일만 업로드 가능합니다.")

    save_path = UPLOAD_DIR / file.filename
    with save_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        result = service.run_inference(save_path, target_col)
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), target_col: str = Form(...)):
    return await _analyze_core(file, target_col)


@app.post("/upload")
async def analyze_legacy(file: UploadFile = File(...), target_col: str = Form(...)):
    # legacy frontend compatibility
    return await _analyze_core(file, target_col)


@app.post("/api/retrain")
def retrain(target_col: str = Form(...)):
    try:
        meta = service.train_target(target_col)
        return {"status": "ok", "meta": meta}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/report")
def generate_report(req: ReportRequest):
    try:
        report = report_service.generate_report(req.analysis_result)
        return {"status": "ok", **report}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
