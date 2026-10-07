"""Optional FastAPI interface (bonus). The CLI is the primary way to run.

    uvicorn screener.api:app --app-dir src --port 8000

    POST /screen   run the pipeline on uploaded resumes, or on the server-side
                   folder named by RESUME_DIR (default ./resumes) when no
                   files are sent
    GET  /results  the most recent results

Both routes call the same `run_batch` the CLI uses, so behaviour and scores
are identical. There is deliberately no auth, database or frontend here.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from . import config, report
from .config import Settings
from .pipeline import run_batch

app = FastAPI(title="AI Resume Screening", version="1.0.0")
_state: dict = {"results": None}

MAX_FILES = 200
MAX_FILE_BYTES = 10 * 1024 * 1024


def _safe_name(name: str, index: int) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]", "_", Path(name or f"resume_{index}").name)
    return stem or f"resume_{index}"


@app.post("/screen")
async def screen(files: list[UploadFile] | None = File(default=None),
                 use_llm: bool = True, use_github: bool = True) -> dict:
    settings = Settings.from_env()
    if files:
        if len(files) > MAX_FILES:
            raise HTTPException(status_code=413, detail=f"Too many files (limit {MAX_FILES})")
        with tempfile.TemporaryDirectory(prefix="resumes_") as folder:
            for index, upload in enumerate(files):
                name = _safe_name(upload.filename or "", index)
                if Path(name).suffix.lower() not in config.SUPPORTED_EXTENSIONS:
                    continue
                content = await upload.read()
                if len(content) > MAX_FILE_BYTES:
                    raise HTTPException(status_code=413, detail=f"{name} is larger than 10 MB")
                (Path(folder) / name).write_bytes(content)
            batch = await run_batch(Path(folder), settings, use_llm=use_llm, use_github=use_github)
    else:
        folder_path = Path(os.getenv("RESUME_DIR", "resumes"))
        if not folder_path.is_dir():
            raise HTTPException(status_code=400, detail="Upload resume files, or set RESUME_DIR to a folder")
        batch = await run_batch(folder_path, settings, use_llm=use_llm, use_github=use_github)

    _state["results"] = report.build_output(batch)
    return _state["results"]


@app.get("/results")
async def results() -> dict:
    if _state["results"] is None:
        raise HTTPException(status_code=404, detail="No results yet. POST /screen first.")
    return _state["results"]
