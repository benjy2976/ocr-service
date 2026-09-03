from fastapi import FastAPI

from app import jobs as jobs_module
from app.routes_classify import router as classify_router
from app.routes_ocr import router as ocr_router
from app.routes_review import router as review_router

app = FastAPI(title="OCR Pilot Service", version="0.1.0")

app.include_router(ocr_router)
app.include_router(review_router)
app.include_router(classify_router)


@app.on_event("startup")
def _start_background_tasks() -> None:
    jobs_module.start_staging_sweeper()
