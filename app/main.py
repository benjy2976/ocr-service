import os
import uuid
import fcntl
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, HttpUrl
from app.ocr_pipeline import DEFAULT_OUT_DIR, run_ocr, run_ocr_file, run_stamp_test
import json
import time

app = FastAPI(title="OCR Pilot Service", version="0.1.0")


class OCRRequest(BaseModel):
    url: HttpUrl
    mode: str | None = None
    lang: str | None = None
    deskew: bool | None = None
    clean: bool | None = None
    remove_vectors: bool | None = None
    psm: str | None = None
    jobs: int | None = None
    mask_stamps: bool | None = None
    mask_signatures: bool | None = None
    mask_grayscale: bool | None = None
    mask_dilate: int | None = None
    stamp_min_area: float | None = None
    stamp_max_area: float | None = None
    stamp_circularity: float | None = None
    stamp_rect_aspect_min: float | None = None
    stamp_rect_aspect_max: float | None = None
    signature_region: float | None = None


class OCRLocalRequest(BaseModel):
    path: str
    mode: str | None = None
    lang: str | None = None
    deskew: bool | None = None
    clean: bool | None = None
    remove_vectors: bool | None = None
    psm: str | None = None
    jobs: int | None = None
    mask_stamps: bool | None = None
    mask_signatures: bool | None = None
    mask_grayscale: bool | None = None
    mask_dilate: int | None = None
    stamp_min_area: float | None = None
    stamp_max_area: float | None = None
    stamp_circularity: float | None = None
    stamp_rect_aspect_min: float | None = None
    stamp_rect_aspect_max: float | None = None
    signature_region: float | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ocr")
def ocr(req: OCRRequest):
    try:
        result = run_ocr(
            url=str(req.url),
            mode=req.mode,
            lang=req.lang,
            deskew=req.deskew,
            clean=req.clean,
            remove_vectors=req.remove_vectors,
            psm=req.psm,
            jobs=req.jobs,
            mask_stamps=req.mask_stamps,
            mask_signatures=req.mask_signatures,
            mask_grayscale=req.mask_grayscale,
            mask_dilate=req.mask_dilate,
            stamp_min_area=req.stamp_min_area,
            stamp_max_area=req.stamp_max_area,
            stamp_circularity=req.stamp_circularity,
            stamp_rect_aspect_min=req.stamp_rect_aspect_min,
            stamp_rect_aspect_max=req.stamp_rect_aspect_max,
            signature_region=req.signature_region,
        )
        output_pdf = Path(result["output_pdf"])
        output_name = output_pdf.name
        result["output_filename"] = output_name
        result["download_path"] = f"/file/{output_name}"
        public_base_url = os.getenv("PUBLIC_BASE_URL")
        if public_base_url:
            result["download_url"] = f"{public_base_url.rstrip('/')}/file/{output_name}"
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def _resolve_local_path(path_str: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    local_root = os.getenv("OCR_LOCAL_ROOT")
    if local_root:
        root = Path(local_root).expanduser().resolve()
        if not path.is_relative_to(root):
            raise HTTPException(status_code=400, detail="Path outside OCR_LOCAL_ROOT")
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=400, detail="File not found")
    if path.suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    return path


@app.post("/ocr/local")
def ocr_local(req: OCRLocalRequest):
    try:
        path = _resolve_local_path(req.path)
        result = run_ocr_file(
            path,
            mode=req.mode,
            lang=req.lang,
            deskew=req.deskew,
            clean=req.clean,
            remove_vectors=req.remove_vectors,
            psm=req.psm,
            jobs=req.jobs,
            mask_stamps=req.mask_stamps,
            mask_signatures=req.mask_signatures,
            mask_grayscale=req.mask_grayscale,
            mask_dilate=req.mask_dilate,
            stamp_min_area=req.stamp_min_area,
            stamp_max_area=req.stamp_max_area,
            stamp_circularity=req.stamp_circularity,
            stamp_rect_aspect_min=req.stamp_rect_aspect_min,
            stamp_rect_aspect_max=req.stamp_rect_aspect_max,
            signature_region=req.signature_region,
        )
        output_pdf = Path(result["output_pdf"])
        output_name = output_pdf.name
        result["output_filename"] = output_name
        result["download_path"] = f"/file/{output_name}"
        public_base_url = os.getenv("PUBLIC_BASE_URL")
        if public_base_url:
            result["download_url"] = f"{public_base_url.rstrip('/')}/file/{output_name}"
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/ocr/file")
def ocr_file(
    file: UploadFile = File(...),
    mode: str | None = Form(None),
    lang: str | None = Form(None),
    deskew: bool | None = Form(None),
    clean: bool | None = Form(None),
    remove_vectors: bool | None = Form(None),
    psm: str | None = Form(None),
    jobs: int | None = Form(None),
    mask_stamps: bool | None = Form(None),
    mask_signatures: bool | None = Form(None),
    mask_grayscale: bool | None = Form(None),
    mask_dilate: int | None = Form(None),
    stamp_min_area: float | None = Form(None),
    stamp_max_area: float | None = Form(None),
    stamp_circularity: float | None = Form(None),
    stamp_rect_aspect_min: float | None = Form(None),
    stamp_rect_aspect_max: float | None = Form(None),
    signature_region: float | None = Form(None),
):
    try:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        tmp_dir = Path(os.getenv("OCR_TMP_DIR", "/data/tmp"))
        tmp_dir.mkdir(parents=True, exist_ok=True)
        token = uuid.uuid4().hex[:12]
        safe_name = Path(file.filename).name
        dst = tmp_dir / f"{token}_{safe_name}"
        with dst.open("wb") as f:
            while True:
                chunk = file.file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        result = run_ocr_file(
            dst,
            mode=mode,
            lang=lang,
            deskew=deskew,
            clean=clean,
            remove_vectors=remove_vectors,
            psm=psm,
            jobs=jobs,
            mask_stamps=mask_stamps,
            mask_signatures=mask_signatures,
            mask_grayscale=mask_grayscale,
            mask_dilate=mask_dilate,
            stamp_min_area=stamp_min_area,
            stamp_max_area=stamp_max_area,
            stamp_circularity=stamp_circularity,
            stamp_rect_aspect_min=stamp_rect_aspect_min,
            stamp_rect_aspect_max=stamp_rect_aspect_max,
            signature_region=signature_region,
        )
        output_pdf = Path(result["output_pdf"])
        output_name = output_pdf.name
        result["output_filename"] = output_name
        result["download_path"] = f"/file/{output_name}"
        public_base_url = os.getenv("PUBLIC_BASE_URL")
        if public_base_url:
            result["download_url"] = f"{public_base_url.rstrip('/')}/file/{output_name}"
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/file/{filename}")
def download_file(filename: str):
    # Prevent path traversal: only allow plain file names
    if "/" in filename or "\\" in filename or filename.startswith("."):
        raise HTTPException(status_code=400, detail="Invalid filename")
    out_dir = Path(DEFAULT_OUT_DIR)
    file_path = out_dir / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=str(file_path), media_type="application/pdf", filename=filename)


@app.post("/stamps/test")
def stamps_test():
    try:
        result = run_stamp_test()
        output_image = Path(result["output_image"])
        result["output_filename"] = output_image.name
        result["download_path"] = f"/image/{output_image.name}"
        public_base_url = os.getenv("PUBLIC_BASE_URL")
        if public_base_url:
            result["download_url"] = f"{public_base_url.rstrip('/')}/image/{output_image.name}"
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/image/{filename}")
def download_image(filename: str):
    if "/" in filename or "\\" in filename or filename.startswith("."):
        raise HTTPException(status_code=400, detail="Invalid filename")
    out_dir = Path(DEFAULT_OUT_DIR)
    file_path = out_dir / filename
    if not file_path.exists():
        candidates = list(out_dir.glob(f"**/{filename}"))
        if candidates:
            file_path = candidates[0]
        else:
            raise HTTPException(status_code=404, detail="File not found")
    suffix = file_path.suffix.lower()
    if suffix == ".png":
        media_type = "image/png"
    elif suffix in (".jpg", ".jpeg"):
        media_type = "image/jpeg"
    else:
        media_type = "application/octet-stream"
    return FileResponse(path=str(file_path), media_type=media_type, filename=filename)


def _review_state_path() -> Path:
    return Path(DEFAULT_OUT_DIR) / "stamp_pages" / "state.json"


def _review_lock_ttl_sec() -> int:
    raw = os.getenv("REVIEW_LOCK_TTL_MIN", "30")
    try:
        return max(1, int(raw)) * 60
    except ValueError:
        return 1800


def _load_review_state() -> dict:
    path = _review_state_path()
    if not path.exists():
        return {"items": {}}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {"items": {}}


def _save_review_state(state: dict) -> None:
    path = _review_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2))
    tmp.replace(path)


def _normalize_state(state: dict) -> dict:
    ttl = _review_lock_ttl_sec()
    now = time.time()
    items = state.get("items", {})
    for name, info in items.items():
        if info.get("status") == "in_process":
            locked_at = info.get("locked_at", 0)
            if now - locked_at > ttl:
                info["status"] = "pending"
                info["user"] = ""
                info["locked_at"] = 0
    state["items"] = items
    return state


def _list_review_images() -> list[str]:
    images_dir = Path(DEFAULT_OUT_DIR) / "stamp_pages" / "images"
    if not images_dir.exists():
        return []
    return [
        p.name
        for p in sorted(images_dir.iterdir())
        if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    ]


_review_total_cache: int | None = None


def _review_total_pages() -> int:
    global _review_total_cache
    if _review_total_cache is not None:
        return _review_total_cache
    pages_csv = Path(DEFAULT_OUT_DIR) / "stamp_pages" / "pages.csv"
    if not pages_csv.exists():
        _review_total_cache = 0
        return _review_total_cache
    try:
        with pages_csv.open("r", encoding="utf-8") as f:
            total = sum(1 for _ in f) - 1
    except OSError:
        total = 0
    _review_total_cache = max(total, 0)
    return _review_total_cache


def _text_review_dir() -> Path:
    return Path(DEFAULT_OUT_DIR) / "text_pages"


def _text_review_state_path() -> Path:
    return _text_review_dir() / "state.json"


def _load_text_review_state() -> dict:
    path = _text_review_state_path()
    if not path.exists():
        return {"items": {}}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {"items": {}}


def _save_text_review_state(state: dict) -> None:
    path = _text_review_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2))
    tmp.replace(path)


def _normalize_text_review_state(state: dict) -> dict:
    ttl = _review_lock_ttl_sec()
    now = time.time()
    items = state.get("items", {})
    for _name, info in items.items():
        if info.get("status") == "in_process":
            locked_at = info.get("locked_at", 0)
            if now - locked_at > ttl:
                info["status"] = "pending"
                info["user"] = ""
                info["locked_at"] = 0
    state["items"] = items
    return state


def _list_text_review_images() -> list[str]:
    images_dir = _text_review_dir() / "images"
    if not images_dir.exists():
        return []
    return [
        p.name
        for p in sorted(images_dir.iterdir())
        if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    ]


_text_review_total_cache: int | None = None


def _text_review_total_pages() -> int:
    global _text_review_total_cache
    if _text_review_total_cache is not None:
        return _text_review_total_cache
    pages_csv = _text_review_dir() / "pages.csv"
    if not pages_csv.exists():
        _text_review_total_cache = 0
        return _text_review_total_cache
    try:
        with pages_csv.open("r", encoding="utf-8") as f:
            total = sum(1 for _ in f) - 1
    except OSError:
        total = 0
    _text_review_total_cache = max(total, 0)
    return _text_review_total_cache


def _text_review_labels_auto_dir() -> Path:
    return _text_review_dir() / "labels_auto"


def _text_review_labels_reviewed_dir() -> Path:
    return _text_review_dir() / "labels_reviewed"


def _text_review_stats_payload(*, include_skipped: bool = True) -> dict:
    state = _normalize_text_review_state(_load_text_review_state())
    items = state.get("items", {})
    validated = 0
    skipped = 0
    per_user: dict[str, int] = {}
    for meta in items.values():
        status = meta.get("status")
        if status == "validated":
            validated += 1
            user = meta.get("user") or "anon"
            per_user[user] = per_user.get(user, 0) + 1
        elif status == "skipped":
            skipped += 1
    users = [
        {"user": u, "count": c}
        for u, c in sorted(per_user.items(), key=lambda item: item[1], reverse=True)
    ]
    payload = {"validated": validated, "users": users}
    if include_skipped:
        payload["skipped"] = skipped
    return payload


def _classify_dir() -> Path:
    return Path(os.getenv("CLASSIFY_DIR", "/data/classify"))


def _classify_state_path() -> Path:
    return _classify_dir() / "state.json"


def _classify_preds_path() -> Path:
    return _classify_dir() / "preds.json"


def _classify_lock_ttl_sec() -> int:
    raw = os.getenv("CLASSIFY_LOCK_TTL_MIN", "30")
    try:
        return max(1, int(raw)) * 60
    except ValueError:
        return 1800


def _classify_conf_threshold() -> float:
    raw = os.getenv("CLASSIFY_CONF_THRESHOLD", "0.99")
    try:
        return float(raw)
    except ValueError:
        return 0.99


def _load_classify_preds() -> dict:
    preds_path = _classify_preds_path()
    if not preds_path.exists():
        return {}
    try:
        return json.loads(preds_path.read_text())
    except Exception:
        return {}


def _load_classify_state() -> dict:
    path = _classify_state_path()
    if not path.exists():
        return {"items": {}}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {"items": {}}


def _save_classify_state(state: dict) -> None:
    path = _classify_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(".lock")
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        if path.exists():
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backup = path.with_name(f"{path.stem}.{ts}.json")
            try:
                backup.write_text(path.read_text())
            except Exception:
                pass
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2))
        tmp.replace(path)
        fcntl.flock(lock_file, fcntl.LOCK_UN)


def _normalize_classify_state(state: dict) -> dict:
    ttl = _classify_lock_ttl_sec()
    now = time.time()
    items = state.get("items", {})
    for name, info in items.items():
        if info.get("status") == "in_process":
            locked_at = info.get("locked_at", 0)
            if now - locked_at > ttl:
                info["status"] = "pending"
                info["user"] = ""
                info["locked_at"] = 0
    state["items"] = items
    return state


def _list_classify_crops() -> list[str]:
    crops_dir = _classify_dir() / "crops"
    if not crops_dir.exists():
        return []
    return [
        p.name
        for p in sorted(crops_dir.iterdir())
        if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    ]


def _classify_rejected_dir() -> Path:
    return _classify_dir() / "rejected"


@app.get("/stamps/review", response_class=HTMLResponse)
def stamps_review():
    review_version = os.getenv("REVIEW_APP_VERSION", "0")
    html = """
<!doctype html>
<html lang="es">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Stamp Review</title>
    <style>
      html, body { height: 100%; }
      body { font-family: Arial, sans-serif; margin: 2px 16px; overflow: hidden; }
      #canvas { border: 1px solid #ccc; }
      .layout { display: flex; gap: 16px; align-items: flex-start; height: calc(100vh - 8px); }
      .sidebar { width: 240px; display: flex; flex-direction: column; gap: 8px; overflow-y: auto; max-height: 100%; min-height: 0; }
      .content { flex: 1; overflow: auto; max-height: 100%; min-height: 0; }
      .row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
      .btn { padding: 8px 12px; border: 1px solid #333; background: #f2f2f2; cursor: pointer; }
      .btn:disabled { opacity: 0.5; cursor: default; }
      .btn.active { background: #ffd966; border-color: #b59b00; }
      .btn.suggested { background: beige; border-color: #c89f00; font-weight: 700; }
      .btn.suggested { background: beige; border-color: #c89f00; font-weight: 700; }
      .meta { font-size: 12px; color: #555; }
      .list { display: flex; flex-direction: column; gap: 6px; max-height: 300px; overflow: auto; border: 1px solid #ddd; padding: 6px; }
      .item { display: flex; align-items: center; gap: 6px; font-size: 12px; }
      .swatch { width: 14px; height: 14px; border: 1px solid #333; }
      .badge { font-size: 11px; color: #333; }
    </style>
  </head>
  <body>
    <div class="layout">
      <div class="sidebar">
        <h3>Revision de sellos</h3>
        <div class="row"></div>
        <div class="row">
          <button class="btn" id="zoomOutBtn">Zoom -</button>
          <button class="btn" id="zoomInBtn">Zoom +</button>
        </div>
        <button class="btn" id="validateBtn">Validar</button>
        <button class="btn" id="addBtn">Agregar sello</button>
        <button class="btn" id="cancelBtn">Cancelar agregar</button>
        <div class="meta" id="meta"></div>
        <div class="meta" id="userMeta"></div>
        <button class="btn" id="changeUserBtn">Cambiar usuario</button>
        <div class="list" id="boxList"></div>
        <p class="meta">Usa la lista para activar/desactivar cajas.</p>
        <p class="meta">Para agregar: click en “Agregar sello” y luego 2 clicks en la imagen.</p>
        <p class="meta" id="statsMeta"></p>
        <div class="meta" id="userStats"></div>
      </div>
      <div class="content">
        <canvas id="canvas"></canvas>
      </div>
    </div>
    <script>
      const REVIEW_VERSION = "__REVIEW_VERSION__";
      const canvas = document.getElementById('canvas');
      const ctx = canvas.getContext('2d');
      const meta = document.getElementById('meta');
      const validateBtn = document.getElementById('validateBtn');
      const addBtn = document.getElementById('addBtn');
      const cancelBtn = document.getElementById('cancelBtn');
      const boxList = document.getElementById('boxList');
      const zoomOutBtn = document.getElementById('zoomOutBtn');
      const zoomInBtn = document.getElementById('zoomInBtn');
      const userMeta = document.getElementById('userMeta');
      const changeUserBtn = document.getElementById('changeUserBtn');
      const statsMeta = document.getElementById('statsMeta');
      const userStats = document.getElementById('userStats');

      let SCALE = 0.5;
      let items = [];
      let idx = 0;
      let currentName = '';
      let totalPages = 0;
      let image = new Image();
      let boxes = [];
      let removed = new Set();
      let addMode = false;
      let addPoints = [];
      let selected = null;
      let dragMode = null;
      const HANDLE = 10;
      const HANDLE_HIT = 20;
      let hoverHandle = null;

      function draw() {
        canvas.width = image.width;
        canvas.height = image.height;
        canvas.style.width = (image.width * SCALE) + 'px';
        canvas.style.height = (image.height * SCALE) + 'px';
        ctx.drawImage(image, 0, 0);
        ctx.lineWidth = 5;
        boxes.forEach((b, i) => {
          const key = String(i);
          if (removed.has(key)) {
            ctx.strokeStyle = 'rgba(200,0,0,0.35)';
          } else {
            ctx.strokeStyle = 'rgba(0,128,0,0.8)';
          }
          ctx.strokeRect(b.x, b.y, b.w, b.h);
          const label = String(i + 1);
          ctx.font = '20px Arial';
          const pad = 6;
          const textW = ctx.measureText(label).width;
          const textH = 20;
          const lx = b.x + 10;
          const ly = b.y + 26;
          ctx.fillStyle = 'rgba(255,255,255,0.75)';
          ctx.fillRect(lx - pad, ly - textH + 4, textW + pad * 2, textH + pad);
          ctx.fillStyle = 'rgba(0,0,0,0.9)';
          ctx.fillText(label, lx, ly);
        });
        ctx.fillStyle = 'rgba(0,0,255,0.9)';
        boxes.forEach((b) => {
          const pts = [
            [b.x, b.y],
            [b.x + b.w, b.y],
            [b.x, b.y + b.h],
            [b.x + b.w, b.y + b.h],
          ];
          pts.forEach(([px, py]) => {
            ctx.fillRect(px - HANDLE, py - HANDLE, HANDLE * 2, HANDLE * 2);
          });
        });
        ctx.lineWidth = 5;
        if (addPoints.length === 1) {
          ctx.strokeStyle = 'rgba(0,0,200,0.8)';
          ctx.strokeRect(addPoints[0].x - 10, addPoints[0].y - 10, 20, 20);
        }
      }

      function loadItem() {
        if (!items.length) return;
        removed = new Set();
        const item = items[idx];
        currentName = item.name;
        meta.textContent = item.name ? item.name : '';
        image.onload = draw;
        image.src = `/image/${encodeURIComponent(item.name)}`;
        fetch(`/stamps/review/labels?name=${encodeURIComponent(item.name)}`)
          .then(r => r.json())
          .then(data => { boxes = data.boxes; renderList(); draw(); });
        updateControls();
      }

      function updateControls() {
        validateBtn.disabled = addMode;
        cancelBtn.disabled = !addMode;
        addBtn.classList.toggle('active', addMode);
      }

      function refreshStats() {
        fetch('/stamps/review/stats')
          .then(r => r.json())
          .then(data => {
            statsMeta.textContent = `Validadas: ${data.validated} / Total: ${totalPages}`;
            userStats.innerHTML = '';
            if (data.users && data.users.length) {
              const title = document.createElement('div');
              title.textContent = 'Usuarios:';
              userStats.appendChild(title);
              data.users.forEach((u) => {
                const row = document.createElement('div');
                row.textContent = `${u.user}: ${u.count}`;
                userStats.appendChild(row);
              });
            }
          });
      }

      function checkVersionAndReload() {
        return fetch('/stamps/review/version')
          .then(r => r.json())
          .then(data => {
            if (data.version && data.version !== REVIEW_VERSION) {
              location.reload();
              return true;
            }
            return false;
          })
          .catch(() => false);
      }

      function renderList() {
        boxList.innerHTML = '';
        boxes.forEach((_, i) => {
          const row = document.createElement('div');
          row.className = 'item';
          const swatch = document.createElement('div');
          swatch.className = 'swatch';
          const isRemoved = removed.has(String(i));
          swatch.style.background = isRemoved ? 'rgba(200,0,0,0.35)' : 'rgba(0,128,0,0.8)';
          const label = document.createElement('span');
          label.className = 'badge';
          label.textContent = `Caja ${i + 1}`;
          const toggle = document.createElement('input');
          toggle.type = 'checkbox';
          toggle.checked = !isRemoved;
          toggle.addEventListener('change', () => {
            const key = String(i);
            if (toggle.checked) removed.delete(key);
            else removed.add(key);
            renderList();
            draw();
          });
          const del = document.createElement('button');
          del.className = 'btn';
          del.textContent = 'Eliminar';
          del.addEventListener('click', () => {
            removeBox(i);
            renderList();
            draw();
          });
          row.appendChild(swatch);
          row.appendChild(label);
          row.appendChild(toggle);
          row.appendChild(del);
          boxList.appendChild(row);
        });
      }

      function removeBox(index) {
        boxes = boxes.filter((_, i) => i !== index);
        const nextRemoved = new Set();
        boxes.forEach((_, i) => {
          const oldIndex = i >= index ? i + 1 : i;
          if (removed.has(String(oldIndex))) nextRemoved.add(String(i));
        });
        removed = nextRemoved;
      }

      function toCanvasPoint(e) {
        const rect = canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left) * (canvas.width / rect.width);
        const y = (e.clientY - rect.top) * (canvas.height / rect.height);
        return { x, y };
      }

      function hitHandle(b, x, y) {
        const handles = {
          tl: [b.x, b.y],
          tr: [b.x + b.w, b.y],
          bl: [b.x, b.y + b.h],
          br: [b.x + b.w, b.y + b.h],
        };
        for (const [key, [hx, hy]] of Object.entries(handles)) {
          if (Math.abs(x - hx) <= HANDLE_HIT && Math.abs(y - hy) <= HANDLE_HIT) {
            return key;
          }
        }
        return null;
      }

      canvas.addEventListener('mousedown', (e) => {
        const { x, y } = toCanvasPoint(e);

        if (addMode) {
          addPoints.push({ x, y });
          if (addPoints.length === 2) {
            const x1 = Math.min(addPoints[0].x, addPoints[1].x);
            const y1 = Math.min(addPoints[0].y, addPoints[1].y);
            const x2 = Math.max(addPoints[0].x, addPoints[1].x);
            const y2 = Math.max(addPoints[0].y, addPoints[1].y);
            boxes.push({ x: x1, y: y1, w: x2 - x1, h: y2 - y1 });
            addPoints = [];
            addMode = false;
            renderList();
            updateControls();
          }
          draw();
          return;
        }

        selected = null;
        dragMode = null;
        for (let i = 0; i < boxes.length; i++) {
          const h = hitHandle(boxes[i], x, y);
          if (h) {
            selected = i;
            dragMode = h;
            break;
          }
          if (x >= boxes[i].x && x <= boxes[i].x + boxes[i].w && y >= boxes[i].y && y <= boxes[i].y + boxes[i].h) {
            selected = i;
          }
        }
        draw();
      });

      canvas.addEventListener('mousemove', (e) => {
        const { x, y } = toCanvasPoint(e);
          if (!dragMode) {
          hoverHandle = null;
          if (selected !== null && boxes[selected]) {
            const h = hitHandle(boxes[selected], x, y);
            if (h) hoverHandle = h;
          }
          if (hoverHandle === 'tl' || hoverHandle === 'br') {
            canvas.style.cursor = 'nwse-resize';
          } else if (hoverHandle === 'tr' || hoverHandle === 'bl') {
            canvas.style.cursor = 'nesw-resize';
          } else {
            canvas.style.cursor = 'default';
          }
          return;
        }
        if (selected === null) return;
        const b = boxes[selected];
        let x1 = b.x, y1 = b.y, x2 = b.x + b.w, y2 = b.y + b.h;
        if (dragMode === 'tl') { x1 = x; y1 = y; }
        if (dragMode === 'tr') { x2 = x; y1 = y; }
        if (dragMode === 'bl') { x1 = x; y2 = y; }
        if (dragMode === 'br') { x2 = x; y2 = y; }
        const nx1 = Math.min(x1, x2);
        const ny1 = Math.min(y1, y2);
        const nx2 = Math.max(x1, x2);
        const ny2 = Math.max(y1, y2);
        boxes[selected] = { x: nx1, y: ny1, w: nx2 - nx1, h: ny2 - ny1 };
        draw();
      });

      canvas.addEventListener('mouseup', () => {
        dragMode = null;
        canvas.style.cursor = 'default';
      });

      canvas.addEventListener('mouseleave', () => {
        dragMode = null;
        canvas.style.cursor = 'default';
      });

      function saveCurrent() {
        const item = items[idx];
        const kept = boxes.filter((_, i) => !removed.has(String(i)));
        return fetch(`/stamps/review/labels?name=${encodeURIComponent(item.name)}`, {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ boxes: kept })
        });
      }

      addBtn.addEventListener('click', () => {
        addMode = true;
        addPoints = [];
        updateControls();
      });
      cancelBtn.addEventListener('click', () => {
        addMode = false;
        addPoints = [];
        draw();
        updateControls();
      });

      validateBtn.addEventListener('click', () => {
        saveCurrent().then(() => {
          fetch(`/stamps/review/validate?name=${encodeURIComponent(currentName)}&user=${encodeURIComponent(userName)}`, { method: 'POST' })
            .then(() => {
              fetchNext();
              refreshStats();
              checkVersionAndReload();
            });
        });
      });

      zoomInBtn.addEventListener('click', () => {
        SCALE = Math.min(2.0, SCALE + 0.1);
        draw();
      });
      zoomOutBtn.addEventListener('click', () => {
        SCALE = Math.max(0.2, SCALE - 0.1);
        draw();
      });

      let userName = localStorage.getItem('review_user');
      if (!userName) {
        userName = prompt('Usuario para revision:') || 'anon';
        localStorage.setItem('review_user', userName);
      }
      userMeta.textContent = `Usuario: ${userName}`;

      changeUserBtn.addEventListener('click', () => {
        const next = prompt('Usuario para revision:', userName);
        if (next) {
          userName = next;
          localStorage.setItem('review_user', userName);
          userMeta.textContent = `Usuario: ${userName}`;
          fetchNext();
          refreshStats();
        }
      });

      function fetchNext() {
        fetch(`/stamps/review/next?user=${encodeURIComponent(userName)}`)
          .then(r => r.json())
          .then(data => { items = [{ name: data.name }]; idx = 0; loadItem(); })
          .then(() => checkVersionAndReload())
          .catch(() => { items = []; meta.textContent = 'Sin pendientes'; draw(); });
      }

      fetch('/stamps/review/total')
        .then(r => r.json())
        .then(data => {
          totalPages = data.total || 0;
          refreshStats();
        });

      fetchNext();
    </script>
  </body>
</html>
"""
    return HTMLResponse(content=html.replace("__REVIEW_VERSION__", review_version))


@app.get("/text/review", response_class=HTMLResponse)
def text_review():
    review_version = os.getenv("TEXT_REVIEW_APP_VERSION", os.getenv("REVIEW_APP_VERSION", "0"))
    html = """
<!doctype html>
<html lang="es">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Text Review</title>
    <style>
      html, body { height: 100%; }
      body { font-family: Arial, sans-serif; margin: 2px 16px; overflow: hidden; }
      #canvas { border: 1px solid #ccc; }
      .layout { display: flex; gap: 16px; align-items: flex-start; height: calc(100vh - 8px); }
      .sidebar { width: 260px; display: flex; flex-direction: column; gap: 8px; overflow-y: auto; max-height: 100%; min-height: 0; }
      .content { flex: 1; overflow: auto; max-height: 100%; min-height: 0; }
      .row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
      .btn { padding: 8px 12px; border: 1px solid #333; background: #f2f2f2; cursor: pointer; }
      .btn:disabled { opacity: 0.5; cursor: default; }
      .btn.active { background: #ffd966; border-color: #b59b00; }
      .meta { font-size: 12px; color: #555; }
      .list { display: flex; flex-direction: column; gap: 6px; max-height: 360px; overflow: auto; border: 1px solid #ddd; padding: 6px; }
      .item { display: flex; align-items: center; gap: 6px; font-size: 12px; }
      .swatch { width: 14px; height: 14px; border: 1px solid #333; }
      .badge { font-size: 11px; color: #333; }
    </style>
  </head>
  <body>
    <div class="layout">
      <div class="sidebar">
        <h3>Revision de texto</h3>
        <div class="row">
          <button class="btn" id="zoomOutBtn">Zoom -</button>
          <button class="btn" id="zoomInBtn">Zoom +</button>
        </div>
        <button class="btn" id="validateBtn">Validar</button>
        <button class="btn" id="skipBtn">Saltar</button>
        <button class="btn" id="addBtn">Agregar bloque</button>
        <button class="btn" id="cancelBtn">Cancelar agregar</button>
        <div class="meta" id="meta"></div>
        <div class="meta" id="userMeta"></div>
        <button class="btn" id="changeUserBtn">Cambiar usuario</button>
        <div class="list" id="boxList"></div>
        <p class="meta">Edita, elimina o agrega cajas de texto.</p>
        <p class="meta">Para agregar: click en “Agregar bloque” y luego 2 clicks en la imagen.</p>
        <p class="meta" id="statsMeta"></p>
        <div class="meta" id="userStats"></div>
      </div>
      <div class="content">
        <canvas id="canvas"></canvas>
      </div>
    </div>
    <script>
      const REVIEW_VERSION = "__REVIEW_VERSION__";
      const canvas = document.getElementById('canvas');
      const ctx = canvas.getContext('2d');
      const meta = document.getElementById('meta');
      const validateBtn = document.getElementById('validateBtn');
      const skipBtn = document.getElementById('skipBtn');
      const addBtn = document.getElementById('addBtn');
      const cancelBtn = document.getElementById('cancelBtn');
      const boxList = document.getElementById('boxList');
      const zoomOutBtn = document.getElementById('zoomOutBtn');
      const zoomInBtn = document.getElementById('zoomInBtn');
      const userMeta = document.getElementById('userMeta');
      const changeUserBtn = document.getElementById('changeUserBtn');
      const statsMeta = document.getElementById('statsMeta');
      const userStats = document.getElementById('userStats');

      let SCALE = 0.5;
      let items = [];
      let idx = 0;
      let currentName = '';
      let totalPages = 0;
      let image = new Image();
      let boxes = [];
      let removed = new Set();
      let addMode = false;
      let addPoints = [];
      let selected = null;
      let dragMode = null;
      const HANDLE = 10;
      const HANDLE_HIT = 20;
      let hoverHandle = null;

      function draw() {
        canvas.width = image.width;
        canvas.height = image.height;
        canvas.style.width = (image.width * SCALE) + 'px';
        canvas.style.height = (image.height * SCALE) + 'px';
        ctx.drawImage(image, 0, 0);
        ctx.lineWidth = 4;
        boxes.forEach((b, i) => {
          const key = String(i);
          ctx.strokeStyle = removed.has(key) ? 'rgba(200,0,0,0.35)' : 'rgba(0,128,0,0.8)';
          ctx.strokeRect(b.x, b.y, b.w, b.h);
          const label = String(i + 1);
          ctx.font = '18px Arial';
          const pad = 6;
          const textW = ctx.measureText(label).width;
          const lx = b.x + 8;
          const ly = b.y + 22;
          ctx.fillStyle = 'rgba(255,255,255,0.75)';
          ctx.fillRect(lx - pad, ly - 16, textW + pad * 2, 20);
          ctx.fillStyle = 'rgba(0,0,0,0.9)';
          ctx.fillText(label, lx, ly);
        });
        ctx.fillStyle = 'rgba(0,0,255,0.9)';
        boxes.forEach((b) => {
          [[b.x,b.y],[b.x+b.w,b.y],[b.x,b.y+b.h],[b.x+b.w,b.y+b.h]].forEach(([px,py]) => {
            ctx.fillRect(px - HANDLE, py - HANDLE, HANDLE * 2, HANDLE * 2);
          });
        });
        if (addPoints.length === 1) {
          ctx.strokeStyle = 'rgba(0,0,200,0.8)';
          ctx.strokeRect(addPoints[0].x - 10, addPoints[0].y - 10, 20, 20);
        }
      }

      function loadItem() {
        if (!items.length) return;
        removed = new Set();
        const item = items[idx];
        currentName = item.name;
        meta.textContent = item.name ? item.name : '';
        image.onload = draw;
        image.src = `/image/${encodeURIComponent(item.name)}`;
        fetch(`/text/review/labels?name=${encodeURIComponent(item.name)}`)
          .then(r => r.json())
          .then(data => { boxes = data.boxes; renderList(); draw(); });
        updateControls();
      }

      function updateControls() {
        validateBtn.disabled = addMode;
        cancelBtn.disabled = !addMode;
        addBtn.classList.toggle('active', addMode);
      }

      function refreshStats() {
        fetch('/text/review/stats')
          .then(r => r.json())
          .then(data => {
            statsMeta.textContent = `Validadas: ${data.validated} / Saltadas: ${data.skipped} / Total: ${totalPages}`;
            userStats.innerHTML = '';
            if (data.users && data.users.length) {
              const title = document.createElement('div');
              title.textContent = 'Usuarios:';
              userStats.appendChild(title);
              data.users.forEach((u) => {
                const row = document.createElement('div');
                row.textContent = `${u.user}: ${u.count}`;
                userStats.appendChild(row);
              });
            }
          });
      }

      function checkVersionAndReload() {
        return fetch('/text/review/version')
          .then(r => r.json())
          .then(data => {
            if (data.version && data.version !== REVIEW_VERSION) {
              location.reload();
              return true;
            }
            return false;
          })
          .catch(() => false);
      }

      function renderList() {
        boxList.innerHTML = '';
        boxes.forEach((_, i) => {
          const row = document.createElement('div');
          row.className = 'item';
          const swatch = document.createElement('div');
          swatch.className = 'swatch';
          const isRemoved = removed.has(String(i));
          swatch.style.background = isRemoved ? 'rgba(200,0,0,0.35)' : 'rgba(0,128,0,0.8)';
          const label = document.createElement('span');
          label.className = 'badge';
          label.textContent = `Caja ${i + 1}`;
          const toggle = document.createElement('input');
          toggle.type = 'checkbox';
          toggle.checked = !isRemoved;
          toggle.addEventListener('change', () => {
            const key = String(i);
            if (toggle.checked) removed.delete(key);
            else removed.add(key);
            renderList();
            draw();
          });
          const del = document.createElement('button');
          del.className = 'btn';
          del.textContent = 'Eliminar';
          del.addEventListener('click', () => {
            removeBox(i);
            renderList();
            draw();
          });
          row.appendChild(swatch);
          row.appendChild(label);
          row.appendChild(toggle);
          row.appendChild(del);
          boxList.appendChild(row);
        });
      }

      function removeBox(index) {
        boxes = boxes.filter((_, i) => i !== index);
        const nextRemoved = new Set();
        boxes.forEach((_, i) => {
          const oldIndex = i >= index ? i + 1 : i;
          if (removed.has(String(oldIndex))) nextRemoved.add(String(i));
        });
        removed = nextRemoved;
      }

      function toCanvasPoint(e) {
        const rect = canvas.getBoundingClientRect();
        return {
          x: (e.clientX - rect.left) * (canvas.width / rect.width),
          y: (e.clientY - rect.top) * (canvas.height / rect.height)
        };
      }

      function hitHandle(b, x, y) {
        const handles = { tl:[b.x,b.y], tr:[b.x+b.w,b.y], bl:[b.x,b.y+b.h], br:[b.x+b.w,b.y+b.h] };
        for (const [key, [hx, hy]] of Object.entries(handles)) {
          if (Math.abs(x - hx) <= HANDLE_HIT && Math.abs(y - hy) <= HANDLE_HIT) return key;
        }
        return null;
      }

      canvas.addEventListener('mousedown', (e) => {
        const { x, y } = toCanvasPoint(e);
        if (addMode) {
          addPoints.push({ x, y });
          if (addPoints.length === 2) {
            const x1 = Math.min(addPoints[0].x, addPoints[1].x);
            const y1 = Math.min(addPoints[0].y, addPoints[1].y);
            const x2 = Math.max(addPoints[0].x, addPoints[1].x);
            const y2 = Math.max(addPoints[0].y, addPoints[1].y);
            boxes.push({ x: x1, y: y1, w: x2 - x1, h: y2 - y1 });
            addPoints = [];
            addMode = false;
            renderList();
            updateControls();
          }
          draw();
          return;
        }

        selected = null;
        dragMode = null;
        for (let i = 0; i < boxes.length; i++) {
          const h = hitHandle(boxes[i], x, y);
          if (h) {
            selected = i;
            dragMode = h;
            break;
          }
          if (x >= boxes[i].x && x <= boxes[i].x + boxes[i].w && y >= boxes[i].y && y <= boxes[i].y + boxes[i].h) {
            selected = i;
          }
        }
        draw();
      });

      canvas.addEventListener('mousemove', (e) => {
        const { x, y } = toCanvasPoint(e);
        if (!dragMode) {
          hoverHandle = null;
          if (selected !== null && boxes[selected]) hoverHandle = hitHandle(boxes[selected], x, y);
          if (hoverHandle === 'tl' || hoverHandle === 'br') canvas.style.cursor = 'nwse-resize';
          else if (hoverHandle === 'tr' || hoverHandle === 'bl') canvas.style.cursor = 'nesw-resize';
          else canvas.style.cursor = 'default';
          return;
        }
        if (selected === null) return;
        const b = boxes[selected];
        let x1 = b.x, y1 = b.y, x2 = b.x + b.w, y2 = b.y + b.h;
        if (dragMode === 'tl') { x1 = x; y1 = y; }
        if (dragMode === 'tr') { x2 = x; y1 = y; }
        if (dragMode === 'bl') { x1 = x; y2 = y; }
        if (dragMode === 'br') { x2 = x; y2 = y; }
        boxes[selected] = { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1) };
        draw();
      });

      canvas.addEventListener('mouseup', () => { dragMode = null; canvas.style.cursor = 'default'; });
      canvas.addEventListener('mouseleave', () => { dragMode = null; canvas.style.cursor = 'default'; });

      function saveCurrent() {
        const kept = boxes.filter((_, i) => !removed.has(String(i)));
        return fetch(`/text/review/labels?name=${encodeURIComponent(currentName)}`, {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ boxes: kept })
        });
      }

      addBtn.addEventListener('click', () => { addMode = true; addPoints = []; updateControls(); });
      cancelBtn.addEventListener('click', () => { addMode = false; addPoints = []; draw(); updateControls(); });
      validateBtn.addEventListener('click', () => {
        saveCurrent().then(() => {
          fetch(`/text/review/validate?name=${encodeURIComponent(currentName)}&user=${encodeURIComponent(userName)}`, { method: 'POST' })
            .then(() => { fetchNext(); refreshStats(); checkVersionAndReload(); });
        });
      });
      skipBtn.addEventListener('click', () => {
        if (!currentName) return;
        fetch(`/text/review/skip?name=${encodeURIComponent(currentName)}&user=${encodeURIComponent(userName)}`, { method: 'POST' })
          .then(() => { fetchNext(); refreshStats(); checkVersionAndReload(); });
      });
      zoomInBtn.addEventListener('click', () => { SCALE = Math.min(2.0, SCALE + 0.1); draw(); });
      zoomOutBtn.addEventListener('click', () => { SCALE = Math.max(0.2, SCALE - 0.1); draw(); });

      let userName = localStorage.getItem('text_review_user');
      if (!userName) {
        userName = prompt('Usuario para revision de texto:') || 'anon';
        localStorage.setItem('text_review_user', userName);
      }
      userMeta.textContent = `Usuario: ${userName}`;

      changeUserBtn.addEventListener('click', () => {
        const next = prompt('Usuario para revision de texto:', userName);
        if (next) {
          userName = next;
          localStorage.setItem('text_review_user', userName);
          userMeta.textContent = `Usuario: ${userName}`;
          fetchNext();
          refreshStats();
        }
      });

      function fetchNext() {
        fetch(`/text/review/next?user=${encodeURIComponent(userName)}`)
          .then(r => r.json())
          .then(data => { items = [{ name: data.name }]; idx = 0; loadItem(); })
          .then(() => checkVersionAndReload())
          .catch(() => { items = []; meta.textContent = 'Sin pendientes'; draw(); });
      }

      fetch('/text/review/total')
        .then(r => r.json())
        .then(data => { totalPages = data.total || 0; refreshStats(); });

      fetchNext();
    </script>
  </body>
</html>
"""
    return HTMLResponse(content=html.replace("__REVIEW_VERSION__", review_version))


@app.get("/text/review/skipped", response_class=HTMLResponse)
def text_review_skipped():
    review_version = os.getenv("TEXT_REVIEW_APP_VERSION", os.getenv("REVIEW_APP_VERSION", "0"))
    html = """
<!doctype html>
<html lang="es">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Text Review Skipped</title>
    <style>
      html, body { height: 100%; }
      body { font-family: Arial, sans-serif; margin: 2px 16px; overflow: hidden; }
      #canvas { border: 1px solid #ccc; }
      .layout { display: flex; gap: 16px; align-items: flex-start; height: calc(100vh - 8px); }
      .sidebar { width: 260px; display: flex; flex-direction: column; gap: 8px; overflow-y: auto; max-height: 100%; min-height: 0; }
      .content { flex: 1; overflow: auto; max-height: 100%; min-height: 0; }
      .row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
      .btn { padding: 8px 12px; border: 1px solid #333; background: #f2f2f2; cursor: pointer; }
      .btn:disabled { opacity: 0.5; cursor: default; }
      .btn.active { background: #ffd966; border-color: #b59b00; }
      .meta { font-size: 12px; color: #555; }
      .list { display: flex; flex-direction: column; gap: 6px; max-height: 360px; overflow: auto; border: 1px solid #ddd; padding: 6px; }
      .item { display: flex; align-items: center; gap: 6px; font-size: 12px; }
      .swatch { width: 14px; height: 14px; border: 1px solid #333; }
      .badge { font-size: 11px; color: #333; }
    </style>
  </head>
  <body>
    <div class="layout">
      <div class="sidebar">
        <h3>Texto saltado</h3>
        <div class="row">
          <button class="btn" id="zoomOutBtn">Zoom -</button>
          <button class="btn" id="zoomInBtn">Zoom +</button>
        </div>
        <button class="btn" id="validateBtn">Validar</button>
        <button class="btn" id="requeueBtn">Reencolar</button>
        <button class="btn" id="addBtn">Agregar bloque</button>
        <button class="btn" id="cancelBtn">Cancelar agregar</button>
        <div class="meta" id="meta"></div>
        <div class="meta" id="userMeta"></div>
        <button class="btn" id="changeUserBtn">Cambiar usuario</button>
        <div class="list" id="boxList"></div>
        <p class="meta">Revision final de paginas saltadas.</p>
        <p class="meta" id="statsMeta"></p>
        <div class="meta" id="userStats"></div>
      </div>
      <div class="content">
        <canvas id="canvas"></canvas>
      </div>
    </div>
    <script>
      const REVIEW_VERSION = "__REVIEW_VERSION__";
      const canvas = document.getElementById('canvas');
      const ctx = canvas.getContext('2d');
      const meta = document.getElementById('meta');
      const validateBtn = document.getElementById('validateBtn');
      const requeueBtn = document.getElementById('requeueBtn');
      const addBtn = document.getElementById('addBtn');
      const cancelBtn = document.getElementById('cancelBtn');
      const boxList = document.getElementById('boxList');
      const zoomOutBtn = document.getElementById('zoomOutBtn');
      const zoomInBtn = document.getElementById('zoomInBtn');
      const userMeta = document.getElementById('userMeta');
      const changeUserBtn = document.getElementById('changeUserBtn');
      const statsMeta = document.getElementById('statsMeta');
      const userStats = document.getElementById('userStats');

      let SCALE = 0.5, items = [], idx = 0, currentName = '', totalPages = 0;
      let image = new Image(), boxes = [], removed = new Set(), addMode = false, addPoints = [];
      let selected = null, dragMode = null;
      const HANDLE = 10, HANDLE_HIT = 20;

      function draw() {
        canvas.width = image.width; canvas.height = image.height;
        canvas.style.width = (image.width * SCALE) + 'px';
        canvas.style.height = (image.height * SCALE) + 'px';
        ctx.drawImage(image, 0, 0);
        ctx.lineWidth = 4;
        boxes.forEach((b, i) => {
          ctx.strokeStyle = removed.has(String(i)) ? 'rgba(200,0,0,0.35)' : 'rgba(0,128,0,0.8)';
          ctx.strokeRect(b.x, b.y, b.w, b.h);
          ctx.font = '18px Arial';
          const label = String(i + 1);
          const textW = ctx.measureText(label).width;
          ctx.fillStyle = 'rgba(255,255,255,0.75)';
          ctx.fillRect(b.x + 2, b.y + 2, textW + 12, 22);
          ctx.fillStyle = 'rgba(0,0,0,0.9)';
          ctx.fillText(label, b.x + 8, b.y + 19);
          [[b.x,b.y],[b.x+b.w,b.y],[b.x,b.y+b.h],[b.x+b.w,b.y+b.h]].forEach(([px,py]) => {
            ctx.fillStyle = 'rgba(0,0,255,0.9)';
            ctx.fillRect(px - HANDLE, py - HANDLE, HANDLE * 2, HANDLE * 2);
          });
        });
      }

      function renderList() {
        boxList.innerHTML = '';
        boxes.forEach((_, i) => {
          const row = document.createElement('div');
          row.className = 'item';
          const swatch = document.createElement('div');
          swatch.className = 'swatch';
          swatch.style.background = removed.has(String(i)) ? 'rgba(200,0,0,0.35)' : 'rgba(0,128,0,0.8)';
          const label = document.createElement('span');
          label.className = 'badge';
          label.textContent = `Caja ${i + 1}`;
          const toggle = document.createElement('input');
          toggle.type = 'checkbox';
          toggle.checked = !removed.has(String(i));
          toggle.addEventListener('change', () => {
            const key = String(i);
            if (toggle.checked) removed.delete(key); else removed.add(key);
            renderList(); draw();
          });
          const del = document.createElement('button');
          del.className = 'btn';
          del.textContent = 'Eliminar';
          del.addEventListener('click', () => {
            boxes = boxes.filter((_, idx) => idx !== i);
            removed = new Set();
            renderList(); draw();
          });
          row.appendChild(swatch); row.appendChild(label); row.appendChild(toggle); row.appendChild(del);
          boxList.appendChild(row);
        });
      }

      function toCanvasPoint(e) {
        const rect = canvas.getBoundingClientRect();
        return { x: (e.clientX - rect.left) * (canvas.width / rect.width), y: (e.clientY - rect.top) * (canvas.height / rect.height) };
      }

      function hitHandle(b, x, y) {
        const handles = { tl:[b.x,b.y], tr:[b.x+b.w,b.y], bl:[b.x,b.y+b.h], br:[b.x+b.w,b.y+b.h] };
        for (const [key, [hx, hy]] of Object.entries(handles)) {
          if (Math.abs(x - hx) <= HANDLE_HIT && Math.abs(y - hy) <= HANDLE_HIT) return key;
        }
        return null;
      }

      function updateControls() {
        validateBtn.disabled = addMode;
        requeueBtn.disabled = addMode;
        cancelBtn.disabled = !addMode;
        addBtn.classList.toggle('active', addMode);
      }

      function saveCurrent() {
        const kept = boxes.filter((_, i) => !removed.has(String(i)));
        return fetch(`/text/review/labels?name=${encodeURIComponent(currentName)}`, {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ boxes: kept })
        });
      }

      function loadItem() {
        if (!items.length) return;
        removed = new Set();
        currentName = items[0].name;
        meta.textContent = currentName || '';
        image.onload = draw;
        image.src = `/image/${encodeURIComponent(currentName)}`;
        fetch(`/text/review/labels?name=${encodeURIComponent(currentName)}`)
          .then(r => r.json())
          .then(data => { boxes = data.boxes; renderList(); draw(); });
        updateControls();
      }

      function refreshStats() {
        fetch('/text/review/stats')
          .then(r => r.json())
          .then(data => {
            statsMeta.textContent = `Validadas: ${data.validated} / Saltadas: ${data.skipped} / Total: ${totalPages}`;
            userStats.innerHTML = '';
            (data.users || []).forEach((u) => {
              const row = document.createElement('div');
              row.textContent = `${u.user}: ${u.count}`;
              userStats.appendChild(row);
            });
          });
      }

      function checkVersionAndReload() {
        return fetch('/text/review/version')
          .then(r => r.json())
          .then(data => {
            if (data.version && data.version !== REVIEW_VERSION) { location.reload(); return true; }
            return false;
          })
          .catch(() => false);
      }

      canvas.addEventListener('mousedown', (e) => {
        const { x, y } = toCanvasPoint(e);
        if (addMode) {
          addPoints.push({ x, y });
          if (addPoints.length === 2) {
            const x1 = Math.min(addPoints[0].x, addPoints[1].x);
            const y1 = Math.min(addPoints[0].y, addPoints[1].y);
            const x2 = Math.max(addPoints[0].x, addPoints[1].x);
            const y2 = Math.max(addPoints[0].y, addPoints[1].y);
            boxes.push({ x: x1, y: y1, w: x2 - x1, h: y2 - y1 });
            addPoints = [];
            addMode = false;
            renderList(); updateControls(); draw();
          }
          return;
        }
        selected = null; dragMode = null;
        for (let i = 0; i < boxes.length; i++) {
          const h = hitHandle(boxes[i], x, y);
          if (h) { selected = i; dragMode = h; break; }
          if (x >= boxes[i].x && x <= boxes[i].x + boxes[i].w && y >= boxes[i].y && y <= boxes[i].y + boxes[i].h) selected = i;
        }
      });

      canvas.addEventListener('mousemove', (e) => {
        if (!dragMode || selected === null) return;
        const { x, y } = toCanvasPoint(e);
        const b = boxes[selected];
        let x1 = b.x, y1 = b.y, x2 = b.x + b.w, y2 = b.y + b.h;
        if (dragMode === 'tl') { x1 = x; y1 = y; }
        if (dragMode === 'tr') { x2 = x; y1 = y; }
        if (dragMode === 'bl') { x1 = x; y2 = y; }
        if (dragMode === 'br') { x2 = x; y2 = y; }
        boxes[selected] = { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1) };
        draw();
      });

      canvas.addEventListener('mouseup', () => { dragMode = null; });
      canvas.addEventListener('mouseleave', () => { dragMode = null; });

      let userName = localStorage.getItem('text_review_skipped_user');
      if (!userName) {
        userName = prompt('Usuario para revisar saltados:') || 'anon';
        localStorage.setItem('text_review_skipped_user', userName);
      }
      userMeta.textContent = `Usuario: ${userName}`;

      changeUserBtn.addEventListener('click', () => {
        const next = prompt('Usuario para revisar saltados:', userName);
        if (next) {
          userName = next;
          localStorage.setItem('text_review_skipped_user', userName);
          userMeta.textContent = `Usuario: ${userName}`;
          fetchNext(); refreshStats();
        }
      });

      addBtn.addEventListener('click', () => { addMode = true; addPoints = []; updateControls(); });
      cancelBtn.addEventListener('click', () => { addMode = false; addPoints = []; updateControls(); draw(); });
      validateBtn.addEventListener('click', () => {
        saveCurrent().then(() => fetch(`/text/review/validate?name=${encodeURIComponent(currentName)}&user=${encodeURIComponent(userName)}`, { method: 'POST' }))
          .then(() => { fetchNext(); refreshStats(); checkVersionAndReload(); });
      });
      requeueBtn.addEventListener('click', () => {
        fetch(`/text/review/requeue?name=${encodeURIComponent(currentName)}&user=${encodeURIComponent(userName)}`, { method: 'POST' })
          .then(() => { fetchNext(); refreshStats(); checkVersionAndReload(); });
      });
      zoomInBtn.addEventListener('click', () => { SCALE = Math.min(2.0, SCALE + 0.1); draw(); });
      zoomOutBtn.addEventListener('click', () => { SCALE = Math.max(0.2, SCALE - 0.1); draw(); });

      function fetchNext() {
        fetch(`/text/review/skipped/next?user=${encodeURIComponent(userName)}`)
          .then(r => r.json())
          .then(data => { items = [{ name: data.name }]; loadItem(); })
          .catch(() => { items = []; meta.textContent = 'Sin saltados'; draw(); });
      }

      fetch('/text/review/total').then(r => r.json()).then(data => { totalPages = data.total || 0; refreshStats(); });
      fetchNext();
    </script>
  </body>
</html>
"""
    return HTMLResponse(content=html.replace("__REVIEW_VERSION__", review_version))


@app.get("/stamps/review/items")
def stamps_review_items():
    images_dir = Path(DEFAULT_OUT_DIR) / "stamp_pages" / "images"
    if not images_dir.exists():
        raise HTTPException(status_code=404, detail="stamp_pages/images not found")
    items = [
        {"name": p.name}
        for p in sorted(images_dir.iterdir())
        if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    ]
    return {"items": items}


@app.get("/text/review/items")
def text_review_items():
    images_dir = _text_review_dir() / "images"
    if not images_dir.exists():
        raise HTTPException(status_code=404, detail="text_pages/images not found")
    items = [
        {"name": p.name}
        for p in sorted(images_dir.iterdir())
        if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    ]
    return {"items": items}


@app.get("/stamps/review/next")
def stamps_review_next(user: str):
    if not user:
        raise HTTPException(status_code=400, detail="user required")
    state = _normalize_state(_load_review_state())
    items_state = state.get("items", {})
    images = _list_review_images()
    for name in images:
        info = items_state.get(name, {"status": "pending"})
        if info.get("status") == "pending":
            items_state[name] = {
                "status": "in_process",
                "user": user,
                "locked_at": time.time(),
            }
            state["items"] = items_state
            _save_review_state(state)
            return {"name": name}
    raise HTTPException(status_code=404, detail="no pending items")


@app.get("/text/review/next")
def text_review_next(user: str):
    if not user:
        raise HTTPException(status_code=400, detail="user required")
    state = _normalize_text_review_state(_load_text_review_state())
    items_state = state.get("items", {})
    images = _list_text_review_images()
    for name in images:
        info = items_state.get(name, {"status": "pending"})
        if info.get("status") == "in_process" and info.get("user") == user:
            return {"name": name}
    for name in images:
        info = items_state.get(name, {"status": "pending"})
        if info.get("status") == "pending":
            items_state[name] = {
                "status": "in_process",
                "user": user,
                "locked_at": time.time(),
            }
            state["items"] = items_state
            _save_text_review_state(state)
            return {"name": name}
    raise HTTPException(status_code=404, detail="no pending items")


@app.get("/text/review/skipped/next")
def text_review_skipped_next(user: str):
    if not user:
        raise HTTPException(status_code=400, detail="user required")
    state = _normalize_text_review_state(_load_text_review_state())
    items_state = state.get("items", {})
    images = _list_text_review_images()
    for name in images:
        info = items_state.get(name, {"status": "pending"})
        if info.get("status") == "in_process" and info.get("user") == user and info.get("from_skipped"):
            return {"name": name}
    for name in images:
        info = items_state.get(name, {"status": "pending"})
        if info.get("status") == "skipped":
            items_state[name] = {
                "status": "in_process",
                "user": user,
                "locked_at": time.time(),
                "from_skipped": True,
            }
            state["items"] = items_state
            _save_text_review_state(state)
            return {"name": name}
    raise HTTPException(status_code=404, detail="no skipped items")


@app.get("/stamps/review/total")
def stamps_review_total():
    return {"total": _review_total_pages()}


@app.get("/text/review/total")
def text_review_total():
    return {"total": _text_review_total_pages()}


@app.get("/stamps/review/version")
def stamps_review_version():
    return {"version": os.getenv("REVIEW_APP_VERSION", "0")}


@app.get("/text/review/version")
def text_review_version():
    return {"version": os.getenv("TEXT_REVIEW_APP_VERSION", os.getenv("REVIEW_APP_VERSION", "0"))}


@app.get("/stamps/review/stats")
def stamps_review_stats():
    state = _normalize_state(_load_review_state())
    items = state.get("items", {})
    validated = 0
    per_user: dict[str, int] = {}
    for meta in items.values():
        if meta.get("validated_at"):
            validated += 1
            user = meta.get("user") or "anon"
            per_user[user] = per_user.get(user, 0) + 1
    users = [
        {"user": u, "count": c}
        for u, c in sorted(per_user.items(), key=lambda item: item[1], reverse=True)
    ]
    return {"validated": validated, "users": users}


@app.get("/text/review/stats")
def text_review_stats():
    state = _normalize_text_review_state(_load_text_review_state())
    items = state.get("items", {})
    validated = 0
    skipped = 0
    per_user: dict[str, int] = {}
    for meta in items.values():
        status = meta.get("status")
        if status == "validated":
            validated += 1
            user = meta.get("user") or "anon"
            per_user[user] = per_user.get(user, 0) + 1
        elif status == "skipped":
            skipped += 1
    users = [
        {"user": u, "count": c}
        for u, c in sorted(per_user.items(), key=lambda item: item[1], reverse=True)
    ]
    return {"validated": validated, "skipped": skipped, "users": users}


@app.post("/stamps/review/validate")
def stamps_review_validate(name: str, user: str):
    state = _normalize_state(_load_review_state())
    items_state = state.get("items", {})
    info = items_state.get(name, {"status": "pending"})
    if info.get("status") == "in_process" and info.get("user") != user:
        raise HTTPException(status_code=409, detail="locked by another user")
    items_state[name] = {
        "status": "validated",
        "user": user,
        "locked_at": 0,
        "validated_at": time.time(),
    }
    state["items"] = items_state
    _save_review_state(state)
    return {"ok": True}


@app.post("/text/review/validate")
def text_review_validate(name: str, user: str):
    state = _normalize_text_review_state(_load_text_review_state())
    items_state = state.get("items", {})
    info = items_state.get(name, {"status": "pending"})
    if info.get("status") == "in_process" and info.get("user") != user:
        raise HTTPException(status_code=409, detail="locked by another user")
    items_state[name] = {
        "status": "validated",
        "user": user,
        "locked_at": 0,
        "validated_at": time.time(),
    }
    state["items"] = items_state
    _save_text_review_state(state)
    return {"ok": True}


@app.post("/stamps/review/release")
def stamps_review_release(name: str, user: str):
    state = _normalize_state(_load_review_state())
    items_state = state.get("items", {})
    info = items_state.get(name, {"status": "pending"})
    if info.get("status") == "in_process" and info.get("user") == user:
        items_state[name] = {
            "status": "pending",
            "user": "",
            "locked_at": 0,
        }
        state["items"] = items_state
        _save_review_state(state)
    return {"ok": True}


@app.post("/text/review/release")
def text_review_release(name: str, user: str):
    state = _normalize_text_review_state(_load_text_review_state())
    items_state = state.get("items", {})
    info = items_state.get(name, {"status": "pending"})
    if info.get("status") == "in_process" and info.get("user") == user:
        items_state[name] = {
            "status": "pending",
            "user": "",
            "locked_at": 0,
        }
        state["items"] = items_state
        _save_text_review_state(state)
    return {"ok": True}


@app.post("/text/review/skip")
def text_review_skip(name: str, user: str):
    if not name or not user:
        raise HTTPException(status_code=400, detail="name and user required")
    state = _normalize_text_review_state(_load_text_review_state())
    items_state = state.get("items", {})
    items_state[name] = {
        "status": "skipped",
        "user": user,
        "locked_at": 0,
        "validated_at": time.time(),
    }
    state["items"] = items_state
    _save_text_review_state(state)
    return {"ok": True}


@app.post("/text/review/requeue")
def text_review_requeue(name: str, user: str):
    if not name or not user:
        raise HTTPException(status_code=400, detail="name and user required")
    state = _normalize_text_review_state(_load_text_review_state())
    items_state = state.get("items", {})
    info = items_state.get(name, {"status": "pending"})
    if info.get("status") == "in_process" and info.get("user") != user:
        raise HTTPException(status_code=409, detail="locked by another user")
    items_state[name] = {
        "status": "pending",
        "user": "",
        "locked_at": 0,
    }
    state["items"] = items_state
    _save_text_review_state(state)
    return {"ok": True}


@app.get("/stamps/review/labels")
def stamps_review_labels(name: str):
    labels_dir = Path(DEFAULT_OUT_DIR) / "stamp_pages" / "labels"
    label_path = labels_dir / f"{Path(name).stem}.txt"
    if not label_path.exists():
        raise HTTPException(status_code=404, detail="label not found")
    lines = [l.strip() for l in label_path.read_text().splitlines() if l.strip()]
    boxes = []
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue
        _, cx, cy, w, h = parts
        boxes.append({"cx": float(cx), "cy": float(cy), "w": float(w), "h": float(h)})

    # Convert to absolute using image size
    img_path = Path(DEFAULT_OUT_DIR) / "stamp_pages" / "images" / name
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="image not found")
    import cv2  # local import to keep startup light
    img = cv2.imread(str(img_path))
    if img is None:
        raise HTTPException(status_code=400, detail="cannot read image")
    h_img, w_img = img.shape[:2]
    abs_boxes = []
    for b in boxes:
        bw = b["w"] * w_img
        bh = b["h"] * h_img
        x = b["cx"] * w_img - bw / 2
        y = b["cy"] * h_img - bh / 2
        abs_boxes.append({"x": x, "y": y, "w": bw, "h": bh})
    return {"boxes": abs_boxes}


@app.get("/text/review/labels")
def text_review_labels(name: str):
    reviewed_path = _text_review_labels_reviewed_dir() / f"{Path(name).stem}.txt"
    auto_path = _text_review_labels_auto_dir() / f"{Path(name).stem}.txt"
    label_path = reviewed_path if reviewed_path.exists() else auto_path
    if not label_path.exists():
        return {"boxes": []}

    lines = [l.strip() for l in label_path.read_text().splitlines() if l.strip()]
    boxes = []
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue
        _, cx, cy, w, h = parts
        boxes.append({"cx": float(cx), "cy": float(cy), "w": float(w), "h": float(h)})

    img_path = _text_review_dir() / "images" / name
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="image not found")
    import cv2
    img = cv2.imread(str(img_path))
    if img is None:
        raise HTTPException(status_code=400, detail="cannot read image")
    h_img, w_img = img.shape[:2]
    abs_boxes = []
    for b in boxes:
        bw = b["w"] * w_img
        bh = b["h"] * h_img
        x = b["cx"] * w_img - bw / 2
        y = b["cy"] * h_img - bh / 2
        abs_boxes.append({"x": x, "y": y, "w": bw, "h": bh})
    return {"boxes": abs_boxes}


@app.post("/stamps/review/labels")
def stamps_review_labels_save(name: str, payload: dict):
    labels_dir = Path(DEFAULT_OUT_DIR) / "stamp_pages" / "labels"
    label_path = labels_dir / f"{Path(name).stem}.txt"
    if not label_path.exists():
        raise HTTPException(status_code=404, detail="label not found")
    img_path = Path(DEFAULT_OUT_DIR) / "stamp_pages" / "images" / name
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="image not found")

    import cv2
    img = cv2.imread(str(img_path))
    if img is None:
        raise HTTPException(status_code=400, detail="cannot read image")
    h_img, w_img = img.shape[:2]

    boxes = payload.get("boxes") or []
    lines = []
    for b in boxes:
        x, y, w, h = b["x"], b["y"], b["w"], b["h"]
        cx = (x + w / 2) / w_img
        cy = (y + h / 2) / h_img
        nw = w / w_img
        nh = h / h_img
        lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""))
    return {"ok": True, "count": len(lines)}


@app.post("/text/review/labels")
def text_review_labels_save(name: str, payload: dict):
    img_path = _text_review_dir() / "images" / name
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="image not found")

    import cv2
    img = cv2.imread(str(img_path))
    if img is None:
        raise HTTPException(status_code=400, detail="cannot read image")
    h_img, w_img = img.shape[:2]

    labels_dir = _text_review_labels_reviewed_dir()
    labels_dir.mkdir(parents=True, exist_ok=True)
    label_path = labels_dir / f"{Path(name).stem}.txt"

    boxes = payload.get("boxes") or []
    lines = []
    for b in boxes:
        x, y, w, h = b["x"], b["y"], b["w"], b["h"]
        cx = (x + w / 2) / w_img
        cy = (y + h / 2) / h_img
        nw = w / w_img
        nh = h / h_img
        lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""))
    return {"ok": True, "count": len(lines)}


@app.get("/stamps/classify/image/{name}")
def stamps_classify_image(name: str):
    file_path = _classify_dir() / "crops" / name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    suffix = file_path.suffix.lower()
    if suffix == ".png":
        media_type = "image/png"
    elif suffix in (".jpg", ".jpeg"):
        media_type = "image/jpeg"
    else:
        media_type = "application/octet-stream"
    return FileResponse(path=str(file_path), media_type=media_type, filename=name)


@app.get("/stamps/classify", response_class=HTMLResponse)
def stamps_classify():
    html = """
<!doctype html>
<html lang="es">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Stamp Classify</title>
    <style>
      html, body { height: 100%; }
      body { font-family: Arial, sans-serif; margin: 2px 16px; overflow: hidden; }
      .layout { display: flex; gap: 16px; align-items: flex-start; height: calc(100vh - 8px); }
      .sidebar { width: 260px; display: flex; flex-direction: column; gap: 8px; overflow-y: auto; max-height: 100%; min-height: 0; }
      .content { flex: 1; overflow: auto; max-height: 100%; min-height: 0; display: flex; align-items: flex-start; justify-content: center; }
      .content-wrap { display: flex; flex-direction: column; gap: 8px; align-items: center; }
      .image-frame { width: 100%; max-width: 700px; height: 320px; display: flex; align-items: center; justify-content: center; border: 1px solid #ddd; background: #fff; }
      .image-frame img { max-width: 100%; max-height: 100%; }
      .btn { padding: 8px 12px; border: 1px solid #333; background: #f2f2f2; cursor: pointer; }
      .btn:disabled { opacity: 0.5; cursor: default; }
      .btn.active { background: #ffd966; border-color: #b59b00; }
      .btn.suggested { background: beige; border-color: #c89f00; font-weight: 700; }
      .meta { font-size: 12px; color: #555; }
      .class-list { display: grid; grid-template-columns: 1fr; gap: 6px; }
      .class-btn { text-align: left; }
      img { max-width: 100%; height: auto; border: 1px solid #ccc; }
    </style>
  </head>
  <body>
    <div class="layout">
      <div class="sidebar">
        <h3>Clasificar recortes</h3>
        <button class="btn" id="rejectBtn">Descartar</button>
        <button class="btn" id="skipBtn">Saltar</button>
        <div class="meta" id="meta"></div>
        <div class="meta" id="progress"></div>
        <div class="meta" id="suggestion"></div>
        <div class="meta" id="userMeta"></div>
        <button class="btn" id="changeUserBtn">Cambiar usuario</button>
        <div class="meta" id="userStats"></div>
      </div>
      <div class="content">
        <div class="content-wrap">
          <div class="image-frame">
            <img id="crop" alt="recorte" />
          </div>
          <div class="class-list" id="classList"></div>
        </div>
      </div>
    </div>
    <script>
      const CLASSES = [
        "sello_redondo",
        "logo",
        "firma",
        "firma_con_huella",
        "sello_completo",
        "sello_cuadrado",
        "huella_digital",
        "sello_proveido",
        "sello_recepcion",
        "sello_fedatario",
      ];
      const rejectBtn = document.getElementById('rejectBtn');
      const skipBtn = document.getElementById('skipBtn');
      const classList = document.getElementById('classList');
      const crop = document.getElementById('crop');
      const meta = document.getElementById('meta');
      const progress = document.getElementById('progress');
      const suggestion = document.getElementById('suggestion');
      const userMeta = document.getElementById('userMeta');
      const changeUserBtn = document.getElementById('changeUserBtn');
      const userStats = document.getElementById('userStats');

      let currentName = '';
      let selectedClass = '';
      let userName = localStorage.getItem('classify_user');

      if (!userName) {
        userName = prompt('Usuario para clasificar:') || 'anon';
        localStorage.setItem('classify_user', userName);
      }
      userMeta.textContent = `Usuario: ${userName}`;

      let classCounts = {};

      function renderClasses(counts) {
        classCounts = counts || {};
        classList.innerHTML = '';
        CLASSES.forEach((cls) => {
          const count = classCounts[cls] || 0;
          const btn = document.createElement('button');
          btn.className = 'btn class-btn';
          btn.dataset.cls = cls;
          btn.textContent = `${cls} (${count})`;
          btn.addEventListener('click', () => {
            selectedClass = cls;
            document.querySelectorAll('.class-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            if (!currentName) return;
            fetch(`/stamps/classify/label?name=${encodeURIComponent(currentName)}&user=${encodeURIComponent(userName)}`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ label: selectedClass }),
            }).then(() => {
              fetchNext();
              refreshProgress();
            });
          });
          classList.appendChild(btn);
        });
      }

      function loadSuggestion(name) {
        fetch(`/stamps/classify/suggestion?name=${encodeURIComponent(name)}`)
          .then(r => r.json())
          .then(data => {
            if (data.label) {
              suggestion.textContent = `Sugerido: ${data.label} (${(data.confidence * 100).toFixed(1)}%)`;
              document.querySelectorAll('.class-btn').forEach(b => b.classList.remove('suggested'));
              const match = document.querySelector(`.class-btn[data-cls="${data.label}"]`);
              if (match) {
                match.classList.add('suggested');
                const cls = match.dataset.cls;
                const count = classCounts[cls] || 0;
                match.textContent = `*${cls} (${count})`;
              }
            } else {
              suggestion.textContent = '';
              document.querySelectorAll('.class-btn').forEach(b => {
                b.classList.remove('suggested');
                const cls = b.dataset.cls;
                const count = classCounts[cls] || 0;
                b.textContent = `${cls} (${count})`;
              });
            }
          })
          .catch(() => {
            suggestion.textContent = '';
            document.querySelectorAll('.class-btn').forEach(b => {
              b.classList.remove('suggested');
              const cls = b.dataset.cls;
              const count = classCounts[cls] || 0;
              b.textContent = `${cls} (${count})`;
            });
          });
      }

      function fetchNext() {
        fetch(`/stamps/classify/next?user=${encodeURIComponent(userName)}`)
          .then(r => r.json())
          .then(data => {
            currentName = data.name || '';
            meta.textContent = currentName ? currentName : 'Sin pendientes';
            crop.src = currentName ? `/stamps/classify/image/${encodeURIComponent(currentName)}` : '';
            selectedClass = '';
            document.querySelectorAll('.class-btn').forEach(b => b.classList.remove('active'));
            if (currentName) loadSuggestion(currentName);
          })
          .catch(() => { meta.textContent = 'Sin pendientes'; crop.src = ''; });
      }

      rejectBtn.addEventListener('click', () => {
        if (!currentName) return;
        fetch(`/stamps/classify/reject?name=${encodeURIComponent(currentName)}&user=${encodeURIComponent(userName)}`, { method: 'POST' })
          .then(() => {
            fetchNext();
            refreshProgress();
          });
      });

      skipBtn.addEventListener('click', () => {
        if (!currentName) return;
        fetch(`/stamps/classify/skip?name=${encodeURIComponent(currentName)}&user=${encodeURIComponent(userName)}`, { method: 'POST' })
          .then(() => {
            fetchNext();
            refreshProgress();
          });
      });

      changeUserBtn.addEventListener('click', () => {
        const next = prompt('Usuario para clasificar:', userName);
        if (next) {
          userName = next;
          localStorage.setItem('classify_user', userName);
          userMeta.textContent = `Usuario: ${userName}`;
          fetchNext();
          refreshProgress();
        }
      });

      function refreshProgress() {
        fetch('/stamps/classify/stats')
          .then(r => r.json())
          .then(data => {
            const done = data.validated + data.rejected;
            progress.textContent = `Avance: ${done} / ${data.total} (rechazados: ${data.rejected}, saltados: ${data.skipped})`;
            renderClasses(data.per_class || {});
            userStats.innerHTML = '';
            if (data.per_user) {
              const title = document.createElement('div');
              title.textContent = 'Usuarios:';
              userStats.appendChild(title);
              Object.entries(data.per_user).forEach(([user, count]) => {
                const row = document.createElement('div');
                row.textContent = `${user}: ${count}`;
                userStats.appendChild(row);
              });
            }
          });
      }

      refreshProgress();
      fetchNext();
    </script>
  </body>
</html>
"""
    return HTMLResponse(content=html)


@app.get("/stamps/classify/next")
def stamps_classify_next(user: str):
    if not user:
        raise HTTPException(status_code=400, detail="user required")
    state = _normalize_classify_state(_load_classify_state())
    items_state = state.get("items", {})
    crops = _list_classify_crops()
    preds = _load_classify_preds()
    threshold = _classify_conf_threshold()
    for name in crops:
        info = items_state.get(name, {"status": "pending"})
        if info.get("status") == "pending":
            pred = preds.get(name)
            if pred and float(pred.get("confidence", 0.0) or 0.0) >= threshold:
                continue
            items_state[name] = {
                "status": "in_process",
                "user": user,
                "locked_at": time.time(),
                "label": info.get("label", ""),
            }
            state["items"] = items_state
            _save_classify_state(state)
            return {"name": name}
    raise HTTPException(status_code=404, detail="no pending items")


@app.post("/stamps/classify/label")
def stamps_classify_label(name: str, user: str, payload: dict):
    label = payload.get("label")
    if not name or not user or not label:
        raise HTTPException(status_code=400, detail="name, user and label required")
    state = _normalize_classify_state(_load_classify_state())
    items = state.get("items", {})
    items[name] = {
        "status": "validated",
        "user": user,
        "locked_at": 0,
        "validated_at": time.time(),
        "label": label,
    }
    state["items"] = items
    _save_classify_state(state)
    return {"ok": True}


@app.post("/stamps/classify/reject")
def stamps_classify_reject(name: str, user: str):
    if not name or not user:
        raise HTTPException(status_code=400, detail="name and user required")
    state = _normalize_classify_state(_load_classify_state())
    items = state.get("items", {})
    items[name] = {
        "status": "rejected",
        "user": user,
        "locked_at": 0,
        "validated_at": time.time(),
        "label": "__rejected__",
    }
    state["items"] = items
    _save_classify_state(state)

    src = _classify_dir() / "crops" / name
    if src.exists():
        dst_dir = _classify_rejected_dir()
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / name
        src.replace(dst)
    return {"ok": True}


@app.post("/stamps/classify/skip")
def stamps_classify_skip(name: str, user: str):
    if not name or not user:
        raise HTTPException(status_code=400, detail="name and user required")
    state = _normalize_classify_state(_load_classify_state())
    items = state.get("items", {})
    items[name] = {
        "status": "skipped",
        "user": user,
        "locked_at": 0,
        "validated_at": time.time(),
        "label": "__skipped__",
    }
    state["items"] = items
    _save_classify_state(state)
    return {"ok": True}


@app.get("/stamps/classify/suggestion")
def stamps_classify_suggestion(name: str):
    preds = _load_classify_preds()
    info = preds.get(name) or {}
    return {
        "label": info.get("label", ""),
        "confidence": float(info.get("confidence", 0.0) or 0.0),
    }


@app.get("/stamps/classify/stats")
def stamps_classify_stats():
    state = _normalize_classify_state(_load_classify_state())
    items = state.get("items", {})
    per_class: dict[str, int] = {}
    per_user: dict[str, int] = {}
    validated = 0
    rejected = 0
    skipped = 0
    for meta in items.values():
        status = meta.get("status")
        if status == "validated":
            validated += 1
            label = meta.get("label") or ""
            per_class[label] = per_class.get(label, 0) + 1
            user = meta.get("user") or "anon"
            per_user[user] = per_user.get(user, 0) + 1
        elif status == "rejected":
            rejected += 1
        elif status == "skipped":
            skipped += 1
    preds = _load_classify_preds()
    threshold = _classify_conf_threshold()
    total = 0
    for name in _list_classify_crops():
        pred = preds.get(name)
        if pred and float(pred.get("confidence", 0.0) or 0.0) >= threshold:
            continue
        total += 1
    return {
        "validated": validated,
        "rejected": rejected,
        "skipped": skipped,
        "total": total,
        "per_class": per_class,
        "per_user": per_user,
    }
