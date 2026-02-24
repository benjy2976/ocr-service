import os
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, HttpUrl
from app.ocr_pipeline import DEFAULT_OUT_DIR, run_ocr, run_ocr_file, run_stamp_test

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


@app.get("/stamps/review", response_class=HTMLResponse)
def stamps_review():
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
        <div class="row">
          <button class="btn" id="prevBtn">Anterior</button>
          <button class="btn" id="nextBtn">Siguiente</button>
        </div>
        <div class="row">
          <button class="btn" id="zoomOutBtn">Zoom -</button>
          <button class="btn" id="zoomInBtn">Zoom +</button>
        </div>
        <button class="btn" id="saveBtn">Guardar</button>
        <button class="btn" id="addBtn">Agregar sello</button>
        <button class="btn" id="cancelBtn">Cancelar agregar</button>
        <div class="meta" id="meta"></div>
        <div class="list" id="boxList"></div>
        <p class="meta">Usa la lista para activar/desactivar cajas.</p>
        <p class="meta">Para agregar: click en “Agregar sello” y luego 2 clicks en la imagen.</p>
      </div>
      <div class="content">
        <canvas id="canvas"></canvas>
      </div>
    </div>
    <script>
      const canvas = document.getElementById('canvas');
      const ctx = canvas.getContext('2d');
      const meta = document.getElementById('meta');
      const prevBtn = document.getElementById('prevBtn');
      const nextBtn = document.getElementById('nextBtn');
      const saveBtn = document.getElementById('saveBtn');
      const addBtn = document.getElementById('addBtn');
      const cancelBtn = document.getElementById('cancelBtn');
      const boxList = document.getElementById('boxList');
      const zoomOutBtn = document.getElementById('zoomOutBtn');
      const zoomInBtn = document.getElementById('zoomInBtn');

      let SCALE = 0.5;
      let items = [];
      let idx = 0;
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
        meta.textContent = `(${idx + 1}/${items.length}) ${item.name}`;
        image.onload = draw;
        image.src = `/image/${encodeURIComponent(item.name)}`;
        fetch(`/stamps/review/labels?name=${encodeURIComponent(item.name)}`)
          .then(r => r.json())
          .then(data => { boxes = data.boxes; renderList(); draw(); });
        updateControls();
      }

      function updateControls() {
        prevBtn.disabled = addMode || idx === 0;
        nextBtn.disabled = addMode || idx === items.length - 1;
        saveBtn.disabled = addMode;
        cancelBtn.disabled = !addMode;
        addBtn.classList.toggle('active', addMode);
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

      saveBtn.addEventListener('click', () => {
        saveCurrent().then(() => alert('Guardado'));
      });

      prevBtn.addEventListener('click', () => { if (idx > 0) { idx--; loadItem(); } });
      nextBtn.addEventListener('click', () => {
        if (idx < items.length - 1) {
          saveCurrent().then(() => { idx++; loadItem(); });
        }
      });
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

      zoomInBtn.addEventListener('click', () => {
        SCALE = Math.min(2.0, SCALE + 0.1);
        draw();
      });
      zoomOutBtn.addEventListener('click', () => {
        SCALE = Math.max(0.2, SCALE - 0.1);
        draw();
      });

      fetch('/stamps/review/items')
        .then(r => r.json())
        .then(data => { items = data.items; loadItem(); });
    </script>
  </body>
</html>
"""
    return HTMLResponse(content=html)


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
