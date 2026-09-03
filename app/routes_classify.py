"""Flujo de clasificacion de recortes de sellos (stamps/classify).

Separado de app/routes_review.py: comparte con el la lectura del directorio
de clasificacion (app/classify_common.py) pero es dueno de su propio estado
(state.json, preds.json) y de sus propias rutas HTTP.
"""

import fcntl
import json
import os
import time
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from app.classify_common import _classify_dir

router = APIRouter()

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


@router.get("/stamps/classify/image/{name}")
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


@router.get("/stamps/classify", response_class=HTMLResponse)
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


@router.get("/stamps/classify/next")
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


@router.post("/stamps/classify/label")
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


@router.post("/stamps/classify/reject")
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


@router.post("/stamps/classify/skip")
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


@router.get("/stamps/classify/suggestion")
def stamps_classify_suggestion(name: str):
    preds = _load_classify_preds()
    info = preds.get(name) or {}
    return {
        "label": info.get("label", ""),
        "confidence": float(info.get("confidence", 0.0) or 0.0),
    }


@router.get("/stamps/classify/stats")
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
