"""
app/main.py  --  minimal FastAPI dashboard (looping version).

Runs the two-camera pipeline in background threads and serves the annotated
frames + live compute-saved stats. When a video reaches its end, it restarts
from the beginning so the dashboard runs continuously.

Start with:  uvicorn app.main:app --reload
Then open:   http://localhost:8000
"""
import threading
import time
import cv2
import yaml
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from src.pipeline import CameraPipeline

app = FastAPI()
CFG = yaml.safe_load(open("configs/default.yaml"))
_latest = {}   # cam_id -> jpeg bytes
_stats = {}    # cam_id -> report dict


def _on_frame(cam_id, frame, report):
    ok, buf = cv2.imencode(".jpg", frame)
    if ok:
        _latest[cam_id] = buf.tobytes()
    _stats[cam_id] = report


def _start(cam_id, source):
    # LOOP: when the video ends, run it again from the top, forever
    while True:
        CameraPipeline(cam_id, source, CFG).run(on_frame=_on_frame)
        time.sleep(0.5)   # brief pause before restarting


# EDIT these two sources: file paths, or 0 / 1 for webcam + phone-cam
SOURCES = {0: "data/cam0.mp4", 1: "data/cam1.mp4"}


@app.on_event("startup")
def launch():
    for cam_id, src in SOURCES.items():
        threading.Thread(target=_start, args=(cam_id, src), daemon=True).start()


def _stream(cam_id):
    while True:
        f = _latest.get(cam_id)
        if f:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + f + b"\r\n"
        time.sleep(0.03)   # ~30 fps cap so the browser stream stays smooth


@app.get("/video/{cam_id}")
def video(cam_id: int):
    return StreamingResponse(_stream(cam_id),
                             media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/stats")
def stats():
    return _stats


@app.get("/", response_class=HTMLResponse)
def home():
    imgs = "".join(f'<img src="/video/{c}" width="480" style="margin:6px">' for c in SOURCES)
    return f"<h2>SentryCV</h2>{imgs}<p>Live stats at <a href='/stats'>/stats</a></p>"
