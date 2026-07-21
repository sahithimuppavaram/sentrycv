# SentryCV — motion-gated multi-camera person tracking

A portfolio project built to mirror how **Verkada**'s CV team works: real-time
detection and tracking of people and vehicles, **across multiple cameras**, done
**cheaply at the edge** via motion gating, in a **privacy-respecting** way.

## Why each piece exists (the Verkada map)
| Stage | File | Mirrors at Verkada |
|---|---|---|
| Motion gate (edge) | `src/motion_gate.py` | Their motion-preprocessing that cut GPU load ~10x |
| Detect + track | `src/pipeline.py` | People/vehicle detection (YOLO) + tracking IDs |
| Cross-camera re-ID | `src/reid.py` | People Analytics: track people across cameras |
| Search + dashboard | `app/main.py` | Command platform: real-time review UI |

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# add data/cam0.mp4 and data/cam1.mp4  (see data/README.md)
python run.py                 # headless: prints compute-saved % + global IDs
uvicorn app.main:app --reload # dashboard at http://localhost:8000
```

## The numbers to report (this is what makes it read as engineering)
- **compute_saved_pct** — detector calls avoided by the motion gate (your headline)
- detection mAP, tracking MOTA / IDF1 (via `motmetrics` if you label a clip)
- re-ID rank-1 accuracy across the two cameras

## 3-day build plan
**Day 1 — one camera, the efficiency win.**
Get `cam0.mp4` in. Run `pipeline.py` on it. Confirm YOLO+ByteTrack draw stable
IDs. Tune `motion_min_area_ratio` until `compute_saved_pct` is meaningful
(aim 40-80% on typical footage). This alone is a strong story.

**Day 2 — second camera + cross-camera re-ID.**
Add `cam1.mp4`. Person crops auto-save per camera. Run the re-ID step in
`run.py` — verify the same person gets ONE global id across both views. Tune
`reid_similarity_threshold`.

**Day 3 — dashboard, metrics, Docker, writeup.**
Bring up the FastAPI dashboard (both feeds + live stats). Write the README
results section with your real numbers. Add a Dockerfile. Push to GitHub.

## Stretch (strong -> undeniable)
- Attribute/person search: "find the person in red" over the gallery embeddings
- Grad-CAM overlays on detections (explainability = trust)
- Privacy-by-default: blur faces unless toggled off  (echoes their mission)

## Notes
- `yolov8n.pt` auto-downloads on first run; move to `yolov8s`/`m` for accuracy.
- On idle (gated) frames the detector is skipped entirely — that's the point.
