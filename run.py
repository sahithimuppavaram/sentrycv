"""Headless run: process both cameras, then print efficiency + do cross-camera re-ID."""
import yaml
from src.pipeline import CameraPipeline
from src.metrics import efficiency_summary

CFG = yaml.safe_load(open("configs/default.yaml"))
SOURCES = {0: "data/cam0.mp4", 1: "data/cam1.mp4"}

reports = []
for cam_id, src in SOURCES.items():
    reports.append(CameraPipeline(cam_id, src, CFG).run())

print("EFFICIENCY:", efficiency_summary(reports))

# Cross-camera re-ID (needs torchreid installed)
try:
    from src.reid import ReID
    reid = ReID(CFG["reid_similarity_threshold"])
    gallery = reid.build_gallery(CFG["crops_dir"])
    global_ids = reid.assign_global_ids(gallery)
    print("GLOBAL IDS:", global_ids)
    print("Unique people across cameras:", len(set(global_ids.values())))
except Exception as e:
    print("Re-ID step skipped:", e)
