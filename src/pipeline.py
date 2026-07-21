"""
pipeline.py  --  per-camera detect + track, gated by MotionGate.

Uses ultralytics YOLOv8 with built-in ByteTrack. When the motion gate says
"idle", we skip the detector entirely and just carry the last annotated frame.
Person crops are saved per (camera, track_id) for the cross-camera re-ID step.
"""
import os
import cv2
from ultralytics import YOLO
from .motion_gate import MotionGate


class CameraPipeline:
    def __init__(self, cam_id, source, cfg):
        self.cam_id = cam_id
        self.source = source
        self.cfg = cfg
        self.model = YOLO(cfg["detector"])
        self.gate = MotionGate(cfg["motion_min_area_ratio"], cfg["motion_cooldown"])
        self.crops_dir = os.path.join(cfg["crops_dir"], f"cam{cam_id}")
        os.makedirs(self.crops_dir, exist_ok=True)
        self.last_annotated = None

    def run(self, on_frame=None, max_frames=None):
        cap = cv2.VideoCapture(self.source)
        i = 0
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            i += 1

            if self.gate.should_process(frame):
                # persist=True keeps ByteTrack IDs stable across frames
                res = self.model.track(
                    frame, persist=True, verbose=False,
                    classes=self.cfg["classes"], tracker=self.cfg["tracker"],
                )[0]
                annotated = res.plot()
                self._save_crops(frame, res)
                self.last_annotated = annotated
            else:
                # idle frame: reuse last annotated view, no detector cost
                annotated = self.last_annotated if self.last_annotated is not None else frame

            if on_frame:
                on_frame(self.cam_id, annotated, self.gate.report())
            if max_frames and i >= max_frames:
                break
        cap.release()
        return self.gate.report()

    def _save_crops(self, frame, res):
        if res.boxes is None or res.boxes.id is None:
            return
        for box, tid, cls in zip(res.boxes.xyxy, res.boxes.id, res.boxes.cls):
            if int(cls) != 0:      # only save people for re-ID
                continue
            x1, y1, x2, y2 = map(int, box.tolist())
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                continue
            path = os.path.join(self.crops_dir, f"id{int(tid)}_f{res.speed and ''}{y1}{x1}.jpg")
            cv2.imwrite(path, crop)
