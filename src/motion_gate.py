"""
motion_gate.py  --  the edge-efficiency piece (Verkada's signature move).

Cheap background-subtraction decides whether the expensive detector runs on
this frame. This is the same idea Verkada used to cut GPU load ~10x: run a
tiny motion check on every frame, only wake the detector when something moves.
"""
import cv2


class MotionGate:
    def __init__(self, min_area_ratio: float = 0.002, cooldown: int = 5):
        # MOG2 = lightweight adaptive background model, runs on CPU in real time
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=25, detectShadows=False
        )
        self.min_area_ratio = min_area_ratio
        self.cooldown = cooldown
        self._cooldown_left = 0

        # stats -- these are your headline "compute saved" numbers
        self.frames_seen = 0
        self.detector_calls = 0

    def should_process(self, frame) -> bool:
        """Return True if the detector should run on this frame."""
        self.frames_seen += 1
        mask = self._bg.apply(frame)
        # fraction of pixels that changed
        changed = cv2.countNonZero(mask) / (frame.shape[0] * frame.shape[1])

        moving = changed > self.min_area_ratio
        if moving:
            self._cooldown_left = self.cooldown

        run = moving or self._cooldown_left > 0
        if self._cooldown_left > 0:
            self._cooldown_left -= 1
        if run:
            self.detector_calls += 1
        return run

    @property
    def savings(self) -> float:
        """Fraction of detector calls avoided vs running every frame."""
        if self.frames_seen == 0:
            return 0.0
        return 1.0 - (self.detector_calls / self.frames_seen)

    def report(self) -> dict:
        return {
            "frames_seen": self.frames_seen,
            "detector_calls": self.detector_calls,
            "detector_calls_saved": self.frames_seen - self.detector_calls,
            "compute_saved_pct": round(100 * self.savings, 1),
        }
