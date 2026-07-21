"""
metrics.py  --  turn raw counts into the numbers that impress a CV team.

The headline is compute_saved_pct from the motion gate. If you have MOT
ground-truth labels, motmetrics gives you MOTA / IDF1 too.
"""

def efficiency_summary(reports: list) -> dict:
    """reports = list of MotionGate.report() dicts, one per camera."""
    frames = sum(r["frames_seen"] for r in reports)
    calls = sum(r["detector_calls"] for r in reports)
    return {
        "total_frames": frames,
        "detector_calls": calls,
        "detector_calls_saved": frames - calls,
        "compute_saved_pct": round(100 * (1 - calls / max(frames, 1)), 1),
    }
