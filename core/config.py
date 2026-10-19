from pathlib import Path
import os
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FFMPEG_BINARY = shutil.which("ffmpeg") or "ffmpeg"
AIS_API_KEY = os.getenv("AIS_API_KEY", "").strip()

# Recommended maximum vessel speed within detection zone (knots)
SLOWDOWN_SPEED_KNOTS = 7.0
BASELINE_SPEED_KNOTS = 15.0

# Detection confidence threshold across inference and evaluation
DETECTION_THRESHOLD = 0.85

NAV_STATUS_MAP = {
    0: "Underway (Engine)", 1: "At Anchor", 2: "Not Under Command",
    3: "Restricted Maneuverability", 4: "Constrained by Draught",
    5: "Moored", 6: "Aground", 7: "Engaged in Fishing",
    8: "Underway (Sailing)", 15: "Undefined"
}

