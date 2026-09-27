"""M2 drip-rate inference wrapper (YOLOv8 + tracker when trained)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.schema import DripRateResult

WEIGHTS = Path(__file__).resolve().parent / "weights" / "drip_yolo.pt"


def infer_drip_rate(frame: np.ndarray | None) -> DripRateResult:
    if frame is None or frame.size == 0:
        return DripRateResult(status="error", detail="empty frame")
    if not WEIGHTS.exists():
        return DripRateResult(
            status="unavailable",
            detail="YOLO weights not trained yet — record labeled drip-chamber clips",
        )
    return DripRateResult(
        status="error",
        detail="weights file present but YOLO loader not implemented",
    )
