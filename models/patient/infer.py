"""M3 patient-distress inference wrapper.

Phase-I: pretrained FER-style model. Do not collect a custom pain dataset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.schema import PatientResult

WEIGHTS = Path(__file__).resolve().parent / "weights" / "distress.pt"


def infer_patient(frame: np.ndarray | None) -> PatientResult:
    if frame is None or frame.size == 0:
        return PatientResult(status="error", detail="empty frame")
    if not WEIGHTS.exists():
        return PatientResult(
            status="unavailable",
            face_detected=False,
            detail="pretrained distress weights not selected/copied yet",
        )
    return PatientResult(
        status="error",
        detail="weights file present but FER loader not implemented",
    )
