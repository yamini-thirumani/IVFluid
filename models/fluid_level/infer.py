"""M1 fluid-level inference wrapper.

Swap this file's internals when the CNN is trained. Callers must keep using
`infer_fluid_level(frame)` so the pipeline does not change.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.schema import FluidLevelResult

WEIGHTS = Path(__file__).resolve().parent / "weights" / "fluid_level.pt"


def infer_fluid_level(frame: np.ndarray | None) -> FluidLevelResult:
    if frame is None or frame.size == 0:
        return FluidLevelResult(
            status="error",
            detail="empty frame",
        )
    if not WEIGHTS.exists():
        return FluidLevelResult(
            status="unavailable",
            detail="CNN weights not trained yet — collect bag frames via the recorder",
        )
    # Training session will load PyTorch weights here.
    return FluidLevelResult(
        status="error",
        detail="weights file present but loader not implemented",
    )
