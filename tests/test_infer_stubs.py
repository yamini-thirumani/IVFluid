import numpy as np

from models.drip_rate.infer import infer_drip_rate
from models.fluid_level.infer import infer_fluid_level
from models.patient.infer import infer_patient


def test_wrappers_unavailable_without_weights():
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    assert infer_fluid_level(frame).status == "unavailable"
    assert infer_drip_rate(frame).status == "unavailable"
    assert infer_patient(frame).status == "unavailable"


def test_wrappers_error_on_empty():
    assert infer_fluid_level(None).status == "error"
