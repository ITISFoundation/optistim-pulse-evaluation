import os
import sys

sys.path.append(os.path.dirname(__file__))


import numpy as np
from pathlib import Path
from typing import Dict
from get_pulse import get_pulse
from get_pulse import SEGMENT_PW, DURATION, NVARS
from s4l_neurofunctions.af.af_data_object import AFDataObject
from s4l_neurofunctions.af.gaf_calculator import GAFCalculatorHomogeneous


def deactivate_tqdm():
    import os

    os.environ["TQDM_DISABLE"] = "1"

    from tqdm import tqdm
    from functools import partialmethod

    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)  # type: ignore


def sigmoid(x: np.ndarray, threshold: float, slope: float) -> np.ndarray:
    return 1 / (1 + np.exp(-slope * (x - threshold)))


deactivate_tqdm()
afdataloaded = AFDataObject(af_type="AF").load(Path(__file__).parent)


def model(**kwargs: Dict[str, float]) -> Dict[str, float]:
    print(f"Evaluating Free Pulse for {kwargs}")
    params = [k for k in kwargs.values()]
    return {
        "1-activation": evaluate_activation(params),
        "energy": evaluate_energy(params),
        "maxamp": evaluate_maxamp(params),
    }


model.__annotations__.pop("kwargs")
model.__annotations__.update({f"p{i+1}": float for i in range(NVARS)})
# TODO dynamically get outputs?


def evaluate_maxamp(x) -> float:
    pulse = get_pulse(*x, segment_pw=SEGMENT_PW, duration=DURATION)
    return np.max(np.abs(pulse.amplitude_list))


def evaluate_activation(x) -> float:
    pulse = get_pulse(*x, segment_pw=SEGMENT_PW, duration=DURATION)

    gafc = GAFCalculatorHomogeneous(dst=10)
    gafc.compute_gaf(
        afdataloaded,
        pulse,
        force_recomputation=True,
        MODE="time_integration",
    )

    gafpeaks = gafc.get_peaks().get_gaf_data()
    gafmax = gafpeaks.AF_max.values
    act = np.mean(sigmoid(gafmax, threshold=14.0, slope=2.0))  # type: ignore
    return 1 - act  # type: ignore


def evaluate_energy(x) -> float:
    pulse = get_pulse(*x, segment_pw=SEGMENT_PW, duration=DURATION)
    R = 1.2e3  # 1.2 kOhm -- from the model, +-1V generates 1.65mA
    ## total work = sum I^2 * R * dt
    energy = [(i * 1e-3) ** 2 * R * (SEGMENT_PW * 1e-3) for i in pulse.amplitude_list]
    energy = sum(energy)
    return energy


if __name__ == "__main__":
    # def example(a: float, b:bool, s:str) -> (float, bool):
    #     return None
    #
    # example.__annotations__
    #    {'a': <class 'float'>, 'b': <class 'bool'>, 's': <class 'str'>, 'return': <class 'float'>}
    #                                                                    'return': [<class 'float'>, <class 'bool'>]}
    #
    # isinstance(1.0, example.__annotations__["a"])
    #     True

    print(model.__annotations__)
    print("DONE")
