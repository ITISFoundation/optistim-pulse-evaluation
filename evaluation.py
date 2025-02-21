import os
import sys

sys.path.append(os.path.dirname(__file__))


import numpy as np
from pathlib import Path
from get_pulse import get_pulse, DST
from s4l_neurofunctions.af.af_data_object import AFDataObject
from s4l_neurofunctions.af.gaf_calculator import GAFCalculatorHeterogeneous
from s4l_neurofunctions.af.titration_predictor import TitrationPredictor

from objective_functions import functional_selectivity_index
from neuron_analysis import get_activation_from_titration

roots = (
    ["L1_DL", "L1_DR", "L2_DL", "L2_DR", "L3_DL", "L3_DR"]
    + ["L4_DL", "L4_DR", "L5_DL", "L5_DR"]
    + ["S1_DL", "S1_DR", "S2_DL", "S2_DR"]
)
THRESHOLD = 14.0


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


def model(**kwargs):
    print(f"Evaluating Free Pulse for {kwargs}")
    params = [k for k in kwargs.values()]

    act, fsi = evaluate_activation(params)
    return {
        "activation": act,
        "fsi": fsi,
        "energy": evaluate_energy(params),
        "maxamp": evaluate_maxamp(params),
    }


## not defined a-priori in the sinusoid case
# model.__annotations__.update({"inputs": {f"A{i}": float for i in range(NVARS)}})
model.__annotations__.update(
    {"outputs": {"activation": float, "fsi": float, "energy": float, "maxamp": float}}
)  ## clearly custom-made annotations; do for now until we have the proper function database
## Users will still need to define inputs & outputs of their python functions when adding them there


def evaluate_maxamp(x) -> float:
    pulse = get_pulse(*x)
    return np.max(np.abs(pulse.amplitude_list))


def evaluate_activation(x) -> float:
    pulse = get_pulse(*x)
    gafc = GAFCalculatorHeterogeneous(dst=DST)
    gafc.compute_gaf(
        afdataloaded,
        pulse,
        force_recomputation=True,
        MODE="BruteForce",
    )

    peaks = gafc.get_peaks()

    gafmax = peaks.get_gaf_data().AF_max.values
    act = np.mean(sigmoid(gafmax, threshold=THRESHOLD, slope=2.0))  # type: ignore

    ## also get the functional SI
    tp = TitrationPredictor(peaks)
    tp.predict(threshold=THRESHOLD)
    unit_activations = get_activation_from_titration(
        titration_data=tp.data,
        key_list=roots,
        titration_factor_key="AFPredictedTitration",
    )

    #

    unit_weights = [-0.33, -0.33, 1, -0.33]

    fsi = functional_selectivity_index(list(unit_activations.values()), unit_weights)

    return (act, fsi)


def evaluate_energy(x) -> float:
    pulse = get_pulse(*x)

    pulse.plot_pulse()
    import matplotlib.pyplot as plt

    plt.savefig("_".join(["pulse"] + [f"{xx:.2e}" for xx in x]) + ".png")

    R = 2e3  ### Have not computed it - I guess I could. R = 1Vdiff/current. But it is just a scale factor, not very important.
    ## total work = sum I^2 * R * dt
    energy = [(i * 1e-3) ** 2 * R * (1 / DST * 1e-3) for i in pulse.amplitude_list]
    energy = sum(energy)
    return energy


if __name__ == "__main__":
    # print(evaluator(**{f"p{i+1}": 0.0 for i in range(int(DURATION / SEGMENT_PW) - 1)}))
    print(model(**{"A0": 0.0, "A1": 1.5, "B0": 0.5, "B1": 0.0}))
