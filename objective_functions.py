import numpy as np


def _compute_functional_activations(root_activations):
    muscle_map = {  ## personalized muscle map from Abdallah
        "Il": [0, 1, 0, 0, 0, 0, 0.0],
        "Qd": [0, 0, 1, 0, 0, 0, 0],
        "BF": [0, 0, 1.9, 6.9, 12.1, 8.7, 0],
        "GM": [0, 0, 0, 0, 0, 0.5, 0.5],
        "TA": [0, 0, 0, 0, 1, 0, 0],
    }
    si_map = {
        "HF": [0.8, 0.2, -1, 0, 0],
        "AE": [0, 0, 0, 1, -1],
    }
    assert len(root_activations) == len(muscle_map["Il"])
    if isinstance(root_activations, np.ndarray):
        for k, v in muscle_map.items():  # normalize each muscle
            muscle_map[k] = np.array(v) / np.sum(v)
        muscle_activations = np.array(
            [m_coeffs @ root_activations for m_coeffs in muscle_map.values()]
        )
        target_activations = {
            k: si_coeffs @ np.log(1 + muscle_activations) / np.log(2)
            for k, si_coeffs in si_map.items()
        }

    else:
        raise ValueError("root_activations must be either np.ndarray or torch.Tensor")

    return target_activations


def _lateral_extra_factor(side1_activations, side2_activations):
    """
    ################################################################################
    ## TODO this last part!! dont really understand it (yet)
    ## I think is to avoid co-activation of both sides
    # -- but I can introduce that through neg weights, and have better control / view, tbh?
    ################################################################################
    diff_GM_Left = (
        muscle_recruitment_dict_left["GM"] - muscle_recruitment_dict_right["GM"]
    )
    diff_GM_Right = (
        muscle_recruitment_dict_right["GM"] - muscle_recruitment_dict_left["GM"]
    )
    sum_GM = (
        muscle_recruitment_dict_left["GM"] + muscle_recruitment_dict_right["GM"]
    )
    SI_GM_LR_Left=diff_GM_Left / sum_GM
    SI_GM_Total_Left= SI_GM_LR_Left * selectivity_tar_left_GM
    ################################################################################
    """
    diff = side1_activations - side2_activations
    sum = side1_activations + side2_activations
    return side1_activations * (diff / sum), side2_activations * (-diff / sum)


def compute_functional_selectivity(unit_activation):
    """From the functions that Abdallah gave us - gives SI for functional movements (HipFlexion & AnkleExtension)

    Unit weigths correspond to HipFlexion(Left), AnkleExtension(Left), HipFlexion(Right), AnkleExtension(Right)

    Note: this function should be maximized. To minimize, first multiply by -1.
    Note: this function ranges between 0 and 1.
    WARNING: it is assumed that roots are ordered as L1_DL, L1_DR, L2_DL, ...
    """
    left_activations = np.array(unit_activation[::2])
    right_activations = np.array(unit_activation[1::2])
    #
    target_selectivity_left = np.stack(
        list(_compute_functional_activations(left_activations).values())
    )
    target_selectivity_right = np.stack(
        list(_compute_functional_activations(right_activations).values())
    )

    # ### TESTING this
    # target_selectivity_left, target_selectivity_right = _lateral_extra_factor(
    #     target_selectivity_left, target_selectivity_right
    # )
    # ###

    target_selectivity = np.hstack(
        [target_selectivity_left, target_selectivity_right]
    ).squeeze()
    return target_selectivity


def maximize_minimize_activation(unit_activation, unit_weights, **kwargs):
    """Maximize activation of targets, minimize activation of off-targets.

    Instead of target activations, we now have weights for each target unit. Those will be positive for targets, and negative for off-targets.
    Bigger absolute weight means more relevance. A simple dot product will give us the objective function.

    Note: this function should be maximized. To minimize, first multiply by -1.

    Note: this function takes values between 1 (best possible value, all targets are fully activated and all off-targets are fully deactivated)
    and -1 (worst possible value, all targets are fully deactivated and all off-targets are fully activated).

    """
    assert len(unit_activation) == len(
        unit_weights
    ), "unit_activation and unit_weights must have the same length"

    if type(unit_activation) in [list, np.ndarray]:
        obj_function = np.dot(unit_activation, unit_weights)
    else:
        raise ValueError("act_per_root must be either np.ndarray or torch.Tensor")

    return obj_function


def functional_selectivity_index(unit_activation, unit_weights, **kwargs):
    """Simply compute an objective function from it"""
    target_selectivity = compute_functional_selectivity(unit_activation)

    ### suppress negative elements
    target_selectivity = np.clip(target_selectivity, a_min=0, a_max=None)

    obj_function = maximize_minimize_activation(target_selectivity, unit_weights)

    return obj_function  ## bring back to +-1 range
