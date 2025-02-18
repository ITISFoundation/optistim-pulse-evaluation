import glob

import os
import pickle
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

############################ Everything below is about Titration ##################################################
## TODO maybe make a Titration Extractor // Data Object class, which integrates the methods below?
# Give it a neuron simulation, save with that as a column, then save by it (as I save AF by emsim)
# ideally also from which emsim it comes from (if possible to get)


def save_titration(titr_df: pd.DataFrame, results_dir: str, name: str):
    """Save a titration dataframe to a csv file.

    Args:
        titr_df: DataFrame to be saved.
        path: Union[str, Path] to the folder where the file will be saved.
        name: Name of the file to be saved.
    """
    if name.endswith(".csv"):
        name = name[:-4]
    os.makedirs(os.path.join(results_dir, "Neuron"), exist_ok=True)
    titr_df.to_csv(os.path.join(results_dir, "Neuron", name + ".csv"))


def load_titration_csv(path: Path, name: str) -> pd.DataFrame:
    """Load a titration dataframe from a csv file.

    Args:
        path: Path to the folder where the file is saved.
        name: Name of the file to be loaded.

    Returns:
        titr_df : DataFrame loaded from the file.
    """
    if name.endswith(".csv"):
        name = name[:-4]
    return pd.read_csv(path / (name + ".csv"), index_col=0)


## Keep the function, but not use it as automatic fallback for load_titration() if no CSV is found
# we do not want to load the old data - in v6.2 sampling was different
def load_titration_pickle(path: Path, name: str) -> pd.DataFrame:
    # print(
    #     "WARNING: this function is for a specific old data format and will not work \
    #       with generic data pickle files which are not formatted in this specific way."
    # )

    if name.endswith(".pkl"):
        name = name[:-4]
    data = pickle.load(open(path / (name + ".pkl"), "rb"))
    df = pd.DataFrame(
        data,
        index=[
            "NeuroSimSpikeNode_fullstr",
            "NeuroSimSpikeTime",
            "NeuroSimTitrationFactor",
        ],
    ).T
    df.reset_index(inplace=True)
    df.rename(columns={"index": "Axon"}, inplace=True)
    df["NeuroSimTitrationFactor"] = df["NeuroSimTitrationFactor"].astype(float)
    df["NeuronSimulation"] = name
    df["NeuroSimSpikeNode"] = df["NeuroSimSpikeNode_fullstr"].apply(
        lambda x: int(x.split("[")[-1].split("]")[0])
    )
    df["Axon"] = (
        df["Axon"].astype(str).apply(lambda x: x.split("  (")[0])
    )  # remove group from name

    df["NeuronModel"] = name.split(" - ")[1]
    df["Pulse"] = name.split(" - ")[-2] + "_pulse"
    emsim = name.split(" - ")[-1]
    electrode = emsim.split("_")[0]
    paddle = "_".join(emsim.split("_")[1:])
    df["EMSim"] = " - ".join(["Monopolar Stim", electrode, paddle])

    return df


## TODO eventually substitute the single load_titration with this one, with name as optional input (as in load AFData)
def load_titrations(path: Path) -> pd.DataFrame:
    """Load all titration dataframes from a folder."""
    list_df = [
        load_titration_csv(path, os.path.basename(file))
        for file in glob.glob(str(path / "*.csv"))
    ]

    ### We actually dont want to load old data - titrations computed with v6.2 might have used different fiber sampling and give incorrect results
    # ## pickle data is old, and not used anymore. Nevertheless, at some point it is useful to load it.
    # if len(list_df) == 0:
    #     print("No CSV titration data found. Trying to load pickle data...")
    #     print(
    #         "WARNING: load_titration_pickle() is for a specific old data format and will not work \
    #       with generic data pickle files which are not formatted in this specific way."
    #     )
    #     list_df = [
    #         load_titration_pickle(path, os.path.basename(file))
    #         for file in glob.glob(str(path / "*.pkl"))
    #     ]
    #     assert len(list_df) > 0, "No titration data found in folder " + str(path)
    #     print(
    #         "Diameter can not be loaded from pickle data. Please set them manually, matching from AF data."
    #     )

    return pd.concat(list_df, ignore_index=True)


############ New version of recruitment curves, more generic, also for AF / GAF prediction data ###########


## TODO maybe do MultiIndex in the future? Now not worth it, too much effort
def split_neuron_df(neuron_df: pd.DataFrame, key_list: List[str]) -> pd.DataFrame:
    """Takes a DataFrame with neuron data, and splits it into groups,
    generating a MultiIndex DataFrame in which the first level are the provided keys.

    Keys are matched into the neuron's name, and the neuron's data is added to the group DataFrame.
    Keys can be nerve names, spinal roots, fascicles inside a nerve...
    The resulting dictionary contains all keys in 'key_list', even those that for which no matches were found (which are empty).
    The resulting dictionary allows for statistical analysis of groups of interest (those provided by the input keys).

    WARNING: The division is carried out based on the key exactly matching somewhere in the name of the neuron.
    If the neuron's name contains multiple keys, it will be added to all of them.
    For example, for separate keys "L5", "DR" and "L5_DR", a neuron whose name contains "L5_DR_..." will be added to all of them.

    Args:
        neuron_df: DataFrame containing the neuron data.
        key_list: List of keys to be used to split the neuron data.

    Returns:
        split_neuron_df: DataFrame containing the neuron data, split into the provided groups as first index.
    """
    raise NotImplementedError
    neuron_data_dict = {k: [] for k in key_list}
    for i, row in neuron_data.iterrows():
        for k in key_list:
            if k in row[key]:
                neuron_data_dict[k].append(row)
    for k in key_list:
        neuron_data_dict[k] = pd.DataFrame(neuron_data_dict[k])
    return neuron_data_dict


def divide_neuron_df(
    neuron_df: pd.DataFrame, key_list: List[str]
) -> Dict[str, pd.DataFrame]:
    """Divide the titration data into groups, based on the provided keys.

    Keys are matched into the neuron's name, and the neuron's titration factor is added to the group list.
    Keys can be nerve names, spinal roots, fascicles inside a nerve...
    The resulting dictionary contains all keys in 'key_list', even those that for which no matches were found (which are empty).
    The resulting dictionary allows for statistical analysis of groups of interest (those provided by the input keys).
    Note that individual neuron's information is lost in this operation.

    WARNING: The division is carried out based on
    the key exactly matching somewhere in the name of the neuron.
    If a neuron's name matches several keys, its results will be added to all of them.

    Args:
        titration_data: Pandas DataFrame of titration results per neuron. Neuron names are expected in the "Axon" column.
        key_list: Those keys which are matched into the neuron's name
            will have the neuron's data added to their entry. The resulting dictionary
            contains all keys in 'key_list', even those that for which no matches were found.
    Returns:
        Dictionary of pandas DataFrames, divided by the provided input keys.
    """
    if "Axon" in neuron_df.columns:
        axons = neuron_df["Axon"]
    else:
        assert isinstance(neuron_df.index.values[0], str)
        axons = neuron_df.index

    divided_neuron_dict = {}
    for key in key_list:
        divided_neuron_dict[key] = neuron_df[axons.str.contains(key)]

    ## check that all keys have at least one value
    for key in key_list:
        if len(divided_neuron_dict[key]) == 0:
            print("WARNING. No neurons were detected for " + key + ".")

        ## print warnings if the amount of values in the dictionary does not match the amount of data
    if np.sum([len(divided_neuron_dict[key]) for key in key_list]) < len(neuron_df):
        print("WARNING. Some axons were not assigned to any key.")
    elif np.sum([len(divided_neuron_dict[key]) for key in key_list]) > len(neuron_df):
        print("WARNING. Some axons were assigned to multiple keys.")

    return divided_neuron_dict


### TO BE Deprecated. Use "divide_neuron_df" instead. ### Still used for get_recruitment_curves and get_activation_from_titration
def divide_titration_data(
    titration_data: pd.DataFrame,
    key_list: Optional[List[str]] = None,
    column: Optional[str] = None,
    partial_match_allowed: bool = True,
    titration_factor_key: str = "NeuroSimTitrationFactor",
) -> Dict[str, List[float]]:
    """Divide the titration data into groups, based on the provided keys (or all unique keys existing in the specified column)

    Keys are matched into the specified column (by default, the neuron's name).
    The neuron's titration factor is added to the group list.
    Keys can be nerve names, spinal roots, fascicles inside a nerve...
    The resulting dictionary contains all keys in 'key_list', even those that for which no matches were found (which are empty).
        A warning will be raised in for each key without matches..
    The resulting dictionary allows for statistical analysis of groups of interest (those provided by the input keys).
    Note that individual neuron's information is lost in this operation.

    WARNING: The division is carried out based on
    the key exactly matching somewhere in the name of the neuron (or the specified column)
    If a neuron's data matches several keys, its results will be added to all of them.

    Args:
        titration_data: Pandas DataFrame of titration results per neuron. Neuron names are expected in the "Axon" column.
        key_list: Those keys which are matched into the neuron's name
            will have the neuron's data added to their entry. The resulting dictionary
            contains all keys in 'key_list', even those that for which no matches were found.
        column: Name of the column in the pandas dataframe containing the key to be matched. "Axon" by default.
        titration_factor_key: Name of the column in the titration data containing the titration factor.
    Returns:
        Dictionary of titration factor lists, divided by the provided input keys.
    """

    ## NOT YET
    # print(
    #     "WARNING: 'divide_titration_data' is deprecated. Use 'divide_neuron_df' instead."
    # )

    ## DEPRECATED. Allow to split by EMSim, Pulse, or whatever column values
    # ## First make sure that the data comes from a single EMSim and Pulse
    # if "EMSim" in titration_data.columns:
    #     assert (
    #         len(titration_data.EMSim.unique()) == 1
    #     ), "Dataframe must contain data from a single EMSim."
    # elif "NeuronSimulation" in titration_data.columns:
    #     assert (
    #         len(titration_data.NeuronSimulation.unique()) == 1
    #     ), "Dataframe must contain data from a single EMSim."
    # else:
    #     raise ValueError(
    #         "Dataframe must contain either NeuronSimulation or EMSim column."
    #     )
    # assert (
    #     len(titration_data.Pulse.unique()) == 1
    # ), "Dataframe must contain data from a single Pulse."

    if column is None:
        column = "Axon"
        assert (
            key_list is not None
        ), "If column is None, key_list must be provided. Otherwise each axon will be assigned to its own key and this function is useless."

    assert (
        column in titration_data.columns
    ), f"Dataframe must does not contain the {column} column."

    if key_list is None:
        key_list = titration_data[column].unique()

    def matching_key(key: str, value: str, partial_match_allowed: bool) -> bool:
        if partial_match_allowed:
            return key in value
        else:
            return key == value

    divided_titration_dict = {}
    for key in key_list:
        divided_titration_dict[key] = [
            data_row[titration_factor_key]
            for idx, data_row in titration_data.iterrows()
            if matching_key(key, data_row[column], partial_match_allowed)
        ]

    ## check that all keys have at least one value
    for key in key_list:
        if len(divided_titration_dict[key]) == 0:
            print("WARNING. No neurons were detected for " + key + ".")

    ## print warnings if the amount of values in the dictionary does not match the amount of data
    if np.sum([len(divided_titration_dict[key]) for key in key_list]) < len(
        titration_data
    ):
        print("WARNING. Some axons were not assigned to any key.")
    elif np.sum([len(divided_titration_dict[key]) for key in key_list]) > len(
        titration_data
    ):
        print("WARNING. Some axons were assigned to multiple keys.")

    return divided_titration_dict


def get_activation_from_titration(
    titration_level: float = 1.0, *args, **kwargs
) -> Dict[str, float]:
    """Calculate the activition per unit for a given titration level, for each key in key_list.
    Activation is normalized to the [0,1] range, where 0 denotes no fiber being active, and 1 denotes all fibers being active.

    Default titration level is 1.0, which means that the function returns the proportion of fibers that are active at the original inputs given to the simulation.
    If a higher level of titration is given, more fibers will be active under such increased stimulation, and a higher activation per unit will be obtained.

    Args:
        Please refer to the documentation of the function 'divide_titration_data' for the meaning of the arguments.

    Returns:
        Dictionary of activations (float between 0 and 1) for each key in key_list.
    """
    divided_titration_dict = divide_titration_data(*args, **kwargs)
    activity_per_unit_dict = {}
    for key, tf_list in divided_titration_dict.items():
        if len(tf_list) != 0:
            activity_per_unit_dict[key] = sum(
                np.array(tf_list) <= titration_level
            ) / len(tf_list)
        else:
            activity_per_unit_dict[key] = 0.0

    return activity_per_unit_dict


def get_recruitment_curves(
    *args,
    **kwargs,  ## for divide_titration_data
) -> Dict[str, Dict]:
    """Provided a dataframe with titration factor data, return the recruitment curves, grouped by a list of keys.

    Args:
        Please refer to the documentation of the function 'divide_titration_data' for the meaning of the arguments.

    Returns:
        Dictionary with the provided input keys. Inside each item, two items can be found:
            - "x" / "TitrationDivisions": values of titration upon which one )or more)
                fiber(s) become active. Horizontal axis of a titration plot.
            - "y" / "RecruitmentValues": proportion of the fibers active at a given
                titration value (corresponding entry in "TitrationDivisions").
                Normalized to 1.
    """
    # # TODO implement "lims_tit" upstream and downstream
    # lims_tit: Optional[Tuple[float, float]] = None,
    divided_titration_dict = divide_titration_data(*args, **kwargs)
    key_list = list(divided_titration_dict.keys())
    recruitment_dict = {k: {} for k in key_list}
    for key in key_list:
        tf_list = divided_titration_dict[key]
        if len(tf_list) != 0:
            tdiv = np.sort(np.unique(tf_list))
            rec = np.array([sum(tf_list <= t) for t in tdiv])
            rec = rec / len(tf_list)  # Normalize recruitment by number of fibers
        else:
            tdiv, rec = [], []

        recruitment_dict[key]["TitrationDivisions"] = tdiv
        recruitment_dict[key]["RecruitmentValues"] = rec
        ## shorthand notation for plots
        recruitment_dict[key]["x"] = tdiv
        recruitment_dict[key]["y"] = rec
    return recruitment_dict


def _resample_recruitment_curve(
    tdiv: np.ndarray,
    rec: np.ndarray,
    drec: float,
    lims_rec_curve: Tuple[float, float],
    log_scale: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample the recruitment curve, to get a regular sampling between specified limits with interval drec."""
    if len(tdiv) == 0:
        return tdiv, rec
    if log_scale:
        tdiv_new = 10 ** np.arange(
            np.log10(lims_rec_curve[0]), np.log10(lims_rec_curve[1]), drec
        )
    else:
        tdiv_new = np.arange(lims_rec_curve[0], lims_rec_curve[1], drec)
    rec_new = np.interp(tdiv_new, tdiv, rec)
    return tdiv_new, rec_new


def get_difference_in_recruitment_curves(
    recr_dict_1: Dict[str, Dict],
    recr_dict_2: Dict[str, Dict],
    drec: float = 0.01,
    log_scale: bool = False,
):
    """Given two sets of recruitment curves, interpolate them onto a common grid, and return the difference.
    Allows plotting, and computation of the area under the difference curve."""

    assert set(recr_dict_1.keys()) == set(
        recr_dict_2.keys()
    ), "The two dictionaries must have the same keys."

    # get the common grid
    tf_list = [dict["x"] for k, dict in recr_dict_1.items()] + [
        dict["x"] for k, dict in recr_dict_2.items()
    ]
    tf_list = np.hstack(tf_list)
    lims_rec_curve = (0.1, np.max(tf_list))

    # interpolate the recruitment curves
    recr_dict_1_interp = {}
    recr_dict_2_interp = {}
    for k in recr_dict_1:
        tdiv, rec = _resample_recruitment_curve(
            recr_dict_1[k]["x"], recr_dict_1[k]["y"], drec, lims_rec_curve, log_scale
        )
        recr_dict_1_interp[k] = {
            "x": tdiv,
            "y": rec,
            "TitrationDivisions": tdiv,
            "RecruitmentValues": rec,
        }

        tdiv, rec = _resample_recruitment_curve(
            recr_dict_2[k]["x"], recr_dict_2[k]["y"], drec, lims_rec_curve, log_scale
        )
        recr_dict_2_interp[k] = {
            "x": tdiv,
            "y": rec,
            "TitrationDivisions": tdiv,
            "RecruitmentValues": rec,
        }

    # compute the difference
    recr_dict_diff = {
        k: {
            "x": recr_dict_1_interp[k]["x"],
            "y": recr_dict_1_interp[k]["y"] - recr_dict_2_interp[k]["y"],
            "TitrationDivisions": recr_dict_1_interp[k]["x"],
            "RecruitmentValues": recr_dict_1_interp[k]["y"]
            - recr_dict_2_interp[k]["y"],
        }
        for k in recr_dict_1_interp
    }

    return recr_dict_diff


#################################################################


def plot_recruitment_curves(
    recruitment_dict: dict,
    title: str = "",
    color_function: Optional[Callable] = None,
    linestyle_function: Optional[Callable] = None,
    add_initial_final_points: bool = True,
    titr_max: Optional[float] = None,
    savepath: Optional[str] = None,
    log_scale: bool = True,
    scatter_at_fiber_titrations: bool = True,
    scatter_size: int = 3,
    scale_to_percentages: bool = True,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Basic plot of recruitment curves, for quick visualization.
    Application-specific visualization are nevertheless encouraged.
    Feel free to re-use this code as desired.

    Args:
        recruitment_dict (dict)
        title (str): Used for the title of the plot. "Nerve Recruitment" gets appended.
        add_initial_final_points (bool): Add an initial point at (0,0) and
            a final point at (max_TitFactor * 1.1, 1), to improve the visibility
            of the first and last real points. Defaults to True.
    """
    groups = list(recruitment_dict.keys())

    if color_function is None:
        color_per_group = {group: "C" + str(i) for i, group in enumerate(groups)}
    else:
        color_per_group = {group: color_function(group) for group in groups}

    if linestyle_function is None:
        line_per_group = {group: "--" for group in groups}
    else:
        line_per_group = {group: linestyle_function(group) for group in groups}

    max_tf = np.max([np.max(recruitment_dict[group]["x"]) for group in groups])

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for group in groups:
        tdiv = recruitment_dict[group]["x"]
        rec = recruitment_dict[group]["y"]
        if scale_to_percentages:
            rec = rec * 100

        # Plot points
        if scatter_at_fiber_titrations:
            ax.scatter(tdiv, rec, c=color_per_group[group], s=scatter_size)

        # Plot lines in between (with additional initial and final point for better visualization)
        if add_initial_final_points and len(tdiv) != 0:
            ## Add additional points at (0,0) and (1.1*max_titf, 1)
            ## for better looking plots (better visibility of first and last real points)
            tdiv = np.concatenate([np.array([0]), tdiv, np.array([1.1 * max_tf])])
            rec = np.concatenate([np.array([0]), rec])  ## add initial 0 point
            rec = (
                np.concatenate([rec, np.array([100])])
                if scale_to_percentages
                else np.concatenate([rec, np.array([1])])  ## add final 1 / 100% point
            )
        ax.plot(
            tdiv,
            rec,
            label=group,
            color=color_per_group[group],
            linestyle=line_per_group[group],
        )

    # Add labels and title
    ax.legend(loc="upper left")
    ax.set_xlabel("Titration")

    if scale_to_percentages:
        ax.set_ylabel("Activation level (%)")
    else:
        ax.set_ylabel("Activation level (normalized)")

    if log_scale:
        ax.set_xscale("log")
    if titr_max is not None:
        ax.set_xlim(0.1, titr_max)
    # plt.xlim(0.5, 10)
    # plt.tight_layout()
    tit = " - ".join(["Nerve recruitment", title])
    ax.set_title(tit)
    # plt.show(block=False)
    if savepath is not None:
        plt.savefig(os.path.join(savepath, tit + ".png"))
    return ax


def plot_difference_recruitment_curves(
    recr_dict_1, recr_dict_2, drec=0.01, *args, **kwargs
):
    """Adapts the plot_recruitment_curves function to plot the difference between two recruitment curves.
    Please refer to the documentation of plot_recruitment_curves for more information.
    """
    diff_recr_dict = get_difference_in_recruitment_curves(
        recr_dict_1, recr_dict_2, drec
    )
    kwargs["add_initial_final_points"] = False
    kwargs["scatter_at_fiber_titrations"] = False
    ax = plot_recruitment_curves(diff_recr_dict, *args, **kwargs)
    ax.set_ylabel("Difference in activation level (normalized)")
    ax.set_ylim(-1.1, 1.1)
    plt.show(block=False)


def _get_color_lines_spinal_roots(roots: List[str]) -> Tuple[dict, dict]:
    ### get a list of all segments present, and assign them colors in automatic order
    segments = []
    for r in roots:
        seg = r.split("_")[0]
        if seg not in segments:
            segments.append(seg)
    colors_per_segment = {seg: "C" + str(i) for i, seg in enumerate(segments)}
    colors_per_root = {r: colors_per_segment[r.split("_")[0]] for r in roots}
    # quadrants always get the same linestyle. Thus, segment+quadrant = unique color+linestyle combination.
    lines_per_quadrant = {"DL": "-", "DR": "--", "VR": ":", "VL": "-."}
    lines_per_root = {r: lines_per_quadrant[r.split("_")[1]] for r in roots}
    return colors_per_root, lines_per_root


def distance_between_recruitment_curves(
    recr_dict_1, recr_dict_2, metric="euclidean", drec=0.01, log_scale=False
):
    """Compute distance between two sets of recruitment curves, by interpolating them onto a common grid, and computing a distance metric between them."""
    recr_dict_diff = get_difference_in_recruitment_curves(
        recr_dict_1, recr_dict_2, drec, log_scale
    )
    dist_dict = {}
    for k in recr_dict_diff:
        if metric in ["euclidean"]:
            dist_dict[k] = np.sqrt(np.sum(recr_dict_diff[k]["y"] ** 2)) * drec
        elif metric in ["manhattan"]:
            dist_dict[k] = np.sum(np.abs(recr_dict_diff[k]["y"])) * drec
        elif metric in [
            "max",
            "chebyshev",
            "infinity_norm",
        ]:  # aka chebyshev or infinity norm
            dist_dict[k] = np.max(np.abs(recr_dict_diff[k]["y"]))
        else:
            raise ValueError("Unknown metric: " + metric)
    return dist_dict, np.mean(list(dist_dict.values()))
