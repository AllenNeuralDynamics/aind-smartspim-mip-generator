import os
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Optional, Union

import dask.array as da

# IO types
PathLike = Union[str, Path]


def create_neuroglancer_json(ng_params, save_path):
    """
    Write json for the MIP neuroglancer link
    """

    fpath = os.path.join(ng_params["directory"], ng_params["filename"])
    data = da.from_zarr(ng_params["url"], 0).squeeze()
    pos = [int(data.shape[1] / 2), int(data.shape[2] / 2), int(data.shape[3] / 2), 0.5]

    json_body = {
        "ng_link": "https://aind-neuroglancer-sauujisjxq-uw.a.run.app/#!" + fpath,
        "title": ng_params["name"],
        "dimensions": {
            "z": [0.000002, "m"],
            "y": [0.0000018, "m"],
            "x": [0.0000018, "m"],
            "t": [0.001, "s"],
        },
        "position": pos,
        "crossSectionOrientation": [0, 0, -0.7071067690849304, 0.7071067690849304],
        "crossSectionScale": 3.5,
        "projectionOrientation": [
            -0.09444133937358856,
            -0.00713761243969202,
            -0.7126563191413879,
            0.6950905323028564,
        ],
        "projectionScale": 8192,
        "layers": [
            {
                "type": "image",
                "source": {
                    "url": "zarr://" + ng_params["url"],
                    "transform": {
                        "outputDimensions": {
                            "t": [0.001, "s"],
                            "c^": [1, ""],
                            "z": [0.000002, "m"],
                            "y": [0.0000018, "m"],
                            "x": [0.0000018, "m"],
                        }
                    },
                },
                "tab": "source",
                "shader": ng_params["shader"],
                "shaderControls": {
                    "red_channel": 500,
                    "green_channel": 500,
                    "blue_channel": 500,
                },
                "crossSectionRenderScale": 0.08,
                "channelDimensions": {"c^": [1, ""]},
                "name": ng_params["name"],
            }
        ],
        "selectedLayer": {"size": 350, "visible": True, "layer": ng_params["name"]},
        "layout": "xy",
    }

    with open(os.path.join(save_path, ng_params["filename"]), "w") as fp:
        json.dump(json_body, fp, indent=2)

    return


def get_zarr_params(plane):

    if plane == "coronal":
        res = [
            1.0,
            1.0,
            2.0,
            1.8,
            1.0,
        ]
    elif plane == "sagittal":
        res = [
            1.0,
            1.0,
            2.0,
            1.8,
            1.0,
        ]
    elif plane == "horizontal":
        res = [
            1.0,
            1.0,
            1.8,
            1.8,
            1.0,
        ]

    zarr_params = {
        "resolution": res,
        "axes_order": [
            "t",
            "c",
            "z",
            "y",
            "x",
        ],
        "units": [
            "millisecond",
            None,
            "micrometer",
            "micrometer",
            "micrometer",
        ],
        "types": [
            "time",
            "channel",
            "space",
            "space",
            "space",
        ],
        "levels": 1,
    }
    return zarr_params


def get_zarrs(input_directory, channels):
    """
    Get a dictionary with the zarrs for the channels requested

    Parameters
    ----------
    input_directory: Pathlike
        The main diectory on AWS that contains stitched images stored in the data folder

    channels: list[str]
        The subfolders within the stitched folder identifying which zarrs
        to pull

    Returns
    ----------
    zarrs: dict
        Dictionary with key = channel value = dask.array

    """

    zarrs = defaultdict(dict)
    for ch in channels:

        file = os.path.join(input_directory, ch[0] + ".zarr")
        ch_array = da.from_zarr(file, 0).squeeze()
        zarrs[zarr_ch]["data"] = ch_array
        zarrs[zarr_ch]["index"] = ch[1]

        dims = ch_array.shape

    return zarrs, dims


def create_folders(axes):
    """
    Create results subfolders for images divided by axis

    Parameters
    ----------
    axes: pathlike
        root pathway for subfolders

    Returns
    ----------
    None

    """
    for k, axis in axes.items():
        os.mkdir(f"../results/OMEZarr/{axis}_MIP.zarr")

    return


def write_zarr(img, axis, chunking, save_path):
    """
    Write OME-Zarr object from array

    Parameters
    ----------
    img : ArrayLike
        The array to be converted into zarr format. needs minimum 3 dimensions
    axis : str
        The axis of the given image
    chunking : list
        The current chunk size of each dimension
    save_path : PathLike
        the directory where you want to save the file

    Returns
    -------
    None.

    """

    params = get_zarr_params()

    if len(chunking) == 3:
        chunking = (1, 1, chunking[0], chunking[1], chunking[2])

    fname = os.path.join(save_path, f"{axis}_MIP.zarr")

    store = parse_url(fname, mode="w").store
    group = zarr.group(store=store)

    # The only dimensions you want to down sample are the X and Y.
    mip = multiscale(img, windowed_mean, (1, 1, 2, 2, 2))
    mip = [np.asarray(mip[i]) for i in range(params["levels"])]

    resolution, units, types = ngc.get_base_params(
        params["resolution"], params["axes_order"]
    )
    axes, trafos = ngc.get_axes_and_transforms(
        mip, params["axes_order"], units, resolution, types
    )

    write_multiscale(
        pyramid=mip,
        group=group,
        axes=axes,
        coordinate_transformations=trafos,
        storage_options=dict(chunks=chunking),
    )

    return


def execute_command_helper(
    command: str,
    print_command: bool = False,
    stdout_log_file: Optional[PathLike] = None,
) -> None:
    """
    Execute a shell command.

    Parameters
    ------------------------

    command: str
        Command that we want to execute.
    print_command: bool
        Bool that dictates if we print the command in the console.

    Raises
    ------------------------

    CalledProcessError:
        if the command could not be executed (Returned non-zero status).

    """

    if print_command:
        print(command)

    if stdout_log_file and len(str(stdout_log_file)):
        save_string_to_txt("$ " + command, stdout_log_file, "a")

    popen = subprocess.Popen(
        command, stdout=subprocess.PIPE, universal_newlines=True, shell=True
    )
    for stdout_line in iter(popen.stdout.readline, ""):
        yield str(stdout_line).strip()
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)

def read_json_as_dict(filepath: str) -> dict:
    """
    Reads a json as dictionary.

    Parameters
    ------------------------

    filepath: PathLike
        Path where the json is located.

    Returns
    ------------------------

    dict:
        Dictionary with the data the json has.

    """

    dictionary = {}

    if os.path.exists(filepath):
        with open(filepath) as json_file:
            dictionary = json.load(json_file)

    return dictionary

def save_string_to_txt(txt: str, filepath: PathLike, mode="w") -> None:
    """
    Saves a text in a file in the given mode.

    Parameters
    ------------------------

    txt: str
        String to be saved.

    filepath: PathLike
        Path where the file is located or will be saved.

    mode: str
        File open mode.

    """

    with open(filepath, mode) as file:
        file.write(txt + "\n")
