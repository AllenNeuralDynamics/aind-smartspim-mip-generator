import os
import subprocess
import zarr

import dask.array as da

from collections import defaultdict
from pathlib import Path
from typing import Optional, Union
from ome_zarr.io import parse_url
from ome_zarr.writer import write_multiscale
from xarray_multiscale import multiscale, windowed_mean

# IO types
PathLike = Union[str, Path]

def create_neuroglancer_json(ng_params, save_path):
    """
    Write json for the MIP neuroglancer link

    Parameters
    ----------
    ng_params = dict
        dictionary with the necessary parameters to build the MIP json

    save_path = str
        where to save the json        

    Results
    --------
    None.

    """

    fpath = os.path.join(ng_params["directory"], ng_params["filename"])

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
                    "red_channel": ng_params['colors']['red'],
                    "green_channel": ng_params['colors']['green'],
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
    '''
    Get Parameters for crating a multi-scale zarr

    Parameters
    ----------
    plane: str
        The plane for each slice of the MIP

    Returns
    -------
    zarr_params: dict
        Dictonary with the resolution, axis order, units, and
        types for the zarr
    '''

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

def get_base_params(res, axes_order):
    
    resolutions = {k:v for (k, v) in zip(axes_order, res)}
    units = {
        "t": "millisecond", 
        "c": None, 
        "z": "micrometer", 
        "y": "micrometer",
        "x": "micrometer"
    }
    types = {
        "t": "time", 
        "c": "channel", 
        "z": "space", 
        "y": "space",
        "x": "space"
    }
    
    return resolutions, units, types

def get_axes_and_transforms(mip, axes, units, resolutions, types):

    axes_list = []
    for ax in axes:
        axis = {
            "name": ax,
            "type": types[ax],
            }
        
        if not isinstance(units[ax], type(None)):
            axis["unit"] = units[ax]

        axes_list.append(axis)

    transforms = []
    for scale_level in range(len(mip)):
        trafo = [
            {
                "scale": [resolutions[ax] * 2**scale_level if ax in "zyx" else resolutions[ax] for ax in axes],
                "type": "scale"
            }
        ]
        transforms.append(trafo)
    
    return axes_list, transforms

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

    params = get_zarr_params(axis)

    if len(chunking) == 3:
        chunking = (1, 3, chunking[0], chunking[1], chunking[2])

    fname = os.path.join(save_path, f"{axis}_MIP.zarr")

    store = parse_url(fname, mode="w").store
    group = zarr.group(store=store)

    # The only dimensions you want to down sample are the X and Y.
    mip = multiscale(img, windowed_mean, (1, 1, 2, 2, 2))
    mip = [np.asarray(mip[i]) for i in range(params["levels"])]

    resolution, units, types = get_base_params(
        params["resolution"], params["axes_order"]
    )
    axes, trafos = get_axes_and_transforms(
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

def get_color_info(mip_array):

    color_dict = {}
    for c, ch in enumerate(['red', 'green']):
        ch_array = mip_array[:, c, :, :, :].squeeze()
        print(f"ch array data type: {ch_array.dtype} and shape: {ch_array.shape}")
        hist = np.histogram(ch_array, bins = 65535, range=(0, 65535))
        minimas = argrelextrema(hist[0], np.less, order = 25)[0]

        if len(minimas) > 0:
            if minimas[0] == 1:
                thresh = minimas[1]
            else:
                thresh = minimas[0]
        else:
            thresh = 50

        thresh_vals = ch_array[ch_array > thresh]
        print(f"thresh_vals data type: {thresh_vals.dtype} and shape: {thresh_vals.shape}")
        ng_thresh = np.quantile(thresh_vals, .995, method='median_unbiased')
        
        color_dict[ch] = int(ng_thresh)
    
    return color_dict

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
