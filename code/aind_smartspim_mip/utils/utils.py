
import os
import subprocess
import dask.array as da

from pathlib import Path
from typing import Optional, Union

# IO types
PathLike = Union[str, Path]

def get_zarr_params():
    zarr_params = {
        'resolution': 
            [
                1.0,
                1.0,
                2.0,
                1.8,
                1.8,
            ],
        'axes_order':
            [
                "t",
                "c",
                "z",
                "y",
                "x",
            ],
        'units':
            [
                "millisecond",
                None,
                "micrometer",
                "micrometer",
                "micrometer",
            ],
        'types':
            [
                "time",
                "channel",
                "space",
                "space",
                "space",
            ],
        'levels': 1
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
        
    zarrs = {}
    for zarr_ch in channels:
        file = os.path.join(input_directory, zarr_ch + '.zarr')
        ch_array = da.from_zarr(file, 0).squeeze()
        zarrs[zarr_ch] = ch_array

    return zarrs

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
        os.mkdir(f"../results/{axis}_MIP_images")

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

    popen = subprocess.Popen(command, stdout=subprocess.PIPE, universal_newlines=True, shell=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield str(stdout_line).strip()
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
        
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

