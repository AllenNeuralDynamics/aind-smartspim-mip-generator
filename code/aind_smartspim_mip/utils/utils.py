
import os
import subprocess
import dask.array as da

from pathlib import Path
from typing import Any, List, Optional, Union

# IO types
PathLike = Union[str, Path]

def get_zarrs(mip_dict, bucket = 'aind-open-data'):
    """
    Get a dictionary with the zarrs for the channels requested

    Parameters
    ----------
    mip_dict: dict
        Dictionary with parameters related to zarr location

    Returns
    ----------
    zarrs: dict
        Dictionary with key = channel value = dask.array
    
    """
    zarrs = {
        0: None, 
        1: None, 
        2: None
    }

    for filt, axis in mip_dict['color_table']:
        try:
            ch_array = da.from_zarr(
                f"{mip_dict['input_data']}/{filt}.zarr/0/"
            ).squeeze()
        except:
            continue
        
        zarrs[axis] = ch_array
    
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

