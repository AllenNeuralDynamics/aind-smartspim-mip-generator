import os
import dask
import boto3
import argparse
import multiprocessing

import numpy as np
import dask.array as da

from tqdm import tqdm
from pathlib import Path
from imageio.v2 import imwrite
from matplotlib import pyplot as plt
from dask.distributed import LocalCluster, Client


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
                f"s3://{bucket}/{mip_dict['input_directory']}/image_tile_fusing/OMEZarr/{filt}.zarr/0/"
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