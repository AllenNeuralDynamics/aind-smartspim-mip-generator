""" top level run script """
import os
from pathlib import Path
from typing import List, Tuple

from aind_smartspim_fuse import fuse
from aind_smartspim_fuse.params import get_yaml
from aind_smartspim_fuse.utils import utils


def get_data_config(
    data_folder: str,
    processing_manifest_path: str = "processing_manifest.json",
    data_description_path: str = "data_description.json",
):
    """
    Returns the first smartspim dataset found
    in the data folder

    Parameters
    -----------
    data_folder: str
        Path to the folder that contains the data

    processing_manifest_path: str
        Path for the processing manifest

    data_description_path: str
        Path for the data description

    Returns
    -----------
    Tuple[Dict, str]
        Dict: Empty dictionary if the path does not exist,
        dictionary with the data otherwise.

        Str: Empty string if the processing manifest
        was not found
    """

    # Returning first smartspim dataset found
    # Doing this because of Code Ocean, ideally we would have
    # a single dataset in the pipeline

    derivatives_dict = utils.read_json_as_dict(
        f"{data_folder}/{processing_manifest_path}"
    )
    data_description_dict = utils.read_json_as_dict(
        f"{data_folder}/{data_description_path}"
    )

    smartspim_dataset = data_description_dict["name"]

    return derivatives_dict, smartspim_dataset

def read_json_as_dict(filepath: str):
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

def main():
    """
    Main function to execute the smartspim MIP generator
    in code ocean
    """

    data_folder = os.path.abspath("../data/")
    data_description_path = f"{data_folder}/data_description.json"

    data_description = read_json_as_dict(data_description_path)
    dataset_name = data_description["name"]

    logger.info(f"Dataset name: {dataset_name}")
    image_path = write_mip.main(dataset_name)
    

if __name__=="__main__":
    main()