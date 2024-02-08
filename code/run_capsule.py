""" top level run script """
import logging
import os
import json
import sys

from aind_smartspim_mip import write_mip
from aind_smartspim_mip.utils import utils

from glob import glob
from typing import Tuple

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s : %(message)s",
    datefmt="%Y-%m-%d %H:%M",
    handlers=[
        logging.StreamHandler(),
        # logging.FileHandler("test.log", "a"),
    ],
)
logging.disable("DEBUG")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def get_yaml_config(filename):
    """
    Get default configuration from a YAML file.

    Parameters
    ------------------------
    filename: str
        String where the YAML file is located.

    Returns
    ------------------------
    Dict
        Dictionary with the configuration
    """

    with open(filename, "r") as stream:
        config = yaml.safe_load(stream)

    return config

def get_data_config(
    data_folder: str,
    processing_manifest_path: str = "MIP_processing_manifest*",
    data_description_path: str = "data_description.json",
) -> Tuple:
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

    derivatives_dict = utils.read_json_as_dict(glob(f"{data_folder}/{processing_manifest_path}")[0])
    data_description_dict = utils.read_json_as_dict(f"{data_folder}/{data_description_path}")

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

    mode = str(sys.argv[1:])
    mode = mode.replace("[", "").replace("]", "").casefold()

    # Absolute paths of common Code Ocean folders
    data_folder = os.path.abspath("../data")
    results_folder = os.path.abspath("../results")

    pipeline_config, smartspim_dataset_name = get_data_config(data_folder=data_folder)
    mip_configs = get_yaml_config('/code/aind_smartsmpim_mip/params/mip_configs.yml')
    mip_configs['plane'] = mode
    
    logger.info(f"Dataset name: {smartspim_dataset_name}")
    image_path = write_mip.main(
        pipeline_config, 
        mip_configs,
        smartspim_dataset_name, 
        results_folder
    )
    
if __name__=="__main__":
    main()