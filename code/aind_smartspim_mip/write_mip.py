import logging
import yaml
from pathlib import Path

import numpy as np

from tqdm import tqdm
from dask.distributed import Client, LocalCluster

from .utils import utils
from .params import params

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

def get_channels(channels):

    channel_data = []

    for key, ch in channels.items():
        if '488' in ch:
            channel_data.append([ch, 0])
        elif '561' in ch:
            channel_data.append([ch, 1])
        elif '445' in ch:
            channel_data.append([ch, 2])
    
    return channel_data

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

def copy_mip_results(output_folder: str, s3_path: str, results_folder: str):
    """
    Copies the smartspim fused results to S3

    Parameters
    -----------
    output_folder: str
        Path where the results are

    s3_path: str
        Path where the results will be
        copied in S3

    results_folder: str
        Results folder where the .txt
        will be placed
    """
    for out in utils.execute_command_helper(f"aws s3 cp --recursive {output_folder} {s3_path}"):
        logger.info(out)

    utils.save_string_to_txt(
        f"Stitched dataset saved in: {s3_path}",
        f"{results_folder}/output_mip_generation.txt",
    )

def main(
    data_description, 
    dataset_name
    ):
   
    '''
    Creating a multi channel MIP volume from full zarr

    Parameters
    ----------
    data : dict
        Metadata related to dataset
    dataset_name: str
        Labtrack ID
    Returns
    -------
    img_MIP: np.array
        array (t x c x Y x X x Z) where Z is the number of images and YxX is the resolution
    '''
    
    mip_configs = get_yaml_config('/code/aind_smartsmpim_mip/params/mip_configs.yml')


    ch_zarrs, dims = get_zarrs(path, channels])
            
    if plane == 'horizontal':
        scale = 2.0
        dim = dims.shape[0]
    elif plane == 'coronal':
        scale = 1.8
        dim = dims.shape[1]
    elif plane == 'sagittal':
        scale = 1.8
        dim = dims.shape[2]
        
    n_planes = np.ceil(mip_configs['depth'] / scale).astype(int)
    start_plane = half_step = np.ceil(n_planes / 2).astype(int)
    
    steps = np.arange(start_plane, dim, int(mip_configd['step'] / scale))
    
    for ch, ch_data in ch_zarrs.items():
        print(f'Creating MIP for {plame}\n')
        s = 0

        for step in tqdm(steps, total = len(steps)):
            
            if plane == 'horizontal':
                mip = ch_data['data'][(step - half_step):(step + half_step), : , :].max(axis = 0)
            elif plane == 'coronal':
                mip = ch_data['data'][:, (step - half_step):(step + half_step) , :].max(axis = 1)
            elif plane == 'sagittal':
                mip = ch_data['data'][:, :, (step - half_step):(step + half_step)].max(axis = 2)
            
            if s == 0:
                mip_array = np.zeros((1, 3, mip.shape[0], mip.shape[1], len(steps))).astype('uint16')
                
            mip_array[0, ch_data['index'], :, :, s] = mip
            s += 1
    
    return mip_array

if __name__ == "__main__":
    main()