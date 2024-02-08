import logging
import os
from pathlib import Path

import numpy as np
import yaml
from dask.distributed import Client, LocalCluster
from tqdm import tqdm

from .params import params
from .utils import utils

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
        if "488" in ch:
            channel_data.append([ch, 0])
        elif "561" in ch:
            channel_data.append([ch, 1])
        elif "445" in ch:
            channel_data.append([ch, 2])

    return channel_data


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
    for out in utils.execute_command_helper(
        f"aws s3 cp --recursive {output_folder} {s3_path}"
    ):
        logger.info(out)

    utils.save_string_to_txt(
        f"Stitched dataset saved in: {s3_path}",
        f"{results_folder}/output_mip_generation.txt",
    )


def main(pipeline_config, mip_configs, smartspim_dataset_name, results_folder):
    """
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
    """

    channels = get_channels(pipeline_config["channel_translation"])

    zarr_path = f"/data/{smartspim_dataset_name}/image_tile_fusing/OMEZarr/"
    ch_zarrs, dims = utils.get_zarrs(zarr_path, channels)

    if mip_configs["plane"] == "horizontal":
        scale = 2.0
        dim = dims.shape[0]
    elif mip_configs["plane"] == "coronal":
        scale = 1.8
        dim = dims.shape[1]
    elif mip_configs["plane"] == "sagittal":
        scale = 1.8
        dim = dims.shape[2]

    n_planes = np.ceil(mip_configs["depth"] / scale).astype(int)
    start_plane = half_step = np.ceil(n_planes / 2).astype(int)

    steps = np.arange(start_plane, dim, int(mip_configs["step"] / scale))

    for ch, ch_data in ch_zarrs.items():
        print(f"Creating MIP for {plame}\n")
        s = 0

        for step in tqdm(steps, total=len(steps)):

            if plane == "horizontal":
                mip = ch_data["data"][
                    (step - half_step) : (step + half_step), :, :
                ].max(axis=0)
            elif plane == "coronal":
                mip = ch_data["data"][
                    :, (step - half_step) : (step + half_step), :
                ].max(axis=1)
            elif plane == "sagittal":
                mip = ch_data["data"][
                    :, :, (step - half_step) : (step + half_step)
                ].max(axis=2)

            if s == 0:
                mip_array = np.zeros(
                    (1, 3, mip.shape[0], mip.shape[1], len(steps))
                ).astype("uint16")

            mip_array[0, ch_data["index"], :, :, s] = mip
            s += 1

    zarr_path = f"{results_folder}/OMEZarr"

    if not os.exists(zarr_path):
        os.mkdir(zarr_path)

    utils.write_zarr(
        mip_array, mip_configs["plane"], mip_configs["chunking"], zarr_path
    )

    s3_path = f"s3://{mip_configs['bucket']}/{smartspim_dataset_name}/{mip_configs['s3_dir']}"
    lt_id = smartspim_dataset_name.split('_')[1]
    
    ng_params = {
        's3_dir': f"s3://{mip_configs['bucket']}/{smartspim_dataset_name}/{mip_configs['s3_dir']}",
        'filename': f"{mip_configs['plane']}_mip_link.json",
        'url': f"{s3_path}/OMEZarr/{mip_configs['plane']}_MIP.zarr",
        'name': f"{mip_configs['plane']} MIP: {lt_id}",
        'shader': mip_configs['shader'],
    }

    create_neuroglancer_json(ng_params, results_folder)

    copy_mip_results(
        results_folder, 
        f"s3://{mip_configs['bucket']}/{smartspim_dataset_name}/{mip_configs['s3_dir']}/", 
        results_folder
    )


if __name__ == "__main__":
    main()
