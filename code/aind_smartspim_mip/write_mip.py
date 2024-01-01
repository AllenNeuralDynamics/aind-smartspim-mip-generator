import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from ._shared.types import PathLike
from .utils import utils
from .params import params

def dask_config():
    dask_folder = Path("/root/capsule/scratch")
    dask.config.set(
        {
            "temporary-directory": dask_folder,
            "local_directory": dask_folder,
            "tcp-timeout": "300s",
            "array.chunk-size": "384MiB",
            "distributed.comm.timeouts": {
                "connect": "300s",
                "tcp": "300s",
            },
            "distributed.scheduler.bandwidth": 100000000,
            "distributed.worker.memory.rebalance.measure": "optimistic",
            "distributed.worker.memory.target": False,  # 0.85,
            "distributed.worker.memory.spill": 0.92,  # False,#
            "distributed.worker.memory.pause": 0.95,  # False,#
            "distributed.worker.memory.terminate": 0.98,  # False, #

        }
    )

    return

def copy_MIP_results(output_folder: str, s3_path: str, results_folder: str):
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

def main(dataset_name):

    # get static variables for mip creation
    mip_configs = params.get_yaml('../code/aind_smartspim_mip/params/mip_configs.yml')
    mip_configs['input_directory'] = dataset_name
    mip_configs['s3_path'] = 's3://aind-open-data'

    logger.info(f"MIP generator input data: {mip_configs}")

    # create results folders
    utils.create_folders(mip_configs['axes'])

    # get the zarrs into dask arrays
    ch_zarrs = utils.get_zarrs(mip_configs)

    # initialize dask cluster
    dask_config()
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        processes=True,
        memory_limit="auto",
        dashboard_address=None,
    )

    client = Client(cluster)
    mip_depth = mip_configs['depth']

    # loop over axes and create images
    for axis, plane in mip_configs['axes'].items():
        n_planes = np.ceil(mip_configs['depth'] / mip_configs['resolution'][int(axis)]).astype(int) 

        if mip_configs['start_plane'] == -1:
            start_plane = half_step = np.ceil(n_planes / 2).astype(int)

        ch_zarrs = {k:da.moveaxis(v, axis, 0), for k, v in ch_zarrs if not isinstance(v, type(None))}

        steps = np.arange(start_plane, ch_zarrs[0].shape[0], mip_configs['step'])
        if (ch_zarrs[0].shape[0] - steps[-1] + 1) < half_step:
            steps = steps[:-1]

        for step in tqdm(steps, total = len(steps)):
            mip = np.zeros(
                (
                    ch_zarrs[0].shape[1], 
                    ch_zarrs[0].shape[2], 
                    3)
                ).astype('uint16')
            
            for k, ch, ch_zarrs.items():
                mip[:, :, k] = ch[(step - half_step):(step + half_step), : , :].max(axis = 0)

            filepath = Path('../results/{plane}_MIP_images').joinpath(f"plane_{plane}_depth_{step}um_mip_{mip_depth}um.tiff")
            imwrite(filepath, mip)

    
    return Path(f"{mip_configs['s3_path']}/{mip_configs['input_directory']}/{mip_configs['output_folder']}/")

if __name__ == "__main__":
    main()