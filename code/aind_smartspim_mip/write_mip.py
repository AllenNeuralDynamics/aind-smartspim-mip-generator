import logging
import dask
from pathlib import Path

import numpy as np
import dask.array as da

from tqdm import tqdm
from imageio.v2 import imwrite
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

@dask.delayed
def create_mip(ch_zarrs, step, half_step):

    mip = np.zeros(
        (
            ch_zarrs[0].shape[1], 
            ch_zarrs[0].shape[2], 
            3
        )
    ).astype('uint16')

    for channel, ch_array in ch_zarrs.items():
        mip[:, :, channel] = ch_array[(step - half_step):(step + half_step), : , :].max(axis = 0)
    
    return mip

@dask.delayed
def save_mip(mip, plane, step, mip_depth):

    filepath = Path('../results/{plane}_MIP_images').joinpath(f"plane_{plane}_depth_{step}um_mip_{mip_depth}um.tiff")
    
    imwrite(filepath, mip)

    return filepath

def create_multi_channel_mip(path, save_path, plane, channels, projection, step_um):
    '''
    Creating a multi channel MIP volume from full zarr

    Parameters
    ----------
    path : PathLike
        path to the root folder for zarrs on AWS
    save_path: PathLike
        where the zarr will be written
    plane: str
        Which plane you want to make a MIP for (i.e coronal, sagittal, horizontal)
    channels: list[str]
        list of channels that are to be included in MIP
    projection : int
        depth in um that you want for each projection
    step_um : int
        distance in microns you would like to have betwqeen each projection

    Returns
    -------
    img_MIP: np.array
        array (tx c x Y x X x Z) where Z is the number of images and YxX is the resolution
    '''
    
    ch_zarrs = get_zarrs(path, channels)
            
    if plane == 'horizontal':
        scale = 2.0
        dim = ch_zarrs[channels[0]].shape[0]
    elif plane == 'coronal':
        scale = 1.8
        dim = ch_zarrs[channels[0]].shape[1]
    elif plane == 'sagittal':
        scale = 1.8
        dim = ch_zarrs[channels[0]].shape[2]
        
    n_planes = np.ceil(projection / scale).astype(int)
    start_plane = half_step = np.ceil(n_planes / 2).astype(int)
    
    steps = np.arange(start_plane, dim, int(step_um / scale))
    
    for ch, ch_array in ch_zarrs.items():
        print(f'Creating MIP for {ch}\n')
        s = 0
        
        if '561' in ch:
            ch_idx = 0
        elif '488' in ch:
            ch_idx = 1
        
        for step in tqdm(steps, total = len(steps)):
            
            if plane == 'horizontal':
                mip = ch_array[(step - half_step):(step + half_step), : , :].max(axis = 0)
            elif plane == 'coronal':
                mip = ch_array[:, (step - half_step):(step + half_step) , :].max(axis = 1)
            elif plane == 'sagittal':
                mip = ch_array[:, :, (step - half_step):(step + half_step)].max(axis = 2)
            
            if s == 0:
                mip_array = np.zeros((1, 3, mip.shape[0], mip.shape[1], len(steps))).astype('uint16')
            
            mip_array[0, ch_idx, :, :, s] = mip
            s += 1
     
    with open(os.path.join(save_path, 'mip_rgb.pkl'), 'wb') as f:
        pickle.dump(mip_array, f)

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
    dask_workers = 8
    cluster = LocalCluster(
        n_workers=dask_workers,
        threads_per_worker=1,
        processes=True,
        memory_limit="auto",
        dashboard_address=None,
    )

    client = Client(cluster)
    mip_depth = mip_configs['depth']

    # loop over axes and create images
    for axis, plane in mip_configs['axes'].items():
        n_planes = np.ceil(mip_configs['depth'] / mip_configs['resolution'][int(axis)]).astype(int)
        step_um =  np.ceil(mip_configs['step'] / mip_configs['resolution'][int(axis)]).astype(int)

        if mip_configs['start_plane'] == -1:
            start_plane = half_step = np.ceil(n_planes / 2).astype(int)

        ch_zarrs = {k:da.moveaxis(v, axis, 0) for k, v in ch_zarrs if not isinstance(v, type(None))}

        steps = np.arange(start_plane, ch_zarrs[0].shape[0], step_um )
        if (ch_zarrs[0].shape[0] - steps[-1] + 1) < half_step:
            steps = steps[:-1]

        results = []
        for step in tqdm(steps, total = len(steps)):

            mip = create_mip(ch_zarrs, step, half_step)
            fpath = save_mip(mip, plane, step, mip_depth)
            results.append(fpath)

        results.compute()
        client.restart()
    

    output_folder = "../results"

    return output_folder

if __name__ == "__main__":
    main()