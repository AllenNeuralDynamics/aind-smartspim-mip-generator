"""
Module to declare the parameters for the stitching package
"""
import platform

import yaml
from argschema import ArgSchema
from argschema.fields import InputDir, InputFile, Str, Int, Float, Dict, List

from .._shared.types import PathLike


class InputFileBasedLinux(InputFile):
    """

    InputFileBasedOS is a :class:`argschema.fields.InputFile`
    subclass which is a path to a file location which can be
    read by the user depending if it's on Linux or not.

    """

    def _validate(self, value: str):
        """
        Validates the filesystem

        Parameters
        -------------
        value: str
            Path where the file is located
        """
        if platform.system() != "Windows":
            super()._validate(value)


class MipParams(ArgSchema):
    """
    Parameters for creating MIP images
    """

    data_folder = InputDir(
        required=True, metadata={"description": "Path where the data is located"}
    )

    resolution = List(
        cls_or_instance=Float(),
        required=True, 
        metadata={"description": "The resolution in um for each axis ordered [DV, AP, ML]"}
    )

    axes = Dict(
        cls_or_instance=Str(),
        required=True, 
        metadata={"description": "Axes for images with key = zarr dimension value = plane"}
    )

    color_table = Dict(
        cls_or_instance=Int(),
        required=True, 
        metadata={"description": "Look-up table for filters to RGB channels"}
    )

    depth = Int(
        required=True, 
        metadata={"description": "Depth in um for the MIP"}
    )

    step = Int(
        required=True, 
        metadata={"description": "Distance in um between images"}
    )

    start_plane = Int(
        requires=True, 
        metadata={"description": "Plane to start setctioning from"}
    )

    input_data = InputDir(
        required=True,
        metadata={"description": "Path where MIP images will be stored"},
    )

    output_path = Str(
        required=True,
        metadata={"description": "Path where MIP images will be stored"},
    )



def get_yaml(yaml_path: PathLike):
    """
    Gets the default configuration from a YAML file

    Parameters
    --------------
    filename: str
        Path where the YAML is located

    Returns
    --------------
    dict
        Dictionary with the yaml configuration
    """

    config = None
    try:
        with open(yaml_path, "r") as stream:
            config = yaml.safe_load(stream)
    except Exception as error:
        raise error

    return config