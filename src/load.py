"""Responsible for loading in data based on types"""

from dataclasses import dataclass
import numpy as np
import re
from logging import getLogger
import abc

logger = getLogger(__name__)


class LoadException(Exception):
    """
    Exceptions for loading
    """


@dataclass
class DataGeneric(abc.ABC):
    """
    Data type
    """
    values: np.ndarray


@dataclass
class VoxelData(DataGeneric):
    """
    Data type to hold Voxels
    """

    x_bounds: []
    y_bounds: []
    z_bound: []
    x_dim_size: int
    y_dim_size: int
    z_dim_size: int


@dataclass
class FileGeneric(abc.ABC):
    """
    Object that holds filename and describes the type of file it is.
    """
    filename: str
    data: DataGeneric
    label_index: int

    @abc.abstractmethod
    def load(self) -> None:
        """
        Loads data into an object of type Data
        :return: Data
        """
        raise NotImplemented


class CNNFile(FileGeneric):
    """

    """

    def load(self) -> None:
        """
        Loads voxel data and associated information into a voxelData object and returns it
        :return: VoxelData
        """
        try:
            with open(self.filename, encoding='gbk', errors='ignore') as voxel_file:
                lines = voxel_file.readlines()
                # Read dimensions from line 18
                match = re.search(r'BOUNDS xyz dim: \[([0-9]+) ([0-9]+) ([0-9]+)]', lines[17])
                match_x_bounds = re.search(r'BOUNDS xneg/xpos: \[-*([0-9]+.[0-9]+) -*([0-9]+.[0-9]+)]', lines[19])
                match_y_bounds = re.search(r'BOUNDS yneg/ypos: \[-*([0-9]+.[0-9]+) -*([0-9]+.[0-9]+)]', lines[20])
                match_z_bounds = re.search(r'BOUNDS zneg/zpos: \[-*([0-9]+.[0-9]+) -*([0-9]+.[0-9]+)]', lines[21])
                x_dim = int(match.group(1))
                y_dim = int(match.group(2))
                z_dim = int(match.group(3))
                data = np.zeros((x_dim * y_dim * z_dim))
                # Reads voxel data starting at line 25
                for i in range(24, len(lines)):
                    line_num, val = lines[i].split()
                    line_num = int(line_num)
                    val = float(val)
                    data[line_num] = val
                data = data.reshape((x_dim, y_dim, z_dim, 1))
                self.data = VoxelData(data, match_x_bounds.groups(), match_y_bounds.groups(),
                                      match_z_bounds.groups(), x_dim, y_dim, z_dim)
        except LoadException as ex:
            LoadException(f"Could not open the file because: {ex}.")
