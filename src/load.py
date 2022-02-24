"""Responsible for loading in data based on types"""

from dataclasses import dataclass
import numpy as np
import re
from logging import getLogger
import abc
from enum import IntEnum
from pathlib import PurePath

logger = getLogger(__name__)


@dataclass
class VoxelResolution(IntEnum):
    """
    Resolution dataclass
    """
    resolution: int


class LoadException(Exception):
    """
    Exceptions for loading
    """


@dataclass
class DataGeneric(abc.ABC):
    """
    Data type
    """
    filepath: PurePath
    label_index: int
    values: np.ndarray


@dataclass
class ThreeDimensionalLattice:
    x_dim: int
    y_dim: int
    z_dim: int


@dataclass
class LatticeBounds:
    bound_neg: int
    bound_pos: int


@dataclass
class VoxelData(DataGeneric):
    """
    Data type to hold Voxels
    """
    resolution: VoxelResolution
    x_bounds: LatticeBounds
    y_bounds: LatticeBounds
    z_bounds: LatticeBounds
    dimensions: ThreeDimensionalLattice


def sort(array) -> bool:
    """
    Sorting pattern
    :param array: value
    :return: Sorted array
    """
    return array[1]


@dataclass
class FileGeneric(abc.ABC):
    """
    Object that holds filename and describes the type of file it is.
    """

    @staticmethod
    @abc.abstractmethod
    def write(data: DataGeneric) -> None:
        """

        """
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def load(filename: str) -> DataGeneric:
        """
        Loads data into an object of type Data
        :return: Data
        """
        raise NotImplementedError


class CNNFile(FileGeneric):
    """

    """

    @staticmethod
    def write(data: VoxelData) -> None:
        """
            Method is responsible for generating cnn file based on 3D numpy array data (raw_cnn_data) that has been sorted
            from largest values to smallest. A threshold can be specified for top values of given ranged.
            :return: None
            """
        with open(VoxelData.filepath.name, 'w') as cnn_file:
            cnn_file.write('#######################################'
                           '###########################################################\n')
            cnn_file.write('##                                     '
                           '                                                         ##\n')
            cnn_file.write('##   #     #    #     #####  ######    '
                           '  #####                   ######                         ##\n')
            cnn_file.write('##   #     #   # #   #     # #     #   '
                           ' #     # #    # #    #    #     #   ##   #####   ##      ##\n')
            cnn_file.write('##   #     #  #   #  #       #     #   '
                           ' #       ##   # ##   #    #     #  #  #    #    #  #     ##\n')
            cnn_file.write('##   #     # #     #  #####  ######    '
                           ' #       # #  # # #  #    #     # #    #   #   #    #    ##\n')
            cnn_file.write('##    #   #  #######       # #         '
                           ' #       #  # # #  # #    #     # ######   #   ######    ##\n')
            cnn_file.write('##     # #   #     # #     # #         '
                           ' #     # #   ## #   ##    #     # #    #   #   #    #    ##\n')
            cnn_file.write('##      #    #     #  #####  #         '
                           '  #####  #    # #    #    ######  #    #   #   #    #    ##\n')
            cnn_file.write('##                                     '
                           '                                                         ##\n')
            cnn_file.write('#######################################'
                           '###########################################################\n')
            cnn_file.write('\n')
            cnn_file.write('#\n')
            cnn_file.write('# Note: cube indices run from 0 to TotalCubes-1 '
                           'as stated below. Missing cubes have zero volume.\n')
            cnn_file.write('#\n')
            cnn_file.write('\n')
            cnn_file.write('BOUNDS xyz dim:[' + str(data.dimensions.x_dim) + ' '
                           + str(data.dimensions.y_dim) + ' ' + str(data.dimensions.z_dim) + ']\n')
            cnn_file.write('BOUNDS resolution: [' + str(data.resolution) + ']\n')
            cnn_file.write('BOUNDS xneg/xpos: [' + str(data.x_bounds.bound_neg) + ' '
                           + str(data.x_bounds.bound_pos) + ']\n')
            cnn_file.write('BOUNDS yneg/ypos: [' + str(data.y_bounds.bound_neg) + ' '
                           + str(data.y_bounds.bound_pos) + ']\n')
            cnn_file.write('BOUNDS zneg/zpos: [' + str(data.z_bounds.bound_neg) + ' '
                           + str(data.z_bounds.bound_pos) + ']\n')
            cnn_file.write('\n')
            cube_number = 0
            values = []                             
            for x in np.nditer(data.values.reshape(-1)):
                if x != 0:
                    values.append((cube_number, x))
                cube_number += 1
            values.sort(key=sort, reverse=True)
            cnn_file.write('NonZeroCubes: ' + str(len(values)) + '  TotalCubes: ' +
                           str(len(data.values.reshape(-1))) + '\n')
            for i in range(0, len(values)):
                cnn_file.write(str(values[i][0]) + ' ' + str(values[i][1]) + ' \n')
            cnn_file.close()

    @staticmethod
    def load(filepath: str) -> VoxelData:
        """
        Loads voxel data and associated information into a voxelData object and returns it
        :return: VoxelData
        """
        try:
            path_obj = PurePath(filepath)
            with open(path_obj.name, encoding='gbk', errors='ignore') as voxel_file:
                lines = voxel_file.readlines()
                # Read dimensions from line 18
                match = re.search(r'BOUNDS xyz dim: \[([0-9]+) ([0-9]+) ([0-9]+)]', lines[17])
                match_x_bounds = re.search(r'BOUNDS xneg/xpos: \[-*([0-9]+.[0-9]+) -*([0-9]+.[0-9]+)]',
                                           lines[19])
                x_neg, x_pos = match_x_bounds.groups()
                match_y_bounds = re.search(r'BOUNDS yneg/ypos: \[-*([0-9]+.[0-9]+) -*([0-9]+.[0-9]+)]',
                                           lines[20])
                y_neg, y_pos = match_y_bounds.groups()
                match_z_bounds = re.search(r'BOUNDS zneg/zpos: \[-*([0-9]+.[0-9]+) -*([0-9]+.[0-9]+)]',
                                           lines[21])
                z_neg, z_pos = match_z_bounds.groups()

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
                return VoxelData(filepath=path_obj, label_index=int(path_obj.parent.name), values=data,
                                 resolution=VoxelResolution(int(path_obj.parent.parent.as_posix())),
                                 x_bounds=LatticeBounds(x_neg, x_pos), y_bounds=LatticeBounds(y_neg, y_neg),
                                 z_bounds=LatticeBounds(z_neg, z_pos),
                                 dimensions=ThreeDimensionalLattice(x_dim, y_dim, z_dim))
        except LoadException as ex:
            LoadException(f"Could not open the file because: {ex}.")
