"""Responsible for loading in data based on types"""

from dataclasses import dataclass

import numpy
import numpy as np
import re
from logging import getLogger
import abc
from enum import Enum
from numpy import load, save
from pathlib import Path

logger = getLogger(__name__)


@dataclass
class VoxelResolution:
    """
    Resolution dataclass
    """
    resolution: float


class LoadException(Exception):
    """
    Exceptions for loading
    """


@dataclass
class DataGeneric(abc.ABC):
    """
    Data type

    :param filepath: filepath for the DataGeneric
    :param label_index: An integer representation of the label of the given data.
    (I.E, 1, 2, or 3 for a three class problem)
    """
    filepath: Path
    label_index: int
    values: np.ndarray


@dataclass
class ThreeDimensionalLattice:
    """

    :param x_dim: integer value representing the dimension size of x
    :param y_dim: integer value representing the dimension size of y
    :param z_dim: integer value representing the dimension size of z

    """
    x_dim: int
    y_dim: int
    z_dim: int


@dataclass
class LatticeBounds:
    """
    Class to hold bounds of a dimension( i.e Cartesian)

    :param bound_neg: integer holding negative bound of dimension
    :param bound_pos: integer holding positive bound of dimension

    """

    bound_neg: float
    bound_pos: float


@dataclass
class VoxelData(DataGeneric):
    """
    Data type to hold Voxels

    :param resolution: holds type VoxelResolution which holds
                       a float representing the voxel's resolution
    :param x_bounds: bounds in x direction held in a LatticeBounds object
    :param y_bounds: bounds in y direction held in a LatticeBounds object
    :param z_bounds: bounds in z direction held in a LatticeBounds object
    :param dimensions: holds dimensions of voxel in ThreeDimensionalLattice object
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
        Writes data to a file of class type.

        :param data: Data to be written to file

        """
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def load(file: Path, label: int) -> DataGeneric:
        """
        Loads data into an object of type Data

        :param file: file to be loaded of type Path
        :param label: label for the given data. (What class does this data belong to).

        :return: Data that was loaded
        """
        raise NotImplementedError


class CNNFile(FileGeneric):
    """
    Responsible for writing and loading VoxelData. Object itself doesn't hold any data.
    """

    @staticmethod
    def write(data: VoxelData) -> None:
        """
            Method is responsible for generating cnn file based on
            VoxelData.values that's been sorted from largest to smallest.

            :param data: voxel data to be written into a CNN file.

            :return: None
            """
        with open(VoxelData.filepath.name, encoding="UTF-8" 'w') as cnn_file:
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
            for voxel_value in np.nditer(data.values.reshape(-1)):
                if voxel_value != 0:
                    values.append((cube_number, voxel_value))
                cube_number += 1
            values.sort(key=sort, reverse=True)
            cnn_file.write('NonZeroCubes: ' + str(len(values)) + '  TotalCubes: ' +
                           str(len(data.values.reshape(-1))) + '\n')
            for voxel in values:
                cnn_file.write(str(voxel[0]) + ' ' + str(voxel[1]) + ' \n')
            cnn_file.close()

    @staticmethod
    def load(file: Path, label: int) -> VoxelData:
        """
        Loads voxel data and associated information into a voxelData object and returns it

        :param file: file to be loaded of type Path
        :param label: label for the given data. (What class does this data belong to).

        :return: VoxelData
        """
        try:

            with file.open(encoding='gbk', errors='ignore') as voxel_file:
                lines = voxel_file.readlines()
                # Read dimensions from line 18
                match_dimensions = re.search(r'BOUNDS xyz dim: \[([0-9]+) ([0-9]+) ([0-9]+)]', lines[17])
                resolution_match = re.search(r'BOUNDS resolution: \[([0-9]+.[0-9]+)]', lines[18])
                match_x_bounds = re.search(r'BOUNDS xneg/xpos: \[-*([0-9]+.[0-9]+) -*([0-9]+.[0-9]+)]',
                                           lines[19])
                match_y_bounds = re.search(r'BOUNDS yneg/ypos: \[-*([0-9]+.[0-9]+) -*([0-9]+.[0-9]+)]',
                                           lines[20])
                match_z_bounds = re.search(r'BOUNDS zneg/zpos: \[-*([0-9]+.[0-9]+) -*([0-9]+.[0-9]+)]',
                                           lines[21])
                x_dim = int(match_dimensions.group(1))
                y_dim = int(match_dimensions.group(2))
                z_dim = int(match_dimensions.group(3))
                data = np.zeros((x_dim * y_dim * z_dim))
                # Reads voxel data starting at line 25
                for i in range(24, len(lines)):
                    line_num, val = lines[i].split()
                    val = float(val)
                    data[int(line_num)] = val
                data = data.reshape((x_dim, y_dim, z_dim, 1))
                return VoxelData(filepath=file.absolute(), label_index=label, values=data,
                                 resolution=VoxelResolution(float(resolution_match.group(1))),
                                 x_bounds=LatticeBounds(float(match_x_bounds.group(1)),
                                                        float(match_x_bounds.group(2))),
                                 y_bounds=LatticeBounds(float(match_y_bounds.group(1)),
                                                        float(match_y_bounds.group(2))),
                                 z_bounds=LatticeBounds(float(match_z_bounds.group(1)),
                                                        float(match_z_bounds.group(2))),
                                 dimensions=ThreeDimensionalLattice(x_dim, y_dim, z_dim)
                                 )
        except LoadException as ex:
            LoadException(f"Could not open the file because: {ex}.")


class NumpyFile(FileGeneric):
    """
    Responsible for writing and loading Numpy Files
    """

    @staticmethod
    def write(data: DataGeneric) -> None:
        """
        Method is responsible for generating NPY file based on
        DataGeneric.values.

        :param data:  data to be written into a NPY file.

        :return: None
        """
        save(file=data.filepath.name, arr=data.values)

    @staticmethod
    def load(file: Path, label: int) -> DataGeneric:
        """
        Loads information into a NPY object and returns it

        :param file: file to be loaded of type Path
        :param label: label for the given data. (What class does this data belong to).

        :return: DataGeneric
        """
        data: numpy.ndarray = load(file.name)
        return DataGeneric(file, label, data)
