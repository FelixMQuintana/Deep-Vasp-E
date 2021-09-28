"""Module for preprocessing files"""

import re
import numpy as np


class Preprocessing:
    """
    Preprocessing class for processing CNN files.
    """

    @staticmethod
    def voxel_parser(filename: str) -> (np.ndarray, int, int, int, list, list, list):
        """
        Read the data in .cnn protein image files into a 3D-array
        :return: 3D-array
        """
        with open(filename, encoding='gbk', errors='ignore') as voxel_file:
            lines = voxel_file.readlines()
            # Reads dimensions from line 18
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
            return data, x_dim, y_dim, z_dim, match_x_bounds.groups(), match_y_bounds.groups(), match_z_bounds.groups()
