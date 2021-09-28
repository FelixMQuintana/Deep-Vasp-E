"""Module for postprocessing results for Analysis"""
import numpy as np
from keras.backend import exp
from tensorflow import GradientTape
from tensorflow.keras.models import load_model, Model
from processing.preprocessing import Preprocessing


def sort(array) -> bool:
    """
    Sorting pattern
    :param array: value
    :return: Sorted array
    """
    return array[1]


class PostProcessing:
    """
    Class for postprocessing results from CNN.
    """

    def __init__(self, model: str = "./pocket_classification_model_serProt_without_1trn.h5",
                 example_image: str = "serProt/1trn/1trnNormalizedToElectrodynamics10.cnn",
                 class_index: int = 1, conv_layer_index: int = 1, number_of_cubes: int = None,
                 lower_limit: int = 0, generated_cnn_name: str = "generated_cnn.CNN",
                 resolution: str = '0.500000') -> None:
        """
        Initializes object with needed values to create a CNN file.
        :param model: name of model used for analysis
        :param example_image:image used for analysis
        :param class_index: class index to be used for CAM
        :param conv_layer_index: convolution layer index of interest, index is respected to model layers not index of
        convolutional layers in model
        :param number_of_cubes:Number of cubes to be generated in CAM
        :param lower_limit:Offset from greatest values for captured cubes
        :param generated_cnn_name: name of CNN file generated from CAM
        :param resolution: resolution for CNN file to be generated
        """
        self._example_image, self._x_dim, self._y_dim, self._z_dim, x_bounds, y_bounds, z_bounds = \
            Preprocessing.voxel_parser(example_image)
        self._model_name = model
        self._class_index = class_index
        self._conv_layer_index = conv_layer_index
        self._num_of_cubes = number_of_cubes
        self._lower_limit = lower_limit
        self._generated_cnn_name = generated_cnn_name
        self._xneg = x_bounds[0]
        self._xpos = x_bounds[1]
        self._yneg = y_bounds[0]
        self._ypos = y_bounds[1]
        self._zneg = z_bounds[0]
        self._zpos = z_bounds[1]
        self._res = resolution
        self._raw_cnn_data = None

    def generate_grad_cam(self) -> None:
        """

        """
        model = load_model(self._model_name)
        grad_model = Model([model.inputs], [model.layers[self._conv_layer_index].output, model.output])
        example_image_data, x_dim, y_dim, z_dim, x_bounds, y_bounds, z_bounds = Preprocessing.voxel_parser(
            self._example_image)
        image = np.empty((1, x_dim, y_dim, z_dim, 1))
        image[0] = example_image_data
        with GradientTape() as tape:
            conv_outputs, predictions = grad_model(image)
            loss = predictions[:, self._class_index]
        outputs = conv_outputs[0]
        grads = tape.gradient(loss, conv_outputs)[0]
        first = exp(loss) * grads
        second = exp(loss) * grads * grads
        third = exp(loss) * grads * grads * grads
        global_sum = np.sum(outputs, axis=(0, 1, 2))
        alpha_num = second
        alpha_denominator = second * 2 + third * global_sum + 1e-10
        alpha = alpha_num / alpha_denominator
        weights = np.sum(alpha * np.maximum(first, 0.0), axis=(0, 1, 2))
        self._raw_cnn_data = np.zeros(shape=(x_dim, y_dim, z_dim))
        for index, w in enumerate(weights):
            self._raw_cnn_data += w * outputs[:, :, :, index]
        self._create_cnn_file()

    def _create_cnn_file(self) -> None:
        """
        Method is responsible for generating cnn file based on 3D numpy array data (raw_cnn_data) that has been sorted
        from largest values to smallest. A threshold can be specified for top values of given ranged.
        :return: None
        """
        with open(self._generated_cnn_name, 'w') as cnn_file:
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
            cnn_file.write('# This data was generated from postprocessing '
                           'script from results of CNN model:' + self._model_name + '\n')
            cnn_file.write('# Note: cube indices run from 0 to TotalCubes-1 '
                           'as stated below. Missing cubes have zero volume.\n')
            cnn_file.write('#\n')
            cnn_file.write('\n')
            cnn_file.write('BOUNDS xyz dim:[' + str(self._x_dim) + ' '
                           + str(self._y_dim) + ' ' + str(self._z_dim) + ']\n')
            cnn_file.write('BOUNDS resolution: [' + self._res + ']\n')
            cnn_file.write('BOUNDS xneg/xpos: [' + self._xneg + ' ' + self._xpos + ']\n')
            cnn_file.write('BOUNDS yneg/ypos: [' + self._yneg + ' ' + self._ypos + ']\n')
            cnn_file.write('BOUNDS zneg/zpos: [' + self._zneg + ' ' + self._zpos + ']\n')
            cnn_file.write('\n')
            cube_number = 0
            values = []
            for x in np.nditer(self._raw_cnn_data.reshape(-1)):
                values.append((cube_number, x))
                cube_number += 1
            values.sort(key=sort, reverse=True)
            if self._num_of_cubes is None:
                self._num_of_cubes = int(len(values) * 0.1)
            cnn_file.write('NonZeroCubes: ' + str(self._num_of_cubes) + '  TotalCubes: ' + str(len(values)) + '\n')
            for i in range(self._lower_limit, self._num_of_cubes+self._lower_limit+1):
                cnn_file.write(str(values[i][0]) + ' ' + str(values[i][1]) + ' \n')
            cnn_file.close()
