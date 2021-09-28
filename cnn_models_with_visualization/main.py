"""
Module for main process to occur.
"""

import os
import argparse
import numpy as np
from time import time
from datetime import datetime
from sklearn.model_selection import KFold
from models.electrostatics_cnn_model import create_model
from random import sample
from processing.preprocessing import Preprocessing
from processing.postprocessing import PostProcessing
from tensorflow import config, device
from tensorflow.keras.models import save_model
from sklearn.metrics import classification_report, auc, roc_curve


ENOLASE_LABELS_SET = [['1mdr', '2ox4', ], ['2pgw'], ['1iyx', '3otr', '1te6']]
SERINE_PROTEASE_LABELS_SET = [['1b0e', '1elt'], ['1ex3'],
                              ['1a0j', '1ane', '1aq7', '1bzx', '1fn8', '1h4w', '1trn', '2eek', '2f91']]


def main(mode, super_family, working_directory, leave_out_index : int, res, generated_cnn_name, num_cubes, offset, class_index, layer_index,
         model_name, example_image):
    """
    Main method for running training flow and evaluation flow.
    :param mode: Mode to run either training or evaluation
    :param super_family: super family to use for training
    :param working_directory: working directory for desired dataset
    :param leave_out_index: index of protein starting at 0 that is left out
    :param res: resolution of data
    :param generated_cnn_name: generated cnn name
    :param num_cubes: number of cnn cubes
    :param offset: offset for cnn cubes
    :param class_index: class index for CAM
    :param layer_index: layer index of what to be visualized
    :param model_name: model name to be used for either saving or loading
    :param example_image: example image to be used for Grad CAM++
    """
    labels_set = []
    example_protein_name = ""
    if mode == "Train":
        if super_family == "enolase":
            labels_set = ENOLASE_LABELS_SET
            example_protein_name = '1mdr'
        elif super_family == "serProt":
            labels_set = SERINE_PROTEASE_LABELS_SET
            example_protein_name = '1b0e'
        flatten_protein_list = [protein_family for sublist in labels_set for protein_family in sublist]
        num_images = len(flatten_protein_list)*599
        pool = [format(image_num, '01') for image_num in range(600)]  # padded to 3 digits
        file_numbers = sample(pool, 600)
        file_name = '{}{}/{}NormalizedToElectrodynamics-0.cnn'.format(working_directory, example_protein_name,
                                                                     example_protein_name)
        x_dim, y_dim, z_dim = Preprocessing.voxel_parser(file_name)[1:4]

        images = np.empty((num_images, x_dim, y_dim, z_dim, 1))
        labels = np.empty(num_images)
        image_num_index = 0
        for category in labels_set:
            for name in category:
                for index in file_numbers:
                    file_name = '{}{}/{}NormalizedToElectrodynamics-{}.cnn'.format(working_directory, name, name,
                                                                                  index)
                    if os.path.isfile(file_name) is False:
                        continue
                    for family in range(3):
                        if name in labels_set[family]:
                            labels[image_num_index] = family
                            images[image_num_index] = Preprocessing.voxel_parser(file_name)[0]
                            image_num_index += 1
                            break
        training_time = time()
        acc_per_fold = []
        loss_per_fold = []
        final_test_images = images[leave_out_index * 599:(leave_out_index+1)*599]
        final_labels = labels[leave_out_index * 599:(leave_out_index+1) * 599]
        images_new = np.delete(images, [range(leave_out_index * 599, (leave_out_index+1) * 599)], axis=0)
        labels_new = np.delete(labels, [range(leave_out_index * 599, (leave_out_index+1) * 599)], axis=0)
        lowest_loss = None
        best_model = None
        with device("/gpu:0"):
            k_fold = KFold(n_splits=10, shuffle=True)
            no_fold = 1
            test_scores = open('Test_Score_' + flatten_protein_list[leave_out_index] + "_neg.txt", "w")
            for train, test in k_fold.split(images_new, labels_new):
                model = create_model(x_dim, y_dim, z_dim)
                model.fit(x=images[train], y=labels[train], batch_size=256, epochs=10, validation_split=.2, verbose=2)
                scores = model.evaluate(images[test], labels[test], verbose=2)

                test_scores.write('Score for fold {}, loss: {}, acc: {}\n'.format(no_fold, scores[0], scores[1] * 100))
                if lowest_loss is None:
                    lowest_loss = loss_per_fold
                    best_model = model
                elif lowest_loss > loss_per_fold:
                    lowest_loss = loss_per_fold
                    best_model = model
                acc_per_fold.append(scores[1] * 100)
                loss_per_fold.append(scores[0])
                no_fold += 1
            test_scores.close()
            raw_predictions = best_model.predict(final_test_images, batch_size=128)
            temp = np.array(np.argmax(raw_predictions, axis=1))
            file = open(
               str(datetime.today().strftime('%d-%b-%Y')) + "_confusionMatrix_" + flatten_protein_list[leave_out_index] + "_neg.txt", "w")
            file.write(classification_report(final_labels, temp, zero_division=0))
            fpr1, tpr1, thresholds1 = roc_curve(final_labels, temp, pos_label=2)
            file.write("AUC for dataset: " + str(auc(fpr1, tpr1)))
            file.close()
        save_model(best_model, model_name, save_format='h5')
        print("Total training time: ", time() - training_time)
        for i in range(0, len(acc_per_fold)):
            print('-----------------------------------------------------------------------------------------')
            print('Fold {} , loss {}, with accuracy {}'.format(i + 1, loss_per_fold[i], acc_per_fold[i]))
        print('-----------------------------------------------------------------------------------------')
        print('Average acc {}, average loss {}'.format(np.mean(acc_per_fold), np.mean(loss_per_fold)))
    elif mode == "Eval":
        processing = PostProcessing(model_name, example_image, class_index, layer_index, num_cubes, offset,
                                    generated_cnn_name, res)
        processing.generate_grad_cam()

########################################################################################################################


if __name__ == '__main__':
    gpus = config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                config.experimental.set_memory_growth(gpu, True)
            logical_gpus = config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as ex:
            # Memory growth must be set before GPUs have been initialized
            print(ex)
    # Parsing command line arguments
    parser = argparse.ArgumentParser(description='Classify 3D voxel images of protein families for two types of super '
                                                 'families, Enolases and SerProt.\n'
                                                 'SerProt: Trypsins '
                                                 '(1a0j, 1ane, 1aq7, 1bzx, 1fn8, 1h4w, 1trn, 2eek, 2f91), '
                                                 'and Elastases (1b0e, 1elt), and Chymotrypsins (1ex3). \n'
                                                 'Enolase: Enolases (1iyx, 3otr, 1te6),'
                                                 ' Mandelate Racemase (1mdr, 2ox4), and Muconate '
                                                 'Lactonizing Enzyme (2pgw). \n')
    parser.add_argument('-m', '--mode', type=str, default='Evaluate', help='What mode would you like to use. Input '
                        'Train for training a model. Otherwise input Eval for Grad CAM++ evalution')
    parser.add_argument('-p', '--proteinSuperFamily', default='enolase', help='(enolase or serProt)')
    parser.add_argument('-w', '--working_dir', default=None, help="please specify the working directory being used to "
                        "find the desired dataset aka enolase/ is my working directory if my dataset is in "
                        "enolase/1iyx/example001.cnn")
    parser.add_argument('-x', '--leave_out', type=int, default=0, help='please indicate index of which protein to '
                        'exclude from training and evaluate performance on.')
    parser.add_argument('-r', '--res', default='0.5', help='resolution of the images to be '
                                                           'trained/validated (0.5, 1, or 2).')
    parser.add_argument('-c', '--cnn', default="generated_cnn.CNN", help="Argument for Grad CAM++ analysis, "
                        "please specify the name of the generated cnn file")
    parser.add_argument('-n', '--num_cubes', default=None, help="Argument for Grad CAM++ analysis, please specify the "
                        "number of cubes interested for grad CAM++ analysis. If none is entered analysis will grab "
                        "the 10% of high scoring cubes")
    parser.add_argument('-o', '--offset', type=int, default=0, help='Argument for Grad CAM++ analysis, please specify '
                        'the offset desired for the desired number of cubes respected to the highest scoring cubes.'
                        ' For ex: num_cubes = 100, with offset of 50 will grab the top 50 to 150 cubes for a'
                        ' given class_index.')
    parser.add_argument('-C', '--class_index', type=int, default=0, help='Argument for Grad CAM++ analysis, '
                        'please specify the class index used for generating Grad Cam++')
    parser.add_argument('-l', '--layer_index', default=None, help="Argument for Grad CAM++ analysis, please specify the"
                        " layer index desired to be used for Grad CAM++ analysis. This layer will be used to calculate "
                        "first, second, and third order gradients")
    parser.add_argument('-M', '--model', default=None, help="please specify model to used for "
                        "Grad CAM++ or name of file to be saved from training.File is expected to be of .h5. "
                        "TODO: Add error handling for wrong tensorflow file type")
    parser.add_argument('-e', '--example', help='Argument for Grad CAM++ analysis, please specify example image to be '
                        'used for analysis.')
    args = parser.parse_args()
    parser.print_usage()

    main(args.mode, args.proteinSuperFamily, args.working_dir, args.leave_out, args.res, args.cnn, args.num_cubes, args.offset,
         args.class_index, args.layer_index, args.model, args.example)
