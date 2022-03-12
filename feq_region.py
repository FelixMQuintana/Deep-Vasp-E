from models.preprocessing import Preprocessing
import numpy as np
import functools
import random
import datetime

def freq_region_sample_generator(working_directory, labels_set, k, union, sample_num):
    """

    :param working_directory: e.g.'2016-CNNdata/2016-CNNdata/enolase/'
    :param labels_set: e.g. [['1mdr', '2ox4'], ['2pgw'], ['1iyx', '3otr', '1te6']]
    :param k: k is how many snapshots are needed to generate overlap,
           e.g. k = 50 means that select 50 snapshots from 600 images to generate overlap
    :param union: union is how many times to repeat overlap generation and then calculate union
           e.g. union is 500, that means repeat 500 times of overlap generation then calculate one union based on 500 overlaps
    :param sample_num: sample_num is the number of union image need to be generated
           e.g. sample_num is 600, means that 600 union
    :return:
    """

    x_dim, y_dim, z_dim = Preprocessing.voxel_parser(init_filename)[1:4]

    # generate random k(50) samples from 600 snapshots (available in each protein MD simulation) union times
    # [[12,3,5,6,77,...k],[],[]] union number
    k_inter = []
    for i in range(0, union):
        k_inter.append(random.sample(range(0, 600), k))

    image_num_index = 0
    # generate empty array: protein * sample_num (e.g. 6 * 600 =3600)
    #protein_length is number of protein included in the superfamily
    protein_length = len([protein_family for sublist in labels_set for protein_family in sublist])
    #sample_num is number of each protein
    images = np.empty((protein_length*sample_num, x_dim, y_dim, z_dim, 1))
    labels = np.empty(protein_length*sample_num)

    # subfamily is 3, inlcude ['1mdr', '2ox4', ], ['2pgw'], ['1iyx', '3otr', '1te6']
    for subfamily in labels_set:
        # protein is just one PDB code, e.g. 1te6, 1mdr
        for protein in subfamily:
            for rep in range(0, sample_num):
                # img_set is the set of 50 index of snapshot [123,43,45,21,46...]
                # test: img_set is one of 8 index sets, each set includes 5 index,  e.g. [1,3,5,8,6]
                k_inter_img = np.empty((union, x_dim, y_dim, z_dim, 1))
                l = 0
                for img_set in k_inter:
                    # k_inter_img is a empty array to save union(500) overlap
                    k_inter_img_single = np.empty((k, x_dim, y_dim, z_dim, 1))
                    s = 0
                    # each index in 50 randomly-select set(500 times)
                    # test: each index in 5 randomly-select set(8 times)
                    for img_num in img_set:
                        file_name = '{}{}/{}-{}-{}.cnn'.format(working_directory, protein, protein, '050',
                                                               str(img_num).rjust(3, '0'))
                        k_inter_img_single[s] = (Preprocessing.voxel_parser(file_name)[0])
                        s += 1

                    # calculate overlap for all selected snapshot,k_inter_img is used to save overlap
                    k_overlap = functools.reduce(lambda a, b: np.minimum(a, b), k_inter_img_single)
                    k_inter_img[l] = k_overlap
                    l += 1

                # generate union for overlap(each overlap is generate by 5 snapshot)
                k_union = functools.reduce(lambda a, b: np.maximum(a, b), k_inter_img)

                # save union in images
                images[image_num_index] = k_union
                labels[image_num_index] = labels_set.index(subfamily)
                image_num_index += 1

    # images dim = n_sample * n_voxel
    return images, labels



if __name__ == '__main__':

    ENOLASE_LABELS_SET = [['1mdr', '2ox4'], ['2pgw'], ['1iyx', '3otr', '1te6']]
    labels_set = ENOLASE_LABELS_SET
    working_directory = '2016-CNNdata/2016-CNNdata/enolase/'
    init_filename = '{}{}/{}-{}-{}.cnn'.format(working_directory, labels_set[0][0],labels_set[0][0], '050', '000')
    starttime = datetime.datetime.now()
    freq_region_sample_generator(working_directory, labels_set, 50, 500, 600)
    endtime = datetime.datetime.now()
    print(endtime-starttime)