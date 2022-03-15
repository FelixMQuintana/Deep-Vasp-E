from processing.preprocessing import Preprocessing
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
    init_filename = '{}{}/{}-{}-{}.cnn'.format(working_directory, labels_set[0][0], labels_set[0][0], '050', '000')
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



def gen_overlaps(working_directory, labels_set, overlap_num = 1000, k = 50):

    """
    #################################################################################
    #      generate 1000 overlaps for each protein based on 600 snapshots           #
    #################################################################################
    :param working_directory:
    :param labels_set: enolase or serine protease
    :param overlap_num: generate how many overlaps for each protein
    :param k: how many snapshots used to generate overlaps
    :return:
    """

    starttime = datetime.datetime.now()

    init_filename = '{}{}/{}-{}-{}.cnn'.format(working_directory, labels_set[0][0], labels_set[0][0], '050', '000')
    x_dim, y_dim, z_dim = Preprocessing.voxel_parser(init_filename)[1:4]

    # generate random index of "k"(50) samples from 600 snapshots (protein MD simulation), repeat "overlap_num"(1000) times
    overlap_sets = []
    for i in range(0, overlap_num):
        overlap_sets.append(random.sample(range(0, 600), k))

    # subfamily is ['1mdr', '2ox4', ] or ['2pgw'] or ['1iyx', '3otr', '1te6']
    for subfamily in labels_set:

        # protein is each PDB code, e.g. 1te6, 1mdr
        for protein in subfamily:
            l = 0

            #overlap_sets is a 1000-length empty list
            #each element(one_set_overlap) in overlap_sets is a 50-length array with index inside
            for one_set_overlap in overlap_sets:

                # one_overlap is an empty 50-length array with shape "x_dim, y_dim, z_dim, 1"
                one_overlap = np.empty((k, x_dim, y_dim, z_dim, 1))
                o = 0

                # img_index is each index of cnn file. e.g 001, or 020 ....
                for img_index in one_set_overlap:

                    # parser each cnn file and save it into one_overlap
                    file_name = '{}{}/{}-{}-{}.cnn'.format(working_directory, protein, protein, '050',
                                                           str(img_index).rjust(3, '0'))
                    one_overlap[o] = (Preprocessing.voxel_parser(file_name)[0])
                    o += 1

                # k_overlap: calculate one overlap based on all selected k snapshots
                k_overlap = functools.reduce(lambda a, b: np.minimum(a, b), one_overlap)

                # save each overlap array as npy
                np.save("{}{}_overlap/{}-{}-{}".format(working_directory, protein,
                                               "overlap", '050', str(l).rjust(3, '0')), k_overlap)
                l += 1

    endtime = datetime.datetime.now()
    # time 0:15:32.571845
    print(endtime - starttime)



def gen_unions(working_directory, labels_set, union_num =600, overlap_num =500):

    """
    ####################################################################################
    #     generate 600 unions for each protein based on 1000 overlaps                  #
    ####################################################################################
    :param working_directory:
    :param labels_set: enolase or serine protease
    :param union_num: generate how many unions for each protein
    :param overlap_num: how many overlaps used to generate one union
    :return:
    """
    starttime = datetime.datetime.now()
    init_filename = '{}{}/{}-{}-{}.cnn'.format(working_directory, labels_set[0][0],
                                               labels_set[0][0], '050', '000')
    x_dim, y_dim, z_dim = Preprocessing.voxel_parser(init_filename)[1:4]

    #union_sets is a 600-length list,
    #each element in union_sets is one 500-length list with index inside
    union_sets = []
    for i in range(0, union_num):
        union_sets.append(random.sample(range(0, 1000), overlap_num))

    # subfamily is 3, includes ['1mdr', '2ox4', ], ['2pgw'], ['1iyx', '3otr', '1te6']
    for subfamily in labels_set:

        # protein is just one PDB code, e.g. 1te6, 1mdr
        for protein in subfamily:
                l = 0

                #one_set_union is each union generated by 500 overlaps
                for one_set_union in union_sets:

                    #one_union is an empty array to save 500 overlaps, which is used to generate 1 union
                    one_union = np.empty((overlap_num, x_dim, y_dim, z_dim, 1))
                    u = 0

                    #overlap_index is each index in 500 overlaps.e.g. overlap-050-541, 541 is overlap_index
                    for overlap_index in one_set_union:
                        file_name = '{}{}_overlap/{}-{}-{}.npy'.format(working_directory, protein, "overlap", '050',
                                                               str(overlap_index).rjust(3, '0'))
                        one_union[u] = np.load(file_name)
                        u += 1

                    # calculate union based on selected 500 overlaps
                    k_union = functools.reduce(lambda a, b: np.maximum(a, b), one_union)
                    np.save("{}{}_union/{}-{}-{}".format(working_directory, protein,
                                                           "union", '050', str(l).rjust(3, '0')), k_union)
                    l+=1

    endtime = datetime.datetime.now()
    #0:16:39.153374
    print(endtime - starttime)


if __name__ == '__main__':

    ENOLASE_LABELS_SET = [['1mdr', '2ox4'], ['2pgw'], ['1iyx', '3otr', '1te6']]

    labels_set = ENOLASE_LABELS_SET
    working_directory = '2016-CNNdata/2016-CNNdata/enolase/'

    gen_overlaps(working_directory, labels_set)
    gen_unions(working_directory, labels_set)

