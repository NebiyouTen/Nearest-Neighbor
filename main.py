import os
import sys
import argparse
import numpy as np
import time
import itertools

from knn import KNN

def generate_combinations(N, k):
    '''
        A function for N choose k
    '''
    elements = list(range(N))
    combinations = list(itertools.combinations(elements, k))
    return combinations

def forward_search(feats, labels):
    total_features = feats.shape[1]

    for num_feats in range(total_features):
        num_feats += 1

        feature_sets = generate_combinations(total_features, num_feats)

        for feature_set in feature_sets:
            selected_feat = feats[:, feature_set]

            knn = KNN(selected_feat, labels)
            print(f"KKK {feature_set}", knn.leave_one_out_val())


def main(args):
    '''
        Main function
        arguments:
            args: Command line flags.
    '''
    if args.debug:
        print("Args: ", args)

    data = np.loadtxt(args.test_file_path)
    class_lables, features = data[:,0], data[:,1:]

    print("data ", features.shape, class_lables.shape)

    knn = KNN(k = args.k, data = features, labels = class_lables)
    knn.normalize_data()
    print("data ", knn.leave_one_out_val())

    forward_search(features, class_lables)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", default = 'forward_selection',
            choices = ['forward_selection', 'backward_elimination'],
            help="The search algorithm to use")
    parser.add_argument("--test_file_path", default = "data/CS205_large_Data__1.txt",
            type = str,
            help="TXT file to use for testing. E.g. data/eamonns_test_2.txt")
    parser.add_argument("--debug", action = 'store_true',
            help="Flag if enabled adds debug information")
    parser.add_argument("--k", default = 1,
            help="K value for the nearest-neighbor (NN) algorithm")

    args = parser.parse_args()

    main(args)
