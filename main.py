import os
import sys
import argparse
import numpy as np
import time
import pickle

from knn import KNN
from search import forward_search, backward_search

def get_output_name(args):
    test_data_filename = os.path.basename(args.test_data)

    return f"{args.algorithm}_{test_data_filename}_{args.normalize_data}_{args.norm_type}.pk"

def main(args):
    '''
        Main function
        arguments:
            args: Command line flags.
    '''
    if args.debug:
        print("Args: ", args)

    data = np.loadtxt(args.test_data)
    class_lables, features = data[:,0], data[:,1:]

    print("data ", features.shape, class_lables.shape)

    knn = KNN(k = args.k
        , data = features
        , labels = class_lables
        , normalize = args.normalize_data
        , norm_type = args.norm_type)

    print("Leave one out validation accuray with using all the features: ", knn.leave_one_out_val())

    if args.algorithm == 'forward_selection':
        acc, feats, meta = forward_search(features, class_lables, args)
    else:
        acc, feats, meta = backward_search(features, class_lables, args)

    exp_log = {
        "args": args,
        "meta": meta
    }

    with open(f'outputs/{get_output_name(args)}', 'wb') as pickle_file:
        pickle.dump(exp_log, pickle_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", default = 'forward_selection',
            choices = ['forward_selection', 'backward_elimination'],
            help="The search algorithm to use")
    parser.add_argument("--test_data", default = "./data/CS205_small_Data__25.txt",
            type = str,
            help="TXT file to use for testing. E.g. data/CS205_small_Data__25.txt (large 3)")
    parser.add_argument("--debug", action = 'store_true',
            help="Flag if enabled adds debug information")
    parser.add_argument("--normalize_data", action = 'store_true',
            help="Flag if enabled normalizes data")
    parser.add_argument("--norm_type", default = 'min_max',
            choices = ['min_max', 'z_norm'],
            help="The data normalilzation method to use")
    parser.add_argument("--k", default = 1,
            help="K value for the nearest-neighbor (NN) algorithm")

    args = parser.parse_args()

    main(args)
