import itertools
from knn import KNN

def generate_combinations(N, k):
    '''
        A function for N choose k
    '''
    elements = list(range(N))
    combinations = list(itertools.combinations(elements, k))

    return combinations

def forward_search(feats, labels, args):
    meta_data = {}
    total_features = feats.shape[1]

    features_idx = list(range(total_features))
    best_feats = set()
    num_feats = 1

    final_best_val, final_best_feat = -1, None

    while len(features_idx) > 0:

        num_feats += 1
        best_val = -1
        beat_feat = None


        for feat_idx in features_idx:
            candidate_feats = best_feats.copy()
            candidate_feats.add(feat_idx)
            selected_feat = feats[:, tuple(candidate_feats)]

            knn = KNN(selected_feat, labels, normalize = args.normalize_data, norm_type = args.norm_type)

            val_acc = knn.leave_one_out_val()
            if val_acc > best_val:
                 best_val = val_acc
                 beat_feat = feat_idx
                 print(f"\t\tFound better val:{best_val}, feature_set: {candidate_feats}" )
        # Add best feature to best_feats and pop from features_idx
        features_idx.remove(beat_feat)
        best_feats.add(beat_feat)

        meta_data[frozenset(best_feats)] = best_val

        if best_val > final_best_val:
             final_best_val = best_val
             final_best_feat = tuple(best_feats)
        else:
            print(f"\t!!!Warning. Accuray dropped. Best so far={final_best_val}, new={best_val}. Continuing search ... ")

        print("\tDone one forward step: Remaining features: ", len(features_idx)
                , " Best features: ", best_feats
                , f" with acc={best_val}")

    print("*"*15, "Final best val ", final_best_val, final_best_feat)

    return final_best_val, final_best_feat, meta_data


def backward_search(feats, labels, args):
    meta_data = {}
    total_features = feats.shape[1]

    features_idx = list(range(total_features))
    best_feats = set(features_idx)
    num_feats = 1

    knn = KNN(feats, labels, normalize = args.normalize_data, norm_type = args.norm_type)
    val_acc = knn.leave_one_out_val()

    final_best_val, final_best_feat = val_acc, best_feats

    while len(features_idx) > 0:

        num_feats += 1
        best_val = -1
        beat_feat = None

        for feat_idx in features_idx:
            candidate_feats = best_feats.copy()
            candidate_feats.remove(feat_idx)
            selected_feat = feats[:, tuple(candidate_feats)]

            knn = KNN(selected_feat, labels, normalize = args.normalize_data, norm_type = args.norm_type)

            val_acc = knn.leave_one_out_val()
            if val_acc > best_val:
                 best_val = val_acc
                 beat_feat = feat_idx
                 print(f"\t\tFound better val:{best_val}, feature_set: {candidate_feats}" )
        # Add best feature to best_feats and pop from features_idx
        features_idx.remove(beat_feat)
        best_feats.remove(beat_feat)

        meta_data[frozenset(best_feats)] = best_val

        if best_val > final_best_val:
             final_best_val = best_val
             final_best_feat = tuple(best_feats)
        else:
            print(f"\t!!!Warning. Accuray dropped. Best so far={final_best_val}, new={best_val}. Continuing search ... ")

        print("\tDone one forward step: Remaining features: ", len(features_idx)
                , " Best features: ", best_feats
                , f" with acc={best_val}")

    print("*"*15, "Final best val ", final_best_val, final_best_feat)

    return final_best_val, final_best_feat, meta_data
