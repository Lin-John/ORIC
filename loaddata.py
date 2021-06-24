import pickle
import numpy as np
import pandas as pd
from os.path import join, exists


def load_feat_info(fp):
    with open(fp, "rb") as handle:
        feat_info = pickle.load(handle)
    return (feat_info['target'],
            feat_info['dense_feat'],
            feat_info['sparse_feat'],
            feat_info['sequence_feat'],
            feat_info['nunique_feat'])


def load_basic_data(data_folder, file_id, file_name="base_data.pkl"):
    feat_info_path = join(data_folder, "feat_info.pkl")
    target, dense_feat, sparse_feat, _, _ = load_feat_info(feat_info_path)

    part_folder = join(data_folder,
        "part{}".format(file_id))
    data = pd.read_pickle(join(part_folder, file_name))
    return data[dense_feat+sparse_feat], data[target].values.ravel()


def load_seqence_data(data_folder, file_id):
    part_folder = join(data_folder,
                       "part{}".format(file_id))
    seq_data_path = join(part_folder, "seq_data.pkl")
    if exists(seq_data_path):
        with open(seq_data_path, "rb") as handle:
            seq_data = pickle.load(handle)
    else:
        seq_data = {}
    return seq_data
