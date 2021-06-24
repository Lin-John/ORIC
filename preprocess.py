import os
import math
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from collections import Counter


class xLabelEncoder(object):
    """Label encode the features.
    All the values appearing less than a given threshold in training set will be labeled by 0.

    Attributes:
        thresh: the threshold of occurrences.
        count: a dict recording the occurrences of each value.
        encode: a dict whose key is the original value and val is the corresponding code.
        decode: a dict whose key is the code and value is the corresponding value.
    """

    def __init__(self, thresh=1):
        """Init class with the threshold."""
        self.thresh = thresh
        self.count = {}
        self.encode = {}
        self.decode = {}

    def fit(self, X):
        """Fit a transformer on the records."""
        count = Counter(X)
        for val in count:
            if val not in self.count:
                self.count[val] = count[val]
            else:
                self.count[val] += count[val]
            if self.count[val] >= self.thresh and val not in self.encode:
                label = len(self.encode) + 1
                self.encode[val] = label
                self.decode[label] = val

    def transform(self, X):
        """Encode the the encoded feature."""
        return np.vectorize(lambda x: self.encode.get(x, 0))(X)

    def fit_transform(self, X):
        """fit a transformer on the records and
        return the result after encoding.
        """
        self.fit(X)
        return self.transform(X)

    def reverse(self, i):
        """return the original value of the corresponding code."""
        return self.decode[i]

    def reset(self):
        """Reset the encoder."""
        self.count = {}
        self.encode = {}
        self.decode = {}



def csv_data_generator(fp, dataset="avazu", names=None, batch_size=1000000):
    assert dataset in ["avazu", "criteo"], "invalid data set!"
    i = 0
    while True:
        if dataset == "avazu":
            data_batch = pd.read_csv(fp, sep=',', skiprows=range(1, i*batch_size+1), nrows=batch_size)
        elif dataset == "criteo":
            data_batch = pd.read_csv(fp, sep='\t', skiprows=i*batch_size, nrows=batch_size, names=names)
        if len(data_batch):
            yield data_batch
        else:
            break
        i += 1


def load_csv(dataset, dataset_folder):
    assert dataset in ["avazu", "criteo"], "invalid data set!"
    if dataset == "avazu":
        fp = os.path.join(dataset_folder, "train")
        target = ["click"]
        names = list(pd.read_csv(fp, sep=',', nrows=0).columns)
        dense_features = []
        sparse_features = [f for f in names if f != "click"]
        data = csv_data_generator(fp, dataset)
        return data, target, dense_features, sparse_features
    elif dataset == "criteo":
        fp = os.path.join(dataset_folder, "train.txt")
        target = ['click']
        dense_features = ['I'+str(i) for i in range(1, 14)]
        sparse_features = ['C'+str(i) for i in range(1, 27)]
        names = target + dense_features + sparse_features
        data = csv_data_generator(fp, dataset, names)
        return data, target, dense_features, sparse_features


def process(data, lbe, sparse_features, dense_features):
    for f in sparse_features:
        data[f] = lbe[f].fit_transform(data[f]).astype('int32')
    for f in dense_features:
        data[f] = np.vectorize(lambda x: int(math.log(x)**2) if x>2 else x)(data[f])
    return data


def save_feat_info(feat_info, feat_info_path):
    with open(feat_info_path, "wb") as handle:
        pickle.dump(feat_info, handle, protocol=pickle.HIGHEST_PROTOCOL)


def split_data_ordered(data, data_folder, n_splits=10):
    kf = KFold(n_splits=n_splits)

    for i, (_, idx) in enumerate(kf.split(data)):
        part_folder = os.path.join(data_folder, "part{}".format(i + 1))
        if not os.path.exists(part_folder):
            os.makedirs(part_folder)
        data.iloc[idx].to_pickle(os.path.join(part_folder, "base_data.pkl"))
        print("save part {} done".format(i + 1))