import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

from preprocess import xLabelEncoder, load_csv, process, save_feat_info, split_data_ordered


if __name__ == "__main__":
    # preprocess Avazu and Criteo
    for dataset in ["criteo", "avazu"]:
        data_folder = os.path.join("./data", dataset)
        fp_clean = os.path.join(data_folder, "train_clean.txt")
        if dataset == "criteo":
            k = 10
        elif dataset == "avazu":
            k = 5

        data, target, dense_features, sparse_features = load_csv(dataset, data_folder)
        names = target + dense_features + sparse_features
        with open(fp_clean, "w") as f:
            f.write(','.join(names) + '\n')
        lbe = {f: xLabelEncoder(k) for f in sparse_features}
        for data_batch in data:
            if dataset == "criteo":
                data_batch[sparse_features] = data_batch[sparse_features].fillna('Missing', )
                data_batch[dense_features] = data_batch[dense_features].fillna(-1, )
                data_batch[target] = data_batch[target].fillna(0, )
            data_batch = process(data_batch, lbe, sparse_features, dense_features)
            data_batch.to_csv(fp_clean, columns=names, mode='a+', index=False, header=False, )
        nunique_feat = {f: len(lbe[f].encode) for f in lbe}
        feat_info = {"target": target,
                     "dense_feat": dense_features,
                     "sparse_feat": sparse_features,
                     "sequence_feat": [],
                     "nunique_feat": nunique_feat}
        feat_info_path = os.path.join(data_folder, "feat_info.pkl")
        save_feat_info(feat_info, feat_info_path)

        data_clean = pd.read_csv(fp_clean, dtype=np.int32)
        split_data_ordered(data_clean, data_folder)

    # preprocess Taobao
    data_folder = os.path.join("data", "taobao")

    ad = pd.read_csv('data/taobao/ad_feature.csv')
    ad['brand'] = ad['brand'].fillna(-1)
    for f in ["cate_id", "brand"]:
        lbe = LabelEncoder()
        ad[f] = lbe.fit_transform(ad[f]) + 1

    user = pd.read_csv('data/taobao/user_profile.csv')
    user = user.fillna(-1)
    user.rename(columns={'new_user_class_level ': 'new_user_class_level'}, inplace=True)

    sample = pd.read_csv('data/taobao/raw_sample.csv')
    sample.rename(columns={'user': 'userid'}, inplace=True)

    data = pd.merge(sample, user, how='left', on='userid', )
    data = pd.merge(data, ad, how='left', on='adgroup_id')

    sparse_features = ['userid', 'adgroup_id', 'pid', 'cms_segid', 'cms_group_id', 'final_gender_code', 'age_level',
                       'pvalue_level', 'shopping_level', 'occupation', 'new_user_class_level', 'campaign_id',
                       'customer']
    dense_features = ['price']
    target = ["clk"]
    nunique_feat = dict(data.nunique())
    feat_info = {"target": target,
                 "dense_feat": dense_features,
                 "sparse_feat": sparse_features,
                 "sequence_feat": [],
                 "nunique_feat": nunique_feat}
    feat_info_path = os.path.join(data_folder, "feat_info.pkl")
    save_feat_info(feat_info, feat_info_path)

    for feat in sparse_features:
        lbe = LabelEncoder()  # or Hash
        data[feat] = lbe.fit_transform(data[feat])
    mms = StandardScaler()
    data[dense_features] = mms.fit_transform(data[dense_features])
    split_data_ordered(data, data_folder)
