import json
import os
import random

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse import csr_matrix
from torch.utils.data import DataLoader, Dataset


def _normalize_user_item_dict(user_item_dict):
    return {int(k): [int(v) for v in values] for k, values in user_item_dict.items()}


def _load_meta_counts(dir_str):
    meta_path = os.path.join(dir_str, "meta.json")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if "user_num" not in meta or "item_num" not in meta:
        return None
    return int(meta["user_num"]), int(meta["item_num"])


def _infer_counts(train_edge, user_item_dict):
    user_ids = train_edge[:, 0].astype(np.int64).tolist()
    item_ids = train_edge[:, 1].astype(np.int64).tolist()
    user_ids.extend(int(k) for k in user_item_dict.keys())
    for items in user_item_dict.values():
        item_ids.extend(int(v) for v in items)
    user_num = (max(user_ids) + 1) if user_ids else 0
    item_num = (max(item_ids) + 1) if item_ids else 0
    return user_num, item_num


def _validate_feature_rows(name, feat, item_num):
    if feat is None:
        return
    if feat.size(0) != item_num:
        raise ValueError(
            f"{name} row count mismatch: expected {item_num}, got {feat.size(0)}. "
            "Please ensure item ids are remapped to contiguous [0, item_num) and feature rows follow the same order."
        )


def _to_device(feat, device):
    if feat is None or device is None:
        return feat
    return feat.to(device)


def _load_numpy_tensor(path, device):
    return _to_device(torch.tensor(np.load(path, allow_pickle=True), dtype=torch.float), device)


def _looks_like_multimodal_package(dir_str):
    return os.path.exists(os.path.join(dir_str, "train.npy")) and os.path.exists(
        os.path.join(dir_str, "user_item_dict.npy")
    )


def data_load(dataset, has_v=True, has_a=True, has_t=True, data_root="./Data", device=None):
    dir_str = os.path.join(data_root, dataset)
    if not os.path.isdir(dir_str):
        raise FileNotFoundError(f"LightGT data directory does not exist: {dir_str}")

    device = torch.device(device) if device is not None else None

    if dataset == "movielens" or _looks_like_multimodal_package(dir_str):
        train_edge = np.asarray(np.load(os.path.join(dir_str, "train.npy"), allow_pickle=True), dtype=np.int64)
        user_item_dict = _normalize_user_item_dict(
            np.load(os.path.join(dir_str, "user_item_dict.npy"), allow_pickle=True).item()
        )
        counts = _load_meta_counts(dir_str)
        if counts is None:
            user_num, item_num = _infer_counts(train_edge, user_item_dict)
        else:
            user_num, item_num = counts
        v_feat = (
            _load_numpy_tensor(os.path.join(dir_str, "FeatureVideo_normal.npy"), device) if has_v else None
        )
        a_feat = (
            _load_numpy_tensor(os.path.join(dir_str, "FeatureAudio_avg_normal.npy"), device) if has_a else None
        )
        t_feat = (
            _load_numpy_tensor(os.path.join(dir_str, "FeatureText_stl_normal.npy"), device) if has_t else None
        )
        _validate_feature_rows("FeatureVideo_normal.npy", v_feat, item_num)
        _validate_feature_rows("FeatureAudio_avg_normal.npy", a_feat, item_num)
        _validate_feature_rows("FeatureText_stl_normal.npy", t_feat, item_num)
    elif dataset == "tiktok":
        user_num = 36656
        item_num = 76085
        train_edge = np.load(os.path.join(dir_str, "train.npy"), allow_pickle=True)
        user_item_dict = np.load(os.path.join(dir_str, "user_item_dict.npy"), allow_pickle=True).item()
        v_feat = _to_device(torch.load(os.path.join(dir_str, "visual_feat_new.pt")).to(dtype=torch.float), device) if has_v else None
        a_feat = _to_device(torch.load(os.path.join(dir_str, "audio_feat_new.pt")).to(dtype=torch.float), device) if has_a else None
        t_feat = _to_device(torch.tensor(np.load(os.path.join(dir_str, "tiktok_t_64.npy"))).to(dtype=torch.float), device) if has_t else None
    elif dataset == "kwai":
        user_num = 7010
        item_num = 86483
        train_edge = np.load(os.path.join(dir_str, "train.npy"), allow_pickle=True)
        user_item_dict = np.load(os.path.join(dir_str, "user_item_dict.npy"), allow_pickle=True).item()
        v_feat = _to_device(torch.load(os.path.join(dir_str, "v_feat.pt")).to(dtype=torch.float), device) if has_v else None
        a_feat = None
        t_feat = _to_device(torch.tensor(np.load(os.path.join(dir_str, "kwai_t_64.npy"))).to(dtype=torch.float), device) if has_t else None
    else:
        raise ValueError(f"Unsupported LightGT dataset: {dataset}")

    train_edge = np.asarray(train_edge, dtype=np.int64).copy()
    train_edge[:, 1] += user_num
    user_item_dict = {i: [j + user_num for j in user_item_dict[i]] for i in user_item_dict.keys()}

    return user_num, item_num, train_edge, user_item_dict, v_feat, a_feat, t_feat


class TrainingDataset(Dataset):
    def __init__(self, dataset, user_num, item_num, user_item_dict, edge_index, src_len, data_root="./Data", device="cpu"):
        self.dir_str = os.path.join(data_root, dataset)
        self.user_num = user_num
        self.item_num = item_num
        self.user_item_dict = user_item_dict
        self.edge_index = edge_index
        self.src_len = src_len
        self.all_set = list(range(user_num, user_num + item_num))
        self.graph = None
        self.device = torch.device(device)

    def __len__(self):
        return len(self.edge_index)

    def __getitem__(self, index):
        user, pos_item = self.edge_index[index]
        while True:
            neg_item = random.sample(self.all_set, 1)[0]
            if neg_item not in self.user_item_dict[user]:
                break

        temp = list(self.user_item_dict[user])
        random.shuffle(temp)
        if len(temp) > self.src_len:
            mask = torch.ones(self.src_len + 1) == 0
            temp = temp[: self.src_len]
        else:
            mask = torch.cat((torch.ones(len(temp) + 1), torch.zeros(self.src_len - len(temp)))) == 0
            temp.extend([self.user_num for _ in range(self.src_len - len(temp))])

        user_item = torch.tensor(temp) - self.user_num
        user_item = torch.cat((torch.tensor([-1]), user_item))

        return torch.LongTensor([user, user]), torch.LongTensor([pos_item, neg_item]), user_item, mask

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        index = torch.from_numpy(np.vstack((coo.row, coo.col))).long()
        data = torch.from_numpy(coo.data).float()
        return torch.sparse_coo_tensor(index, data, torch.Size(coo.shape))

    def get_sparse_graph(self):
        if self.graph is None:
            try:
                norm_adj = sp.load_npz(os.path.join(self.dir_str, "s_pre_adj_mat.npz"))
            except Exception:
                adj_mat = sp.dok_matrix((self.user_num + self.item_num, self.user_num + self.item_num), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                train_user = self.edge_index[:, 0]
                train_item = self.edge_index[:, 1] - self.user_num
                R = csr_matrix(
                    (np.ones(len(train_user)), (train_user, train_item)),
                    shape=(self.user_num, self.item_num),
                ).tolil()
                adj_mat[: self.user_num, self.user_num :] = R
                adj_mat[self.user_num :, : self.user_num] = R.T
                adj_mat = adj_mat.todok()

                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum + 1e-5, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.0
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                sp.save_npz(os.path.join(self.dir_str, "s_pre_adj_mat.npz"), norm_adj)

            self.graph = self._convert_sp_mat_to_sp_tensor(norm_adj).coalesce().to(self.device)

        return self.graph


class EvalDataset(Dataset):
    def __init__(self, dataset, user_num, item_num, user_item_dict, src_len, data_root="./Data"):
        self.dir_str = os.path.join(data_root, dataset)
        self.user_num = user_num
        self.item_num = item_num
        self.user_item_dict = user_item_dict
        self.src_len = src_len

    def __len__(self):
        return self.user_num

    def __getitem__(self, index):
        user = index

        temp = list(self.user_item_dict[user])
        random.shuffle(temp)
        if len(temp) > self.src_len:
            mask = torch.ones(self.src_len + 1) == 0
            temp = temp[: self.src_len]
        else:
            mask = torch.cat((torch.ones(len(temp) + 1), torch.zeros(self.src_len - len(temp)))) == 0
            temp.extend([self.user_num for _ in range(self.src_len - len(temp))])

        user_item = torch.tensor(temp) - self.user_num
        user_item = torch.cat((torch.tensor([-1]), user_item))

        return torch.LongTensor([user]), user_item, mask
