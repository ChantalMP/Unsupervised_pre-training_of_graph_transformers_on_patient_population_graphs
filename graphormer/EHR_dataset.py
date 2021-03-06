import json
import os
import os.path as osp
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Dataset, Data

from graphormer import wrapper
from graphormer.utils.utils import summarize_acu_task


class EHRDataset(Dataset):
    def __init__(self, root, mask, name, raw_file_name, offset=3, bin_split_idx=14, transform=None, pre_transform=None, parts=None, split=None,
                 drop_val_patients=False, cross_val_split=None, fold=None, mask_all=False, k=5, mask_ratio=0.1, task=""):
        # if true masking task, else classification
        self.mask = mask
        self.mask_value = 95  # no class label but inside offset
        self.mask_all = False
        self.task = task
        self.parts = parts
        part_str = f'_truncated_{parts}parts' if parts else ''
        self.root = root
        self.raw_file_name = raw_file_name
        self.tranductive_pre = False
        self.k = k
        self.mask_ratio = mask_ratio

        self.discrete_columns = ['node_ID', 'DX_bl', 'PTGENDER', 'APOE4', 'CDRSB', 'ADAS11', 'MMSE', 'RAVLT_immediate']
        self.data_path = f'{name}_graph_class{part_str}{f"_drop_val_{split}" if drop_val_patients else ""}_fold{fold}{"_transductive_pre" if self.tranductive_pre else ""}{"_sim"}{f"_k{self.k}" if self.k != 5 else ""}.pt'
        self.drop_val_patients = drop_val_patients
        self.cross_val = False

        if split:
            self.split = split
            self.test_idxs = np.load('data/tadpole/split/test_idxs.npy')  # we do everyting with cross validation, test idxs are never used
            if cross_val_split is None:
                self.train_idxs = np.load(f'data/tadpole/split/cross_val/train_idxs_fold{fold}_strat.npy')
                self.val_idxs = np.load(f'data/tadpole/split/cross_val/val_idxs_fold{fold}_strat.npy')
            else:
                self.cross_val = True
                self.cross_val_split = cross_val_split
                self.train_idxs = np.load(f'data/tadpole/split/cross_val/train_idxs_fold{cross_val_split}_strat.npy')
                self.val_idxs = np.load(f'data/tadpole/split/cross_val/val_idxs_fold{cross_val_split}_strat.npy')
        else:
            self.split = None

        super().__init__(root, transform, pre_transform)

        processed_data = torch.load(osp.join(self.processed_dir, self.data_path))
        pre_processed_data_path = Path(osp.join(self.processed_dir, f'{name}_graph_class_preprocessed{part_str}'
                                                                    f'{f"_drop_val_{split}" if drop_val_patients else ""}_fold{fold}'
                                                                    f'{"_transductive_pre" if self.tranductive_pre else ""}{"_sim"}{f"_k{self.k}" if self.k != 5 else ""}.pt'))

        if pre_processed_data_path.exists():
            self.pre_processed_data = torch.load(pre_processed_data_path)
        else:
            self.pre_processed_data = wrapper.preprocess_item_ehr(item=processed_data, offset=offset, edges=True, data_path=pre_processed_data_path,
                                                                  bin_split_idx=bin_split_idx)

    @property
    def raw_file_names(self):
        return [self.raw_file_name]

    @property
    def processed_file_names(self):
        return [self.data_path]

    def create_edges(self, df, age_key, sex_key, feature_type=1):
        # edge_index
        # edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edges = []
        edge_features = []
        for idx1, node1 in enumerate(df.iterrows()):
            if idx1 % 100 == 0:
                print(idx1)
            for idx2, node2 in enumerate(df.iterrows()):
                if idx1 != idx2:
                    age_diff = round(abs(node1[1][age_key] - node2[1][age_key]))
                    sex_diff = abs(node1[1][sex_key] - node2[1][sex_key])
                    if age_diff <= 2:
                        if feature_type == 1:  # 1 feature for age and sex together
                            weight = 3 - age_diff  # 3 for same age, 2 for 1 year diff, 1 for 2 years diff and diff gender
                            if sex_diff == 0:
                                weight += 3  # 6 for same age, 5 for 1 year diff, 4 for 2 years diff and same gender
                            edge_features.append(int(weight))
                        elif feature_type == 2:  # seperate features for age and sex
                            age_weight = 3 - age_diff
                            sex_weight = 1 - sex_diff  # one if same gender, else 0
                            weight = torch.tensor([age_weight, sex_weight])
                            edge_features.append(weight.int())

                        # add edge if age difference small enough
                        edges.append((idx1, idx2))

        edge_idx = torch.tensor(edges, dtype=torch.long)
        edge_features = torch.tensor(edge_features, dtype=torch.int32) if feature_type == 1 else torch.stack(edge_features)
        return edge_idx, edge_features

    def create_x_y(self, X_norm, df, label_key):

        # x: node feature matrix [num_nodes, num_node_features]
        x = torch.tensor(X_norm.values.astype(np.float32))

        # replace all missing values with mask value
        missing_mask = torch.isnan(x)
        x[missing_mask] = self.mask_value

        # y: Target to train against, node-level targets of shape [num_nodes, *]
        if self.mask:
            y = x.clone()  # input features should be predicted, no features masked yet
        else:
            y = torch.tensor(df[label_key].values, dtype=torch.float)

        return x, y

    def save_data(self, node_id, x, edge_idx, edge_features, y, final_mask, train_mask, val_mask, test_mask, not_mask_column_indices, node_idx=None,
                  split=None):
        data = Data(node_id=node_id, x=x, edge_index=edge_idx.t().contiguous(), edge_attr=edge_features, y=y, update_mask=final_mask,
                    train_mask=train_mask, val_mask=val_mask, test_mask=test_mask, mask_value=self.mask_value,
                    not_mask_column_indices=not_mask_column_indices, split=split)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        if node_idx is not None:
            torch.save(data, osp.join(self.processed_dir, f'one_node_graphs{"" if self.mask else "_class"}/graph_{node_idx}'))
        else:
            torch.save(data, osp.join(self.processed_dir, self.data_path))

    def process(self):
        pass

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = self.pre_processed_data
        data.mask_task = None
        if self.mask:
            data.y = data.orig_x.clone()  # (data.orig_x.clone(), data.y) only for neighbour label metric
        return data


class TadpoleDataset(EHRDataset):

    def tadpole_train_val_test_masks(self, df):
        train_idxs, val_idxs, test_idxs = self.train_idxs, self.val_idxs, self.test_idxs
        if self.drop_val_patients:
            train_mask = torch.zeros(len(self.train_idxs), dtype=torch.bool)
            val_mask = torch.zeros(len(self.train_idxs) + len(self.val_idxs), dtype=torch.bool)
            test_mask = torch.zeros(len(self.train_idxs) + len(self.test_idxs), dtype=torch.bool)
            train_idxs = [df.index.get_loc(df.index[df.node_ID == train_idx][0]) for train_idx in train_idxs] if self.split == 'train' else []
            val_idxs = [df.index.get_loc(df.index[df.node_ID == val_idx][0]) for val_idx in val_idxs] if self.split == 'val' else []
            test_idxs = [df.index.get_loc(df.index[df.node_ID == test_idx][0]) for test_idx in test_idxs] if self.split == 'test' else []

        else:
            train_mask = torch.zeros(564, dtype=torch.bool) if not self.tranductive_pre else torch.ones(564,
                                                                                                        dtype=torch.bool)  # pretrain with all nodes
            val_mask = torch.zeros(564, dtype=torch.bool) if not self.tranductive_pre else torch.ones(564, dtype=torch.bool)  # no validation set
            test_mask = torch.zeros(564, dtype=torch.bool)
        train_mask[train_idxs] = True
        val_mask[val_idxs] = True
        test_mask[test_idxs] = True

        return train_mask, val_mask, test_mask

    def compute_dem_similarity(self, node1_age, node2_age, node1_sex, node2_sex, node1_apoe, node2_apoe):
        age_diff = abs(node1_age - node2_age)
        sex_diff = abs(node1_sex - node2_sex)
        apoe_diff = abs(node1_apoe == node2_apoe)
        return torch.tensor(((age_diff <= 2) + (1 - sex_diff) + apoe_diff) / 3)

    def compute_cog_test_similarity(self, node1_cog, node2_cog):
        if np.isnan(node1_cog).any() or np.isnan(node2_cog).any():
            nan_mask = ~np.logical_or(np.isnan(node1_cog), np.isnan(node2_cog))
            dist = torch.tensor(np.linalg.norm(node1_cog[nan_mask] - node2_cog[nan_mask], axis=0))
        else:
            dist = torch.tensor(np.linalg.norm(node1_cog - node2_cog, axis=0))
        return 1 - (dist - 0) / (116 - 0)  # normalized

    def compute_imaging_similarity(self, node1_img, node2_img):
        dist = torch.tensor(np.linalg.norm(node1_img - node2_img, axis=0))
        return 1 - torch.sigmoid(dist)

    def create_edges_similarity_tadpole(self, df):
        # edge_index
        # edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edges = []
        edge_features = []
        ages = {}
        sexs = {}
        apoes = {}
        cogs = {}
        imgs = {}
        for idx1, node1 in enumerate(df.iterrows()):
            if idx1 in ages:
                node1_age = ages[idx1]
                node1_sex = sexs[idx1]
                node1_apoe = apoes[idx1]
                node1_cog = cogs[idx1]
                node1_img = imgs[idx1]
            else:
                ages[idx1] = node1[1]['AGE']
                sexs[idx1] = node1[1]['PTGENDER']
                apoes[idx1] = node1[1]['APOE4']
                cogs[idx1] = node1[1][['CDRSB', 'ADAS11', 'MMSE', 'RAVLT_immediate']].values
                imgs[idx1] = node1[1][['Hippocampus', 'WholeBrain', 'Entorhinal', 'MidTemp', 'FDG']].values
                node1_age = ages[idx1]
                node1_sex = sexs[idx1]
                node1_apoe = apoes[idx1]
                node1_cog = cogs[idx1]
                node1_img = imgs[idx1]

            all_sims = {}
            if idx1 % 100 == 0:
                print(idx1)
            for idx2, node2 in enumerate(df.iterrows()):
                if idx1 != idx2:

                    if idx2 in ages:
                        node2_age = ages[idx2]
                        node2_sex = sexs[idx2]
                        node2_apoe = apoes[idx2]
                        node2_cog = cogs[idx2]
                        node2_img = imgs[idx2]
                    else:
                        ages[idx2] = node2[1]['AGE']
                        sexs[idx2] = node2[1]['PTGENDER']
                        apoes[idx2] = node2[1]['APOE4']
                        cogs[idx2] = node2[1][['CDRSB', 'ADAS11', 'MMSE', 'RAVLT_immediate']].values
                        imgs[idx2] = node2[1][['Hippocampus', 'WholeBrain', 'Entorhinal', 'MidTemp', 'FDG']].values
                        node2_age = ages[idx2]
                        node2_sex = sexs[idx2]
                        node2_apoe = apoes[idx2]
                        node2_cog = cogs[idx2]
                        node2_img = imgs[idx2]

                    # compute all similarity features
                    dem_similarity = self.compute_dem_similarity(node1_age, node2_age, node1_sex, node2_sex, node1_apoe, node2_apoe)
                    cog_similarity = self.compute_cog_test_similarity(node1_cog, node2_cog)
                    imaging_similarity = self.compute_imaging_similarity(node1_img, node2_img)

                    all_sims[idx2] = [np.mean([dem_similarity, cog_similarity, imaging_similarity]), dem_similarity, cog_similarity,
                                      imaging_similarity]

            sorted_sims = sorted(all_sims.items(), key=lambda x: x[1][0], reverse=True)
            for i in range(self.k):  # add 5 nearest neighbours to edges
                edges.append((idx1, sorted_sims[i][0]))
                edge_features.append(
                    torch.tensor(
                        [torch.round(sorted_sims[i][1][1] * 3), torch.round(sorted_sims[i][1][2] * 100), torch.round(sorted_sims[i][1][3] * 100)],
                        dtype=torch.int))

        edge_idx = torch.tensor(edges, dtype=torch.long)
        edge_features = torch.stack(edge_features)
        return edge_idx, edge_features

    def process(self):
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            df = pd.read_csv(raw_path)

            # only select labels of start set
            df = df[['node_ID', 'DX_bl', 'AGE', 'PTGENDER', 'APOE4', 'CDRSB', 'ADAS11', 'MMSE', 'RAVLT_immediate', 'Hippocampus', 'WholeBrain',
                     'Entorhinal', 'MidTemp', 'FDG']]

            if self.drop_val_patients:  # inductive setup, transductive currently not supported
                if self.split == 'train':
                    idxs = self.train_idxs
                elif self.split == 'val':
                    idxs = np.concatenate((self.val_idxs, self.train_idxs))
                else:  # test split
                    idxs = np.concatenate((self.test_idxs, self.train_idxs))

                # inductive: normalize only given training data
                for col in [col for col in df.columns if col not in self.discrete_columns and col != 'AGE']:  # normalize imaging features
                    df[col] = (df[col] - df[col][self.train_idxs].min()) / (df[col][self.train_idxs].max() - df[col][self.train_idxs].min())

                df_norm = df.copy()  # in df age remains unnormalized, so we can compute age similarity
                df_norm['AGE'] = (df['AGE'] - df['AGE'][self.train_idxs].min()) / (
                        df['AGE'][self.train_idxs].max() - df['AGE'][self.train_idxs].min())  # normalize age for processing
                df = df[df.node_ID.isin(idxs)]  # unnormalized age
                df_norm = df_norm[df_norm.node_ID.isin(idxs)]  # normalized age

            node_ids = np.array(df_norm['node_ID'].values)
            # drop labels and edge building features
            drop_columns = ['DX_bl', 'node_ID']

            X = df_norm.drop(drop_columns, axis=1)

            # discrete_columns without dropped ones
            discrete_features = [col for col in self.discrete_columns if col not in drop_columns]
            # sort columns so discrete features are in the front
            X = X[discrete_features + [col for col in X.columns if col not in discrete_features]]

            if self.mask_all:
                mask_columns = ['APOE4', 'CDRSB', 'ADAS11', 'MMSE', 'RAVLT_immediate']
                not_mask_column_indices = np.where(np.isin(X.columns, mask_columns, invert=True))
            else:
                mask_columns = ['APOE4', 'CDRSB', 'ADAS11', 'ADAS13', 'MMSE', 'RAVLT_immediate', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'MidTemp',
                                'FDG']
                not_mask_column_indices = np.where(np.isin(X.columns, mask_columns, invert=True))  # dont mask age and gender

            x, y = self.create_x_y(X_norm=X, df=df_norm, label_key='DX_bl')

            train_mask, val_mask, test_mask = self.tadpole_train_val_test_masks(df_norm)

            edge_idx, edge_features = self.create_edges_similarity_tadpole(df)

            self.save_data(node_id=node_ids, x=x, edge_idx=edge_idx, edge_features=edge_features, y=y, final_mask=None, train_mask=train_mask,
                           val_mask=val_mask, test_mask=test_mask, not_mask_column_indices=not_mask_column_indices, split=self.split)


class MIMICDataset(Dataset):
    def __init__(self, root, drop_val_patients=False, use_treatment_input=True, task="", split='train', transform=None, pre_transform=None,
                 edge_vars='age',
                 num_graphs=43, gcn=False, mlp=False, gat=False, graphsage=False, gin=False, rotation=0, predict=False, pad_mode='original',
                 mask_ratio=0.15,
                 block_size=6, k=5, use_simple_acu_task=False):
        # if true masking task, else classification
        self.mask_value = 0
        self.my_root = root
        self.split = split
        self.rotation = rotation
        self.gcn = gcn
        self.mlp = mlp
        self.gat = gat
        self.graphsage = graphsage
        self.gin = gin
        self.pad_mode = pad_mode
        self.predict = predict
        self.drop_val_patients = drop_val_patients
        self.edge_vars = edge_vars
        self.k = k
        if self.edge_vars == 'vals':  # final decision, code only supports edges based on measurements ("vals")
            edge_id = f'5_k{self.k}' if self.k != 5 else 5

        self.lin_interpolation = True
        self.mask_col = (self.pad_mode == 'pad_emb')
        self.mask_ratio = mask_ratio
        self.block_size = block_size
        if self.split == 'test':
            self.all_graph_idx_files = [f'rot{rotation}/random_graph_subset_{i}.json' for i in range(num_graphs)]
            self.all_graphs = [
                f'rotations/rot{rotation}/mimic_graph_test_edge_{edge_id}_knn_{"lin_interpol_" if self.lin_interpolation else ""}subset_{i}.pt' for i
                in range(num_graphs)]
            self.all_pre_processed_graphs = [
                f'rotations/rot{rotation}/mimic_graph_test_processed_edge_{edge_id}{"_mask_col" if self.mask_col else ""}_knn_{"lin_interpol_" if self.lin_interpolation else ""}subset_{i}.pt'
                for i in range(num_graphs)]
            self.data_path = f'rotations/rot{rotation}/mimic_graph_test_edge_{edge_id}_knn_{"lin_interpol_" if self.lin_interpolation else ""}subset_'

        else:
            self.all_graph_idx_files = [f'rot{rotation}/random_graph_subset_dev_{i}.json' for i in
                                        range(num_graphs)]  # 55 for age graph, 43 for random graph
            self.all_graphs = [
                f'rotations/rot{rotation}/mimic_graph_full_edge_{edge_id}_knn_{"lin_interpol_" if self.lin_interpolation else ""}subset_{i}.pt' if not drop_val_patients or self.split != 'train' else f'rotations/rot{rotation}/mimic_graph_train_edge_{edge_id}_knn_{"lin_interpol_" if self.lin_interpolation else ""}subset_{i}.pt'
                for i in
                range(num_graphs)]
            self.all_pre_processed_graphs = [
                f'rotations/rot{rotation}/mimic_graph_full_processed_edge_{edge_id}{"_mask_col" if self.mask_col else ""}_knn_{"lin_interpol_" if self.lin_interpolation else ""}subset_{i}.pt' if not drop_val_patients or self.split != 'train'
                else f'rotations/rot{rotation}/mimic_graph_train_processed_edge_{edge_id}{"_mask_col" if self.mask_col else ""}_knn_{"lin_interpol_" if self.lin_interpolation else ""}subset_{i}.pt'
                for i in range(num_graphs)]
            self.data_path = f'rotations/rot{rotation}/mimic_graph_full_edge_{edge_id}_knn_{"lin_interpol_" if self.lin_interpolation else ""}subset_' if not self.drop_val_patients or self.split != 'train' else f'rotations/rot{rotation}/mimic_graph_train_edge_{edge_id}_knn_{"lin_interpol_" if self.lin_interpolation else ""}subset_'
        self.use_treatment_input = use_treatment_input
        self.task = task  # possible tasks: mask_random, mask_next_step, mask_treatment, los, acu
        self.use_simple_acu_task = use_simple_acu_task
        self.mask = task in ['mask_random', 'mask_next_step', 'mask_treatment']

        super().__init__(root + f'/{split}', transform, pre_transform)

        if not gcn:
            for graph in self.all_graphs:
                subset = graph.split('.')[0].split('_')[-1]
                processed_data = torch.load(osp.join(self.processed_dir, graph))
                if self.split == 'test':
                    pre_processed_data_path = Path(osp.join(self.processed_dir,
                                                            f'rotations/rot{rotation}/mimic_graph_test_processed_edge_{edge_id}{"_mask_col" if self.mask_col else ""}_knn_{"lin_interpol_" if self.lin_interpolation else ""}subset_{subset}.pt'))

                else:
                    pre_processed_data_path = Path(osp.join(self.processed_dir,
                                                            f'rotations/rot{rotation}/mimic_graph_full_processed_edge_{edge_id}{"_mask_col" if self.mask_col else ""}_knn_{"lin_interpol_" if self.lin_interpolation else ""}subset_{subset}.pt' if not self.drop_val_patients or self.split != 'train'
                                                            else f'rotations/rot{rotation}/mimic_graph_train_processed_edge_{edge_id}{"_mask_col" if self.mask_col else ""}_knn_{"lin_interpol_" if self.lin_interpolation else ""}subset_{subset}.pt'))

                if not pre_processed_data_path.exists():
                    # preprocess and save results
                    wrapper.preprocess_item_ehr_mimic(item=processed_data, data_path=pre_processed_data_path, edge_vars=self.edge_vars)

    @property
    def raw_file_names(self):
        return self.all_graph_idx_files

    @property
    def processed_file_names(self):
        return self.all_graphs

    def train_val_test_masks(self, splits):
        train_mask, val_mask, test_mask, train_dev_mask, dev_mask = None, None, None, None, None
        # inductive setup
        if self.split == 'train':
            # get all elements in split where second element is 'train'
            train_data = [split for split in splits if split[1] == 'train' or split[1] == 'dev']
            train_mask = torch.tensor([split[1] == 'train' or split[1] == 'dev' for split in train_data])  # for regular training ignore dev set
            train_dev_mask = torch.tensor([split[1] == 'train' for split in splits])

        elif self.split == 'val':
            # get all elements in split where second element is 'val' or 'train'
            val_data = [split for split in splits if split[1] == 'val' or split[1] == 'train' or split[1] == 'dev']
            train_mask = torch.tensor([split[1] == 'train' or split[1] == 'dev' for split in val_data])
            train_dev_mask = torch.tensor([split[1] == 'train' for split in splits])
            dev_mask = torch.tensor([split[1] == 'dev' for split in val_data])
            val_mask = torch.tensor([split[1] == 'val' for split in val_data])

        elif self.split == 'test':
            # get all elements in split where second element is 'test' or 'train' or 'val'
            test_data = [split for split in splits if split[1] == 'test' or split[1] == 'train' or split[1] == 'val' or split[1] == 'dev']
            test_mask = torch.tensor([split[1] == 'test' for split in test_data])

        return train_mask, val_mask, test_mask, train_dev_mask, dev_mask

    def patient_part_of_split(self, patient_split):
        if self.drop_val_patients:
            if patient_split == 'train' or patient_split == 'dev':
                return self.split == 'train' or self.split == 'val' or self.split == 'test'
            elif patient_split == 'val':
                return self.split == 'val' or self.split == 'test'
            elif patient_split == 'test':
                return self.split == 'test'
        else:
            return self.split == 'train' or self.split == 'val' or self.split == 'test'  # for now we do not use test set nodes at all


    def get_or_compute_val_descriptor(self, idx, vals, descriptor_cache):
        if idx in descriptor_cache:
            return descriptor_cache[idx]
        else:
            descriptors = np.stack([vals.mean(axis=0), vals.std(axis=0), vals.min(axis=0)[0], vals.max(axis=0)[0]])
            descriptor_cache[idx] = descriptors
            return descriptors

    def compute_vals_similarity(self, vals_descriptors1, vals_descriptors2):
        # per column compute similarity of time-series
        dist = torch.tensor(np.linalg.norm(vals_descriptors1 - vals_descriptors2, axis=0).mean())
        return 1 - torch.sigmoid(dist)  # in which range is this value? in test graph: 0.01 - 0.42 for half features, 0.02 - 0.4 for full features

    def create_edges_feat_similarity_knn(self, patient_half_vals):
        # edge_index
        # edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edges = []
        edge_features = []
        descriptor_cache = {}
        for idx1, half_vals in enumerate(patient_half_vals):
            all_sims = {}
            if idx1 % 100 == 0:
                print(idx1)
            for idx2, half_vals2 in enumerate(patient_half_vals):
                if idx1 != idx2:
                    vals_descriptors1 = self.get_or_compute_val_descriptor(idx1, half_vals, descriptor_cache)
                    vals_descriptors2 = self.get_or_compute_val_descriptor(idx2, half_vals2, descriptor_cache)
                    vals_similarity = self.compute_vals_similarity(vals_descriptors1, vals_descriptors2)
                    all_sims[idx2] = vals_similarity

            # compute 5 nearest neighbors and add edges
            sorted_sims = sorted(all_sims.items(), key=lambda x: x[1], reverse=True)
            for i in range(self.k):
                edges.append((idx1, sorted_sims[i][0]))
                edge_features.append(torch.round((sorted_sims[i][1] * 100)).int())  # discretize similarity to 0-17

        edge_idx = torch.tensor(edges, dtype=torch.long)
        edge_features = torch.stack(edge_features)[:, None]
        return edge_idx, edge_features


    def save_data(self, node_id, vals, demographics, treatments, is_measured, edge_idx, edge_features, y, final_mask, train_mask, val_mask, test_mask,
                  train_dev_mask, dev_mask, subset_id):
        data = Data(node_id=node_id, vals=vals, demographics=demographics, treatments=treatments, is_measured=is_measured,
                    edge_index=edge_idx.t().contiguous(), edge_attr=edge_features, y=y, update_mask=final_mask, train_mask=train_mask,
                    train_dev_mask=train_dev_mask, dev_mask=dev_mask, val_mask=val_mask, test_mask=test_mask, split=self.split,
                    mask_value=self.mask_value)

        # create rotation dataset if it does not exist
        if not os.path.exists(osp.join(self.processed_dir, f'rot{self.rotation}')):
            os.makedirs(osp.join(self.processed_dir, f'rot{self.rotation}'))

        torch.save(data, osp.join(self.processed_dir, self.data_path + f'{subset_id}.pt'))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        if self.gcn or self.mlp or self.gat or self.graphsage or self.gin:
            data = torch.load(osp.join(self.processed_dir, self.all_graphs[idx]))
            data = wrapper.preprocess_item_mimic_gcn(item=data)
            data.mask_task = None  # as they are not collated, have to set manually - anyway only needed for multi-task PT
            data.num_nodes = len(data.vals)
        else:
            data = torch.load(osp.join(self.processed_dir, self.all_pre_processed_graphs[idx]))
        window_length = 24
        # form y vector given task
        if self.task == 'pre_mask' or self.task == 'patient_prediction':
            # y_acu = torch.tensor([d['acu'] for d in data.y])
            # y_acu = summarize_acu_task(y_acu)
            y_los = torch.tensor([d['los'] for d in data.y])
            y = (torch.stack([d['vals'][:window_length] for d in data.y]), torch.stack([d['treatments_hour'][:window_length] for d in data.y]), y_los)
        elif self.task == 'treat_prediction':
            y = torch.stack([d['treatments_bin'][:14] for d in data.y])
        elif self.task == 'multi_task_pt':  # collect gt for all possible tasks (treatment prediction, feature masking, patient masking)
            y = (torch.stack([d['vals'][:window_length] for d in data.y]), torch.stack([d['treatments_hour'][:window_length] for d in data.y]),
                 torch.stack([d['treatments_bin'][:14] for d in data.y]))
        else:
            if self.task == 'acu':
                y = torch.tensor([d[self.task] for d in data.y])
                assert y.max() <= 17, f'ACU label higher 17 found {y.max()}'
                if self.use_simple_acu_task:
                    y = summarize_acu_task(y)
            else:
                y = torch.tensor([d[self.task] for d in data.y])

        padding_mask = None
        if self.split == 'val' or self.split == 'train' or self.split == 'test':  # currently always
            treatments = torch.stack([d[:window_length] for d in data['treatments']])
            is_measured = torch.stack([d[:window_length] for d in data['is_measured']])
            vals = torch.stack([d[:window_length] for d in data['vals']])
            data.y = y
        # set changed variables in data
        data.vals = vals
        data.treatments = treatments
        data.is_measured = is_measured
        data.predict = self.predict
        data.padding_mask = padding_mask
        return data

    def process(self):
        for raw_path in self.raw_paths:
            subset_id = raw_path.split('.')[0].split('_')[-1]
            # Read data from `raw_path`.
            patient_icuids_and_splits = json.load(open(raw_path, 'r'))  # gives list of patients that should be in graph together with their split

            # Keep vals and treatments separate as we sometimes want treatments in training data and sometimes not
            patient_icuids = []
            patient_vals = []
            patient_half_vals = []
            patient_half_is_measured = []
            patient_half_vals_node = []
            patient_half_is_measured_node = []
            patient_dems = []
            patient_graph_dems = []
            patient_treatments = []
            patient_is_measured = []
            patient_ys = []  # dict for several prediction tasks
            # go through all patient folders with id in patient_icuids
            for info in patient_icuids_and_splits:
                if self.rotation == 0:
                    patient_icuid, split = info
                    folder = split
                else:
                    patient_icuid, split, folder = info

                patient_icuid = int(patient_icuid)
                if self.patient_part_of_split(split):
                    if folder == 'dev':
                        folder = 'train'
                    patient_dir = osp.join(self.my_root, folder,
                                           f'patient_{patient_icuid}')  # for other rotations than 0 the patient might not be saved in the splits folder, because all is saved according to rotation 0
                    statics = pd.read_csv(patient_dir + '/' + 'statics.csv')
                    ts_vals = pd.read_csv(patient_dir + '/' + 'ts_vals_linear_imputed.csv') if self.lin_interpolation else pd.read_csv(
                        patient_dir + '/' + 'ts_vals.csv')
                    ts_treatment = pd.read_csv(patient_dir + '/' + 'ts_treatment.csv')
                    ts_is_measured = pd.read_csv(patient_dir + '/' + 'ts_is_measured.csv')
                    static_tasks_binary_multilabel = pd.read_csv(patient_dir + '/' + 'static_tasks_binary_multilabel.csv')
                    final_acuity_outcome = pd.read_csv(patient_dir + '/' + 'Final Acuity Outcome.csv')

                    # here we save full ICU stay in graph, so that we can do random cropping + padding during training also for same graph
                    patient_icuids.append(patient_icuid)
                    patient_vals.append(torch.tensor(ts_vals.values.astype(np.float32)))
                    patient_dems.append(
                        torch.tensor(statics.drop(['ethnicity', 'insurance'], axis=1).values.astype(np.float32)))  # drop for ethical reasons
                    patient_graph_dems.append(statics[['age', 'gender']])  # can stay df as it will not be used as model input or label
                    patient_treatments.append(torch.tensor(ts_treatment.values.astype(np.float32)))
                    patient_is_measured.append(torch.tensor(ts_is_measured.values.astype(np.float32)))

                    y_dict = {}
                    # create tensor for treatments which is one if any element in column is 1
                    y_dict['treatments_bin'] = torch.tensor(np.any(ts_treatment.values, axis=0).astype(np.float32))
                    y_dict['treatments_hour'] = torch.tensor(ts_treatment.values.astype(np.float32))
                    y_dict['vals'] = torch.tensor(
                        ts_vals.values.astype(np.float32))  # for next timestepp task has to be created when cropping is done
                    y_dict['los'] = torch.tensor(static_tasks_binary_multilabel['Long LOS'].values.astype(np.float32))
                    y_dict['acu'] = torch.tensor(final_acuity_outcome.values.astype(np.float32))
                    patient_ys.append(y_dict)

            train_mask, val_mask, test_mask, train_dev_mask, dev_mask = self.train_val_test_masks(patient_icuids_and_splits)

            if self.edge_vars == 'vals':
                edge_idx, edge_features = self.create_edges_feat_similarity_knn(patient_vals)

            self.save_data(node_id=torch.tensor(patient_icuids), vals=patient_vals, demographics=torch.stack(patient_dems),
                           treatments=patient_treatments, is_measured=patient_is_measured, edge_idx=edge_idx,
                           edge_features=edge_features, y=patient_ys, final_mask=None, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
                           train_dev_mask=train_dev_mask, dev_mask=dev_mask, subset_id=subset_id)


class SepsisDataset(Dataset):
    def __init__(self, root, split='train', rotation=None, set_id='A', label_ratio=1.0):
        # if true masking task, else classification
        self.my_root = root
        self.split = split
        self.rotation = rotation
        self.set_id = set_id  # possible values A, B, AB

        self.k = 5
        num_graphs = 40

        if label_ratio != 1.0:
            num_graphs = max(int(num_graphs * label_ratio), 1)  # at least 1 graph, so 500 nodes per dataset -> equals 2.5% instead of 1%

        if rotation is not None:
            # 5 splits (0-4)
            train_idxs, val_idxs, test_idxs = self.get_train_val_test_split(rotation=rotation, label_ratio=label_ratio)
            idxs = train_idxs if split == 'train' else val_idxs if split == 'val' else test_idxs
            if label_ratio == 0.01 and split == 'train':  # create smaller graphs for this case
                self.all_graph_idx_files = [f'cross_val_A/random_graph_lr001_rot{rotation}.json',
                                            f'cross_val_B/random_graph_lr001_rot{rotation}.json']
                self.all_graphs = [f'sepsis_graph_lr001_rot{rotation}_A.pt', f'sepsis_graph_lr001_rot{rotation}_B.pt']
                self.all_pre_processed_graphs = [f'sepsis_graph_processed_lr001_rot{rotation}_A_24h.pt',
                                                 f'sepsis_graph_processed_lr001_rot{rotation}_B_24h.pt']
                self.data_path = f'sepsis_graph_lr001_'
            else:
                self.all_graph_idx_files = [f'cross_val_A/random_graph_subset_{i}.json' for i in idxs] + [f'cross_val_B/random_graph_subset_{i}.json'
                                                                                                          for i in idxs]
                self.all_graphs = [f'sepsis_graph_subset_{i}_A.pt' for i in idxs] + [f'sepsis_graph_subset_{i}_B.pt' for i in idxs]
                self.all_pre_processed_graphs = [f'sepsis_graph_processed_subset_{i}_A_24h.pt' for i in idxs] + [
                    f'sepsis_graph_processed_subset_{i}_B_24h.pt' for i in idxs]
                self.data_path = f'sepsis_graph_subset_'
        else:
            if self.split == 'test':
                if self.set_id == 'AB':  # use all 4 test graphs for testing
                    self.all_graph_idx_files = [f'training_setA/random_test_graph_subset_{i}.json' for i in range(4)] + [
                        f'training_setB/random_test_graph_subset_{i}.json' for i in range(4)]  # always use full test set
                    self.all_graphs = [f'sepsis_test_graph_test_subset_{i}_A.pt' for i in range(4)] + \
                                      [f'sepsis_test_graph_test_subset_{i}_B.pt' for i in range(4)]
                    self.all_pre_processed_graphs = [f'sepsis_test_graph_test_processed_subset_{i}_A_24h.pt' for i in range(4)] + [
                        f'sepsis_test_graph_test_processed_subset_{i}_B_24h.pt' for i in range(4)]

                else:
                    self.all_graph_idx_files = [f'training_set{self.set_id}/random_test_graph_subset_{i}.json' for i in range(4)]
                    self.all_graphs = [f'sepsis_test_graph_test_subset_{i}_{self.set_id}.pt' for i in range(4)]
                    self.all_pre_processed_graphs = [f'sepsis_test_graph_test_processed_subset_{i}_{self.set_id}.pt_24h' for i in range(4)]
                self.data_path = 'sepsis_test_graph_test_subset_'

            elif self.split == 'val':  # use all validation graphs
                if self.set_id == 'AB':
                    self.all_graph_idx_files = [f'training_setA/random_val_graph_subset_{i}.json' for i in range(4)] + \
                                               [f'training_setB/random_val_graph_subset_{i}.json' for i in range(4)]
                    self.all_graphs = [f'sepsis_val_graph_full_subset_{i}_A.pt' for i in range(4)] + \
                                      [f'sepsis_val_graph_full_subset_{i}_B.pt' for i in range(4)]
                    self.all_pre_processed_graphs = [f'sepsis_val_graph_full_processed_subset_{i}_A_24h.pt' for i in range(4)] + \
                                                    [f'sepsis_val_graph_full_processed_subset_{i}_B_24h.pt' for i in range(4)]

                else:
                    self.all_graph_idx_files = [f'training_set{self.set_id}/random_val_graph_subset_{i}.json' for i in range(4)]
                    self.all_graphs = [f'sepsis_val_graph_full_subset_{i}_{self.set_id}.pt' for i in range(4)]
                    self.all_pre_processed_graphs = [f'sepsis_val_graph_full_processed_subset_{i}_{self.set_id}_24h.pt' for i in range(4)]

                self.data_path = f'sepsis_val_graph_full_subset_'


            else:  # training - use graphs according to label ratio
                if self.set_id == 'AB':
                    self.all_graph_idx_files = [f'training_setA/random_graph_subset_{i}.json' for i in range(num_graphs)] + [
                        f'training_setB/random_graph_subset_{i}.json' for i in range(num_graphs)]
                    self.all_graphs = [f'sepsis_graph_{"train" if self.split == "train" else "full"}_subset_{i}_A.pt' for i in range(num_graphs)] + \
                                      [f'sepsis_graph_{"train" if self.split == "train" else "full"}_subset_{i}_B.pt' for i in range(num_graphs)]
                    self.all_pre_processed_graphs = [f'sepsis_graph_{"train" if self.split == "train" else "full"}_processed_subset_{i}_A_24h.pt' for
                                                     i in range(num_graphs)] + \
                                                    [f'sepsis_graph_{"train" if self.split == "train" else "full"}_processed_subset_{i}_B_24h.pt' for
                                                     i in range(num_graphs)]

                else:
                    self.all_graph_idx_files = [f'training_set{self.set_id}/random_graph_subset_{i}.json' for i in range(num_graphs)]
                    self.all_graphs = [f'sepsis_graph_{"train" if self.split == "train" else "full"}_subset_{i}_{self.set_id}.pt' for i in
                                       range(num_graphs)]
                    self.all_pre_processed_graphs = [
                        f'sepsis_graph_{"train" if self.split == "train" else "full"}_processed_subset_{i}_{self.set_id}_24h.pt' for i in
                        range(num_graphs)]

                self.data_path = f'sepsis_graph_{"train" if self.split == "train" else "full"}_subset_'

        super().__init__(root + f'/{split}' if rotation is None else root, transform=None, pre_transform=None)

        for graph in self.all_graphs:
            subset = graph.split('.')[0].split('_')[-2]
            set_id = graph.split('.')[0].split('_')[-1]
            processed_data = torch.load(osp.join(self.processed_dir, graph))
            if rotation is not None:
                if label_ratio == 0.01 and split == 'train':
                    pre_processed_data_path = Path(osp.join(self.processed_dir, f'sepsis_graph_processed_lr001_{subset}_{set_id}_24h.pt'))
                else:
                    pre_processed_data_path = Path(osp.join(self.processed_dir, f'sepsis_graph_processed_subset_{subset}_{set_id}_24h.pt'))
            else:
                if self.split == 'test':
                    pre_processed_data_path = Path(osp.join(self.processed_dir, f'sepsis_test_graph_test_processed_subset_{subset}_{set_id}_24h.pt'))
                elif self.split == 'val':
                    pre_processed_data_path = Path(osp.join(self.processed_dir, f'sepsis_val_graph_full_processed_subset_{subset}_{set_id}_24h.pt'))
                else:
                    pre_processed_data_path = Path(osp.join(self.processed_dir, f'sepsis_graph_train_processed_subset_{subset}_{set_id}_24h.pt'))

            if not pre_processed_data_path.exists():
                # preprocess and save results
                wrapper.preprocess_item_ehr_sepsis(item=processed_data, data_path=pre_processed_data_path)

    @property
    def raw_file_names(self):
        return self.all_graph_idx_files

    @property
    def processed_file_names(self):
        return self.all_graphs

    def get_train_val_test_split(self, rotation=0, label_ratio=1.0):
        # write a function that rotates through the numbers from 0-39 and returns the train, val, test split
        # train split needs to always have 32 elements, val and test split need to have 4 elements each
        if rotation == 0:
            train_idxs, val_idxs, test_idxs = [i for i in range(32)], [i for i in range(32, 36)], [i for i in range(36, 40)]
        elif rotation == 1:
            train_idxs, val_idxs, test_idxs = [i for i in range(8, 40)], [i for i in range(0, 4)], [i for i in range(4, 8)]
        elif rotation == 2:
            train_idxs, val_idxs, test_idxs = [i for i in range(16, 40)] + [i for i in range(0, 8)], [i for i in range(8, 12)], [i for i in
                                                                                                                                 range(12, 16)]
        elif rotation == 3:
            train_idxs, val_idxs, test_idxs = [i for i in range(24, 40)] + [i for i in range(0, 16)], [i for i in range(16, 20)], [i for i in
                                                                                                                                   range(20, 24)]
        elif rotation == 4:
            train_idxs, val_idxs, test_idxs = [i for i in range(32, 40)] + [i for i in range(0, 24)], [i for i in range(24, 28)], [i for i in
                                                                                                                                   range(28, 32)]

        if label_ratio != 1.0:
            # need to restrict the training set
            num_graphs = round(36 * label_ratio)  # 0.01 - 0, 0.05 - 2, 0.1 - 3, 0.5 - 16
            if label_ratio == 0.01:
                # need to create special graph because 1 graph has already 2.5 % of the data
                train_idxs = train_idxs[0:1]  # only the first graph and will need to drop some of its nodes
            else:
                train_idxs = train_idxs[:num_graphs]

        return train_idxs, val_idxs, test_idxs

    def get_or_compute_val_descriptor(self, idx, vals, descriptor_cache):
        if idx in descriptor_cache:
            return descriptor_cache[idx]
        else:
            descriptors = np.stack([vals.mean(axis=0), vals.std(axis=0), vals.min(axis=0)[0], vals.max(axis=0)[0]])
            descriptor_cache[idx] = descriptors
            return descriptors

    def compute_vals_similarity(self, vals_descriptors1, vals_descriptors2):
        # per column compute similarity of time-series
        dist = torch.tensor(np.linalg.norm(vals_descriptors1 - vals_descriptors2, axis=0).mean())
        return 1 - torch.sigmoid(dist)  # in which range is this value? in test graph: 0.01 - 0.42 for half features, 0.02 - 0.4 for full features

    def create_edges(self, all_vals):
        # edge_index
        # edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edges = []
        edge_features = []
        descriptor_cache = {}
        for idx1, vals in enumerate(all_vals):
            all_sims = {}
            if idx1 % 100 == 0:
                print(idx1)
            for idx2, vals2 in enumerate(all_vals):
                if idx1 != idx2:
                    vals_descriptors1 = self.get_or_compute_val_descriptor(idx1, vals, descriptor_cache)
                    vals_descriptors2 = self.get_or_compute_val_descriptor(idx2, vals2, descriptor_cache)
                    vals_similarity = self.compute_vals_similarity(vals_descriptors1, vals_descriptors2)
                    all_sims[idx2] = vals_similarity

            # compute 5 nearest neighbors and add edges
            sorted_sims = sorted(all_sims.items(), key=lambda x: x[1], reverse=True)
            for i in range(self.k):
                edges.append((idx1, sorted_sims[i][0]))
                edge_features.append(torch.round((sorted_sims[i][1] * 100)).int())  # discretize similarity

        edge_idx = torch.tensor(edges, dtype=torch.long)
        edge_features = torch.stack(edge_features)[:, None]
        return edge_idx, edge_features

    def save_data(self, node_id, vals, demographics, is_measured, edge_idx, edge_features, y, final_mask, train_mask, val_mask, test_mask, subset_id,
                  set_id):
        data = Data(node_id=node_id, vals=vals, demographics=demographics, is_measured=is_measured,
                    edge_index=edge_idx.t().contiguous(), edge_attr=edge_features, y=y, update_mask=final_mask, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask, split=self.split)

        print(f'Saving data to {osp.join(self.processed_dir, self.data_path + subset_id + "_" + set_id + ".pt")}')
        torch.save(data, osp.join(self.processed_dir, self.data_path + f'{subset_id}_{set_id}.pt'))

    def len(self):
        return len(self.processed_file_names)

    def patient_part_of_split(self, patient_split):
        if patient_split == 'train':
            return self.split == 'train' or self.split == 'val' or self.split == 'test'
        elif patient_split == 'val':
            return self.split == 'val' or self.split == 'test'
        elif patient_split == 'test':
            return self.split == 'test'

    def train_val_test_masks(self, splits):
        train_mask, val_mask, test_mask, = None, None, None
        if self.split == 'train':
            # get all elements in split where second element is 'train'
            train_data = [split for split in splits if split[1] == 'train']
            train_mask = torch.tensor([split[1] == 'train' for split in train_data])

        elif self.split == 'val':
            # get all elements in split where second element is 'val' or 'train'
            val_data = [split for split in splits if split[1] == 'val' or split[1] == 'train']
            train_mask = torch.tensor([split[1] == 'train' for split in val_data])
            val_mask = torch.tensor([split[1] == 'val' for split in val_data])

        elif self.split == 'test':
            # get all elements in split where second element is 'test' or 'train' or 'val'
            test_data = [split for split in splits if split[1] == 'test' or split[1] == 'train' or split[1] == 'val']
            test_mask = torch.tensor([split[1] == 'test' for split in test_data])

        return train_mask, val_mask, test_mask

    def process(self):
        if self.set_id == 'AB':
            set_ids = ['A', 'B']
        else:
            set_ids = [self.set_id]
        for my_set_id in set_ids:
            # load training set mean and std from json
            with open(f'data/sepsis/value_means{my_set_id}.json', 'r') as f:
                value_means = json.load(f)
            with open(f'data/sepsis/value_stds{my_set_id}.json', 'r') as f:
                value_stds = json.load(f)

            for raw_path in [path for path in self.raw_paths if
                             f"set{my_set_id}" in path]:  # [path for path in self.raw_paths if f'cross_val_{my_set_id}' in path]:
                subset_id = raw_path.split('.')[0].split('_')[-1]  # delete p in beginning
                set_id = raw_path.split('/')[-2][-1]  # get from folder name
                # Read data from `raw_path`.
                patient_icuids_and_splits = json.load(open(raw_path, 'r'))  # gives list of patients that should be in graph together with their split

                # Keep vals and treatments separate as we sometimes want treatments in training data and sometimes not
                patient_icuids = []
                patient_vals = []
                patient_dems = []
                patient_is_measured = []
                patient_ys = []  # dict for several prediction tasks
                # go through all patient folders with id in patient_icuids
                for info in patient_icuids_and_splits:
                    patient_icuid, split = info

                    if self.patient_part_of_split(split):
                        patient_file = osp.join(self.my_root,
                                                f'training_set{set_id}/p{patient_icuid}.psv')  # for other rotations than 0 the patient might not be saved in the splits folder, because all is saved according to rotation 0
                        data = pd.read_csv(patient_file, sep="|")
                        if 'Hour.1' in data.columns:
                            data = data.drop(columns=['Hour.1'])

                        patient_icuids.append(patient_icuid)

                        # extract vals and dems from data
                        vals = data.drop(
                            columns=['Hour', 'age', 'gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS', 'SepsisLabel', 'Patient_ID', 'sepsis_bin'])

                        # normalize vals and fill empty colomns with mean (0)
                        for col in vals.columns:
                            if vals[col].isnull().any():
                                assert vals[col].isnull().all()
                                vals[col] = 0.0  # normalized to N(0,1) -> all means are 0
                            else:
                                vals[col] = (np.array(vals[col]) - value_means[col]) / value_stds[col]

                        patient_vals.append(torch.tensor(vals.values.astype(np.float32)))

                        data[['Unit1', 'Unit2']] = data[['Unit1', 'Unit2']].fillna(2)  # if Unit is missing, fill with 2
                        data['HospAdmTime'] = data['HospAdmTime'].fillna(0)  # if HospAdmTime is missing, fill with 0 (mean)
                        dems = data[['age', 'gender', 'Unit1', 'Unit2', 'HospAdmTime']].values[0]
                        # assert that all demographics data stay same over hospital stay of patient
                        assert (dems == data[['age', 'gender', 'Unit1', 'Unit2', 'HospAdmTime']].max(axis=0).values).min()
                        # normalize age and HospAdmTime
                        dems[0] = (dems[0] - value_means['age']) / value_stds['age']
                        dems[4] = (dems[4] - value_means['HospAdmTime']) / value_stds['HospAdmTime']
                        patient_dems.append(torch.tensor(dems.astype(np.float32)))

                        is_measured = np.load(osp.join(self.my_root, f'training_set{set_id}/is_measuredp{patient_icuid}.npy'))
                        patient_is_measured.append(torch.tensor(is_measured.astype(np.float32)))

                        y_dict = {}
                        # extract labels from data
                        y_dict['y_bin'] = torch.tensor(data[['sepsis_bin']].values[0].astype(np.float32))
                        y_dict['y_cont'] = torch.tensor(data[['SepsisLabel']].values.astype(np.float32))

                        patient_ys.append(y_dict)

                train_mask, val_mask, test_mask = self.train_val_test_masks(patient_icuids_and_splits)

                edge_idx, edge_features = self.create_edges(all_vals=patient_vals)

                self.save_data(node_id=np.array(patient_icuids), vals=patient_vals, demographics=torch.stack(patient_dems),
                               is_measured=patient_is_measured, edge_idx=edge_idx, edge_features=edge_features, y=patient_ys, final_mask=None,
                               train_mask=train_mask, val_mask=val_mask, test_mask=test_mask, subset_id=subset_id, set_id=set_id)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, self.all_pre_processed_graphs[idx]))
        data.train_mask = torch.ones_like(data.y).bool()
        data.val_mask = torch.ones_like(data.y).bool()
        data.test_mask = torch.ones_like(data.y).bool()
        padding_mask = None
        data.padding_mask = padding_mask
        return data


if __name__ == '__main__':
    pass
