# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import pyximport
import torch

from graphormer.EHR_dataset import TadpoleDataset, MIMICDataset, SepsisDataset
from graphormer.utils.mask_utils import create_treat_mask_mimic, create_vals_mask_mimic, create_patient_mask_mimic

pyximport.install(setup_args={'include_dirs': np.get_include()})
from graphormer import algos
import time


def convert_to_single_emb(x, offset=512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + \
                     torch.arange(0, feature_num * offset, offset, dtype=torch.int32)
    x = x + feature_offset
    return x


def preprocess_item_ehr(item, offset=512, edges=False, data_path=None, bin_split_idx=0, one_node=False):
    start_p = time.time()
    edge_attr, edge_index, x, node_id = item.edge_attr, item.edge_index, item.x, item.node_id
    orig_x = x.clone()
    print("num nodes: ", x.shape[0])
    print("num edges: ", len(edge_attr))
    N = x.size(0)
    bin_feat = x[:, :bin_split_idx]
    reg_feat = x[:, bin_split_idx:]

    bin_feat = convert_to_single_emb(bin_feat, offset)
    x = torch.cat([bin_feat, reg_feat], dim=1)

    # node adj matrix [N, N] bool
    adj = torch.zeros([N, N], dtype=torch.bool)
    if not one_node:
        adj[edge_index[0, :], edge_index[1, :]] = True

    if edges:
        # edge feature here
        if len(edge_attr.size()) == 1:
            edge_attr = edge_attr[:, None]
        attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.int)
        if not one_node:
            attn_edge_type[edge_index[0, :], edge_index[1, :]] = convert_to_single_emb(edge_attr, offset=3) + 1

    start = time.time()
    shortest_path_result, path = algos.floyd_warshall(adj.numpy())
    shortest_path_result = shortest_path_result.astype(np.int32)
    print("floyd_warshall: ", time.time() - start)
    if edges:
        max_dist = np.amax(shortest_path_result)
        start = time.time()
        edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
        print("gen_edge_input: ", time.time() - start)
    spatial_pos = torch.from_numpy((shortest_path_result)).long()
    attn_bias = torch.zeros(
        [N + 1, N + 1], dtype=torch.float)  # with graph token

    # combine
    item.x = x
    item.node_id = node_id
    item.orig_x = orig_x
    item.adj = adj
    item.attn_bias = attn_bias
    item.spatial_pos = spatial_pos
    item.in_degree = adj.long().sum(dim=1).view(-1)
    item.out_degree = adj.long().sum(dim=0).view(-1)
    if edges:
        item.edge_input = torch.from_numpy(edge_input).long()
        item.attn_edge_type = attn_edge_type
    print('full preprocessing: ', time.time() - start_p)
    if data_path and not one_node:
        torch.save(item, data_path)
    return item


def preprocess_item_ehr_mimic(item, data_path=None, edge_vars='age'):
    start_p = time.time()
    edge_attr, edge_index, vals, demographics, treatments = item.edge_attr, item.edge_index, item.vals, item.demographics, item.treatments
    if len(edge_attr.shape) == 1:
        edge_attr = edge_attr[:, None]
    if len(edge_attr) == 0 and len(edge_index) != 0:  # do not do for "no edge" case
        edge_attr = torch.ones([len(edge_index[1]), 1])  # for random edges include dummy edge features of 1
    print("num nodes: ", len(vals))
    print("num edges: ", len(edge_attr))
    N = len(vals)

    # add masking column for each feature column in vals
    for i, val in enumerate(vals):
        val_new = torch.zeros(val.shape[0], val.shape[1] * 2)
        for idx in range(val.shape[1]):
            val_new[:, idx * 2] = val[:, idx]
        vals[i] = val_new

    dem_embs = []
    for d in demographics:
        dem_embs.append(torch.cat([d[:, 0:1], convert_to_single_emb(d[:, 1:], offset=6)], dim=1))

    # node adj matrix [N, N] bool
    adj = torch.zeros([N, N], dtype=torch.bool)
    if len(edge_index) > 0:
        adj[edge_index[0, :], edge_index[1, :]] = True

    if len(edge_index) > 0:
        attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.int)
        attn_edge_type[edge_index[0, :], edge_index[1, :]] = convert_to_single_emb(edge_attr, offset=3 if edge_vars == 'age' else 200).int() + 1
    else:
        attn_edge_type = torch.zeros([N, N, 1], dtype=torch.int)

    start = time.time()
    shortest_path_result, path = algos.floyd_warshall(adj.numpy())
    shortest_path_result = shortest_path_result.astype(np.int32)
    print("floyd_warshall: ", time.time() - start)

    max_dist = np.amax(shortest_path_result)
    start = time.time()
    edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
    print("gen_edge_input: ", time.time() - start)

    spatial_pos = torch.from_numpy((shortest_path_result)).long()
    attn_bias = torch.zeros(
        [N + 1, N + 1], dtype=torch.float)  # with graph token

    # combine
    item.adj = adj
    item.demographics = torch.stack(dem_embs)
    item.attn_bias = attn_bias
    item.spatial_pos = spatial_pos
    item.in_degree = adj.long().sum(dim=1).view(-1)
    item.out_degree = adj.long().sum(dim=0).view(-1)
    item.edge_input = torch.from_numpy(edge_input).long()
    item.attn_edge_type = attn_edge_type
    item.vals = vals
    print('full preprocessing: ', time.time() - start_p)
    torch.save(item, data_path)


def preprocess_item_ehr_sepsis(item, data_path):
    start_p = time.time()
    edge_attr, edge_index, vals, demographics = item.edge_attr, item.edge_index, item.vals, item.demographics
    if len(edge_attr.shape) == 1:
        edge_attr = edge_attr[:, None]
    print("num nodes: ", len(vals))
    print("num edges: ", len(edge_attr))
    N = len(vals)

    # add masking column for each feature column in vals
    for i, val in enumerate(vals):
        val_new = torch.zeros(val.shape[0], val.shape[1] * 2)
        for idx in range(val.shape[1]):
            val_new[:, idx * 2] = val[:, idx]
        vals[i] = val_new

    dem_embs = []
    for d in demographics:
        dem_embs.append(torch.cat([d[0:1], convert_to_single_emb(d[1:-1], offset=3),
                                   d[-1:]]))  # convert gender, unit1 and unit2 to single embeddings, keep age and admTime as is

    # get sepsis starts and cut data to 24 hours

    sepsis_bin = torch.tensor([d['y_bin'] for d in item.y])
    sepsis_starts = torch.tensor(
        [d['y_cont'].argmax().item() + 6 if sepsis_bin[idx] == 1.  # for sepsis patients cut off directly before actual sepsis would start
         else torch.randint(low=6, high=len(d['y_cont']), size=(1,)) for idx, d in
         enumerate(item.y)])  # for non-sepsis patients, cut off at random time

    vals = [d[max(0, sepsis_starts[idx] - 24):sepsis_starts[idx]] for idx, d in enumerate(vals)]
    assert len([d for d in vals if
                len(d) == 0]) == 0  # should never happen as we use the first 6 hours of sepsis label = 1 and for others cut off not before 6 hours

    # node adj matrix [N, N] bool
    adj = torch.zeros([N, N], dtype=torch.bool)
    if len(edge_index) > 0:
        adj[edge_index[0, :], edge_index[1, :]] = True

    if len(edge_index) > 0:
        attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.int)
        attn_edge_type[edge_index[0, :], edge_index[1, :]] = convert_to_single_emb(edge_attr, offset=200).int() + 1
    else:
        attn_edge_type = torch.zeros([N, N, 1], dtype=torch.int)

    start = time.time()
    shortest_path_result, path = algos.floyd_warshall(adj.numpy())
    shortest_path_result = shortest_path_result.astype(np.int32)
    print("floyd_warshall: ", time.time() - start)

    max_dist = np.amax(shortest_path_result)
    start = time.time()
    edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
    print("gen_edge_input: ", time.time() - start)

    spatial_pos = torch.from_numpy((shortest_path_result)).long()
    attn_bias = torch.zeros(
        [N + 1, N + 1], dtype=torch.float)  # with graph token

    # combine
    item.adj = adj
    item.demographics = torch.stack(dem_embs)
    item.attn_bias = attn_bias
    item.spatial_pos = spatial_pos
    item.in_degree = adj.long().sum(dim=1).view(-1)
    item.out_degree = adj.long().sum(dim=0).view(-1)
    item.edge_input = torch.from_numpy(edge_input).long()
    item.attn_edge_type = attn_edge_type
    item.vals = vals
    item.y = sepsis_bin
    print('full preprocessing: ', time.time() - start_p)
    torch.save(item, data_path)


def preprocess_item_mimic_gcn(item):
    edge_index, vals, demographics = item.edge_index, item.vals, item.demographics
    N = len(vals)

    # add masking column for each feature column in vals
    for i, val in enumerate(vals):
        val_new = torch.zeros(val.shape[0], val.shape[1] * 2)
        for idx in range(val.shape[1]):
            val_new[:, idx * 2] = val[:, idx]
        vals[i] = val_new

    dem_embs = []
    for d in demographics:
        dem_embs.append(torch.cat([d[:, 0:1], convert_to_single_emb(d[:, 1:], offset=6)], dim=1))

    # combine
    item.demographics = torch.stack(dem_embs)
    item.edge_index = edge_index
    item.vals = vals
    return item


def add_masking(item, mask_ratio=0.1):
    # mask further features with masked value in input and create update mask only true for masked values, do not mask missing values
    # use 0.0 and not 95 for continuous masked features (as linear layer can not deal with such a different value)
    missing_mask_disc = (item.orig_x[:, 1:6] == item.mask_value)
    missing_mask_cont = (item.orig_x[:, 7:] == item.mask_value)
    item.x[:, 7:][missing_mask_cont] = -1.0  # make recognizable
    mask = torch.rand_like(item.x)
    mask = (mask < torch.tensor([mask_ratio])).bool()
    if item.not_mask_column_indices:
        item.not_mask_column_indices = [0, 6]  # only age and gender
        mask[:, item.not_mask_column_indices] = False

    item.x[:, 1:6][mask[:, 1:6]] = item.mask_value  # mask discrete features with mask values
    item.x[:, 7:][mask[:, 7:]] = -1.0  # make recognizable

    all_masked_disc = (item.x[:, 1:6] == item.mask_value)
    all_masked_cont = (item.x[:, 7:] == -1.0)
    final_mask_disc = torch.logical_and(all_masked_disc, ~missing_mask_disc)  # only values that were not missing before already
    final_mask_cont = torch.logical_and(all_masked_cont, ~missing_mask_cont)  # only values that were not missing before already

    # set missing continous values to 0
    item.x[:, 7:][all_masked_cont] = 0.0
    mask[:, 1:6] = final_mask_disc
    mask[:, 7:] = final_mask_cont
    item.update_mask = mask
    return item


def add_masking_mimic(item, mask_ratio=0.15, block_size=6):
    # currently we mask 10% of the columns with 6 hour blocks

    # randomly select 10% of columns in vals and treatments to have missing blocks
    binary_treatment_mask = create_treat_mask_mimic(item, mask_ratio=mask_ratio, block_size=block_size)
    item.treatments[binary_treatment_mask] = 0  # after convert_to_single_embedding treatment values are never 0

    # for measurements
    binary_vals_mask_big, binary_mask_mask_big, final_vals_mask = create_vals_mask_mimic(item, mask_ratio=mask_ratio, block_size=block_size)
    # set masked values to 0 and corresponding is_masked column to 1

    item.vals[binary_vals_mask_big] = 0  # continuous values, no exact 0 values exist
    item.vals[binary_mask_mask_big] = 1  # mark as masked
    # only values that were not missing before already
    item.update_mask = [final_vals_mask[None], torch.from_numpy(binary_treatment_mask)[None]]
    return item


def add_treatment_masking_mimic(item):
    # set all treatments to 0 (last 2 columns DNR and CMO are no "treatments")
    item.treatments[:, :, :14] = 0
    return item


def drop_patients_tadpole(item, mask_ratio):
    missing_mask_disc = (item.orig_x[:, 1:6] == item.mask_value)
    missing_mask_cont = (item.orig_x[:, 7:] == item.mask_value)
    item.x[:, 7:][missing_mask_cont] = -1.0  # make recognizable

    # make sure to drop items in train and validation set if split=val
    if item.split == 'val':
        # get idxs where item.val_mask is True
        val_idxs = item.val_mask.nonzero().squeeze()
        # get idxs where item.val_mask is False
        train_idxs = (item.val_mask == False).nonzero().squeeze()
        drop_idxs = np.concatenate((np.random.choice(train_idxs, size=int(mask_ratio * len(train_idxs)), replace=False),
                                    np.random.choice(val_idxs, size=int(mask_ratio * len(val_idxs)), replace=False)))
    else:  # train split, no validation nodes exist
        train_idxs = item.train_mask.nonzero().squeeze()
        drop_idxs = np.random.choice(train_idxs, size=int(mask_ratio * len(train_idxs)), replace=False)  # which patients to mask
    mask = torch.zeros_like(item.x, dtype=torch.bool)
    mask[drop_idxs] = True
    mask[:, [0, 6]] = False
    # for test, only mask half of values
    # set some patient nodes entirely to zero to augment the graph
    item.x[:, 1:6][mask[:, 1:6]] = item.mask_value  # mask discrete features with mask values
    item.x[:, 7:][mask[:, 7:]] = -1.0  # make recognizable
    # column 0 and 6 are gender and age, we do not want to mask them

    all_masked_disc = (item.x[:, 1:6] == item.mask_value)
    all_masked_cont = (item.x[:, 7:] == -1.0)
    final_mask_disc = torch.logical_and(all_masked_disc, ~missing_mask_disc)  # only values that were not missing before already
    final_mask_cont = torch.logical_and(all_masked_cont, ~missing_mask_cont)  # only values that were not missing before already

    # set missing continous values to 0
    item.x[:, 7:][all_masked_cont] = 0.0
    mask[:, 1:6] = final_mask_disc
    mask[:, 7:] = final_mask_cont
    item.update_mask = mask
    return item


def drop_patients_mimic(item, mask_ratio):
    # set some patient nodes entirely to zero to augment the graph
    binary_treatment_mask, binary_vals_mask_big, binary_mask_mask_big, final_vals_mask = create_patient_mask_mimic(item, mask_ratio=mask_ratio)
    item.treatments[binary_treatment_mask] = 0

    # set masked values to 0 and corresponding is_masked column to 1
    item.vals[binary_vals_mask_big] = 0  # continuous values, no exact 0 values exist
    item.vals[binary_mask_mask_big] = 1  # mark as masked
    # only values that were not missing before already
    item.update_mask = [final_vals_mask[None], torch.from_numpy(binary_treatment_mask)[None]]
    return item


class MyTadpoleDataset(TadpoleDataset):
    def process(self):
        super(MyTadpoleDataset, self).process()

    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.get(self.indices()[idx]).clone()
            item.idx = idx
            if self.mask:
                if self.task == "patient_prediction":
                    item = drop_patients_tadpole(item, mask_ratio=self.mask_ratio)
                else:
                    item = add_masking(item, mask_ratio=self.mask_ratio)
            # if item.split == 'train':
            #     item = drop_patients(item, rate=0.1, mask=self.mask)
            return item
        else:
            return self.index_select(idx)


class MyTadpoleTestDataset(TadpoleDataset):
    def process(self):
        super(MyTadpoleTestDataset, self).process()

    # return unmasked item, so in eval_step we can sample 100 fixed masks
    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.get(self.indices()[idx]).clone()
            item.idx = idx
            return item
        else:
            return self.index_select(idx)


class MyMIMICDataset(MIMICDataset):
    def process(self):
        super(MyMIMICDataset, self).process()

    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.get(self.indices()[idx]).clone()
            item.idx = idx
            if self.task.startswith('pre') and item.predict == False:
                item = add_masking_mimic(item, mask_ratio=self.mask_ratio, block_size=self.block_size)
            elif self.task == "treat_prediction":
                item = add_treatment_masking_mimic(item)
            elif self.task == "patient_prediction":
                item = drop_patients_mimic(item, mask_ratio=self.mask_ratio)
            elif self.task == "multi_task_pt":
                # compute all masks, later select one type per batch
                item_pre_mask = add_masking_mimic(item.clone(), mask_ratio=0.3, block_size=24)
                item_TP = add_treatment_masking_mimic(item.clone())
                item_PM = drop_patients_mimic(item.clone(), mask_ratio=0.1)
                item_BM = add_masking_mimic(item.clone(), mask_ratio=1.0, block_size=6)
                item = (item_pre_mask, item_TP, item_PM, item_BM)
            return item
        else:
            return self.index_select(idx)


class MySepsisDataset(SepsisDataset):
    def process(self):
        super(MySepsisDataset, self).process()

    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.get(self.indices()[idx]).clone()
            item.idx = idx
            return item
        else:
            return self.index_select(idx)
