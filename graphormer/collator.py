# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# code was adapted for our data
import random

import numpy as np
import torch


def pad_1d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_y_unsqueeze(x, padlen):
    # pad id = -1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype) - 1
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_mask_unsqueeze(x, padlen):
    # pad id = 0 -> will not be used for train/val/test, same as normal False nodes
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_pad_mask_unsqueeze(x, padlen):
    # pad id = 0 -> will not be used for train/val/test, same as normal False nodes
    x1, x2 = x.size()
    if x1 < padlen:
        new_x = x.new_zeros([padlen, x2], dtype=x.dtype)
        new_x[:x1] = x
        x = new_x
    return x.unsqueeze(0)


def pad_2d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_feat_graph_unsqueeze(x, padlen1, padlen2, padlen3, pad_mode):
    if pad_mode == 'original' or pad_mode == 'emb':
        x = x + 1
    xlen1, xlen2, xlen3 = x.size()
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = x.new_zeros([padlen1, padlen2, padlen3], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :xlen3] = x
        x = new_x
    return x.unsqueeze(0)


def convert_to_single_emb(x, offset=512):
    feature_num = x.size(2)
    feature_offset = 1 + \
                     torch.arange(0, feature_num * offset, offset, dtype=torch.int32)
    x = x + feature_offset
    return x


def pad_treat_unsqueeze(x, padlen1, padlen2, padlen3, ):
    xlen1, xlen2, xlen3 = x.size()
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = x.new_zeros([padlen1, padlen2, padlen3], dtype=x.dtype) + 2  # padded elements are 2, classes 0 and 1
        new_x[:xlen1, :xlen2, :xlen3] = x
        x = new_x
    x = convert_to_single_emb(x, offset=3)  # after this 0 will be free for masking
    return x.unsqueeze(0)


def pad_y_graph_unsqueeze(x, padlen1, padlen2, padlen3):
    xlen1, xlen2, xlen3 = x.size()
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = x.new_zeros([padlen1, padlen2, padlen3], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :xlen3] = x
        x = new_x
    return x.unsqueeze(0)


def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros(
            [padlen, padlen], dtype=x.dtype).fill_(float('-inf'))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)


def pad_edge_type_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, x.size(-1)], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_spatial_pos_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_3d_unsqueeze(x, padlen1, padlen2, padlen3):
    x = x + 1
    xlen1, xlen2, xlen3, xlen4 = x.size()
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = x.new_zeros([padlen1, padlen2, padlen3, xlen4], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :xlen3, :] = x
        x = new_x
    return x.unsqueeze(0)


class Batch():
    def __init__(self, idx, y, update_mask, train_mask, val_mask, test_mask, node_id, attn_bias=None, attn_edge_type=None, spatial_pos=None,
                 in_degree=None, out_degree=None, edge_input=None, x=None, vals=None, treatments=None, demographics=None, is_measured=None,
                 edge_index=None,
                 dev_mask=None, train_dev_mask=None, attn_mask=None, padding_mask=None, mask_task=None, adj=None):
        super(Batch, self).__init__()
        self.idx = idx
        self.in_degree, self.out_degree = in_degree, out_degree
        self.x = x
        self.attn_mask = attn_mask
        # mimic
        self.vals = vals
        self.treatments = treatments
        self.is_measured = is_measured
        self.demographics = demographics
        self.y, self.node_id = y, node_id
        self.attn_bias, self.attn_edge_type, self.spatial_pos = attn_bias, attn_edge_type, spatial_pos
        self.edge_input = edge_input
        self.update_mask = update_mask
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.dev_mask = dev_mask
        self.train_dev_mask = train_dev_mask
        self.edge_index = edge_index
        self.padding_mask = padding_mask
        self.mask_task = mask_task
        self.adj = adj

    def to(self, device):
        self.idx = self.idx.to(device)
        if self.update_mask is not None:  # None for classification task
            if type(self.update_mask) == list:
                self.update_mask[0] = self.update_mask[0].to(device)
                self.update_mask[1] = self.update_mask[1].to(device)
            else:
                self.update_mask = self.update_mask.to(device)
        self.train_mask = self.train_mask.to(device) if self.train_mask is not None else None
        self.val_mask = self.val_mask.to(device) if self.val_mask is not None else None
        self.test_mask = self.test_mask.to(device) if self.test_mask is not None else None
        self.dev_mask = self.dev_mask.to(device) if self.dev_mask is not None else None
        self.train_dev_mask = self.train_dev_mask.to(device) if self.train_dev_mask is not None else None
        if self.x is not None:
            self.x = self.x.to(device)
        if self.attn_mask is not None:
            self.attn_mask = self.attn_mask.to(device)
        if self.vals is not None:
            if type(self.vals) == tuple:  # sepsis: for mimic it's already a tensor, for sepsis not
                self.vals = [[elem.to(device) for elem in graph] for graph in self.vals]
                self.demographics = self.demographics.to(device)
            else:  # mimic
                self.vals = self.vals.to(device)
                self.treatments = self.treatments.to(device)
                self.is_measured = self.is_measured.to(device)
                self.demographics = self.demographics.to(device)
        if type(self.y) == list:
            self.y[0] = self.y[0].to(device)
            self.y[1] = self.y[1].to(device)
        elif type(self.y) == tuple:
            self.y = (self.y[0][0].to(device), self.y[0][1].to(device))
        else:
            self.y = self.y.to(device)
        if self.attn_edge_type is not None and self.edge_input is not None:
            self.attn_edge_type = self.attn_edge_type.to(device)
            self.edge_input = self.edge_input.to(device)
        if self.attn_bias is not None:
            self.attn_bias, self.spatial_pos = self.attn_bias.to(
                device), self.spatial_pos.to(device)
            self.in_degree, self.out_degree = self.in_degree.to(
                device), self.out_degree.to(device)
        if self.edge_index is not None:
            if type(self.edge_index) == list:
                self.edge_index = [elem.to(device) for elem in self.edge_index]
            else:
                self.edge_index = self.edge_index.to(device)
        if self.padding_mask is not None:
            self.padding_mask = self.padding_mask.to(device)
        if self.adj:
            self.adj = [elem.to(device) for elem in self.adj]
        return self

    def __len__(self):
        if type(self.y) == list:
            return self.y[0].size(0)
        else:
            return self.y.size(0)


def collator(items, max_node=512, multi_hop_max_dist=20, spatial_pos_max=20, dataset='tadpole', gcn=False, pad_mode='original'):
    if dataset == 'mimic':
        if gcn:
            mask_task = None
            items = [item for item in items if item is not None]  # and item.vals.size(0) <= max_node]
            items = [(item.idx, item.vals, item.treatments, item.edge_index,
                      item.demographics, item.is_measured, item.node_id, item.y,
                      item.update_mask, item.train_mask, item.val_mask, item.test_mask) for item in items]
            idxs, valss, treatmentss, edge_indexs, demographicss, is_measureds, node_ids, \
            ys, update_masks, train_masks, val_masks, test_masks = zip(*items)
            dev_masks, train_dev_masks = [None], [None]
            edge_indexs = torch.cat(edge_indexs, dim=1)  # if multiple graphs combine all edge_indexs to one array for batch processing
        else:
            items = [item for item in items if item is not None]  # and item.vals.size(0) <= max_node]
            mask_task = None
            if type(items[0]) == tuple:  # multi-task pre-training -> select mask task for whole batch
                pre_mask_items, TP_items, PM_items, BM_items = zip(*items)
                mask_task = random.choice(["pre_mask", "PM", "BM", "TP"])
                if mask_task == "pre_mask":
                    items = pre_mask_items
                elif mask_task == "TP":
                    items = TP_items
                elif mask_task == "PM":
                    items = PM_items
                elif mask_task == "BM":
                    items = BM_items

            items = [(item.adj, item.idx, item.attn_bias, item.attn_edge_type, item.spatial_pos, item.in_degree, item.out_degree, item.vals,
                      item.treatments,
                      item.demographics, item.is_measured, item.node_id, item.edge_input[:, :, :multi_hop_max_dist, :], item.y,
                      item.update_mask, item.train_mask, item.val_mask, item.test_mask, item.padding_mask) for item in
                     items]  # , item.edge_index
            adjs, idxs, attn_biases, attn_edge_types, spatial_poses, in_degrees, out_degrees, valss, treatmentss, demographicss, is_measureds, node_ids, \
            edge_inputs, ys, update_masks, train_masks, val_masks, test_masks, padding_masks = zip(*items)
            dev_masks, train_dev_masks = [None], [None]
    elif dataset == 'sepsis':
        items = [item for item in items if item is not None]  # and item.vals.size(0) <= max_node]

        items = [(item.adj, item.idx, item.attn_bias, item.attn_edge_type, item.spatial_pos, item.in_degree, item.out_degree, item.vals,
                  item.demographics, item.is_measured, item.node_id, item.edge_input[:, :, :multi_hop_max_dist, :], item.y,
                  item.update_mask, item.train_mask, item.val_mask, item.test_mask) for item in
                 items]  # , item.edge_index
        adjs, idxs, attn_biases, attn_edge_types, spatial_poses, in_degrees, out_degrees, valss, demographicss, is_measureds, node_ids, \
        edge_inputs, ys, update_masks, train_masks, val_masks, test_masks = zip(*items)

    else:
        items = [item for item in items if item is not None and item.x.size(0) <= max_node]
        items = [(item.adj, item.idx, item.attn_bias, item.attn_edge_type, item.spatial_pos, item.in_degree,
                  item.out_degree, item.x, torch.tensor(item.node_id), item.edge_input[:, :, :multi_hop_max_dist, :], item.y, item.update_mask,
                  item.train_mask, item.val_mask, item.test_mask) for item in items]
        adjs, idxs, attn_biases, attn_edge_types, spatial_poses, in_degrees, out_degrees, xs, node_ids, edge_inputs, ys, update_masks, train_masks, val_masks, test_masks \
            = zip(*items)

    if not gcn:
        for idx, _ in enumerate(attn_biases):
            attn_biases[idx][1:, 1:][spatial_poses[idx] >= spatial_pos_max] = float('-inf')
        max_dist = max(i.size(-2) for i in edge_inputs)
    if dataset == 'sepsis':
        max_node_num = max(len(i) for i in valss)  # still as list
        max_hour_num = max(max([p.size(0) for p in i]) for i in valss)
    else:
        max_node_num = max(i.size(0) for i in valss) if dataset == 'mimic' else max(i.size(0) for i in xs)
        max_hour_num = max(i.size(1) for i in valss) if dataset == 'mimic' else None

    if dataset == 'mimic':  # want to do batching
        val_feature_num = valss[0].shape[2]
        if type(ys[0]) == tuple:  # for masking we have two ys for val and treatments
            # check pre_mask type to form y
            if mask_task is not None:
                vals_y, treat_y, treat_bin_y = zip(*ys)
                if mask_task == 'TP':
                    y = torch.cat([pad_pad_mask_unsqueeze(i, max_node_num) for i in treat_bin_y])
                else:
                    y = [torch.cat([pad_y_graph_unsqueeze(i, max_node_num, max_hour_num, vals_y[0].shape[2]) for i in vals_y]),
                         torch.cat([pad_y_graph_unsqueeze(i, max_node_num, max_hour_num, 16) for i in treat_y])]
            else:
                vals_y, treat_y, class_y = zip(*ys)
                y = [torch.cat([pad_y_graph_unsqueeze(i, max_node_num, max_hour_num, vals_y[0].shape[2]) for i in vals_y]),
                     torch.cat([pad_y_graph_unsqueeze(i, max_node_num, max_hour_num, 16) for i in treat_y]),
                     torch.cat([pad_y_unsqueeze(i, max_node_num) for i in class_y])]
        else:
            if len(ys[0].shape) == 1:
                y = torch.cat([pad_y_unsqueeze(i, max_node_num) for i in
                               ys])  # padded ys are set to -1 just to make sure an error would occur if they are used (no binary classification anymore)
            else:
                y = torch.cat([pad_pad_mask_unsqueeze(i, max_node_num) for i in ys])

    elif dataset == 'sepsis':
        val_feature_num = valss[0][0].shape[1]
        y = torch.cat([pad_y_unsqueeze(i, max_node_num) for i in ys])
    else:
        y = torch.cat(ys)
    if dataset == 'sepsis':
        node_id = torch.cat([pad_y_unsqueeze(torch.tensor(i.astype(np.int)), max_node_num) for i in node_ids])
    else:
        node_id = torch.cat([pad_y_unsqueeze(i, max_node_num) for i in node_ids])

    if dataset == 'mimic':
        vals = torch.cat(
            [pad_feat_graph_unsqueeze(i, max_node_num, max_hour_num, val_feature_num, pad_mode) for i in valss])  # max_num_hours, num_features
        is_measured = torch.cat([pad_feat_graph_unsqueeze(i, max_node_num, max_hour_num, 56, pad_mode) for i in is_measureds])

        demographics = torch.cat([pad_2d_unsqueeze(i.squeeze(1), max_node_num) for i in demographicss])  # could also delete +1 but do not need to

        if pad_mode == 'pad_emb' or pad_mode == 'emb':
            treatments = torch.cat([pad_treat_unsqueeze(i, max_node_num, max_hour_num, 16) for i in treatmentss])
        else:
            treatments = torch.cat([pad_feat_graph_unsqueeze(i, max_node_num, max_hour_num, 16, pad_mode) for i in treatmentss])



    elif dataset == 'sepsis':
        # vals need to be padded after transformer layer + mean because if the different stay lengths
        demographics = torch.cat([pad_2d_unsqueeze(i.squeeze(1), max_node_num) for i in demographicss])

    else:
        x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])
    if update_masks[0] is not None:  # None for classification task
        if type(update_masks[0]) == list:  # in mimic we have two update masks for val and treatments
            vals_update_masks, treat_update_masks = zip(*update_masks)
            update_mask = [torch.cat([pad_y_graph_unsqueeze(i.squeeze(), max_node_num, max_hour_num, 56) for i in vals_update_masks]),
                           torch.cat([pad_y_graph_unsqueeze(i.squeeze(), max_node_num, max_hour_num, 16) for i in treat_update_masks])]
        else:
            update_mask = torch.cat(update_masks)
    else:
        update_mask = None
    train_mask = torch.cat([pad_mask_unsqueeze(i, max_node_num) for i in train_masks]) if train_masks[0] is not None else None
    val_mask = torch.cat([pad_mask_unsqueeze(i, max_node_num) for i in val_masks]) if val_masks[0] is not None else None
    test_mask = torch.cat([pad_mask_unsqueeze(i, max_node_num) for i in test_masks]) if test_masks[0] is not None else None

    if dataset == 'mimic':
        dev_mask = torch.cat([pad_mask_unsqueeze(i, max_node_num) for i in dev_masks]) if dev_masks[0] is not None else None
        train_dev_mask = torch.cat([pad_mask_unsqueeze(i, max_node_num) for i in train_dev_masks]) if train_dev_masks[0] is not None else None
        if not gcn:
            padding_mask = torch.cat([pad_pad_mask_unsqueeze(i, max_node_num) for i in padding_masks]) if padding_masks[0] is not None else None
    if not gcn:
        # these are Graphormer related inputs, they are all processed by an embedding layer so they can work with the regular +1 padding
        edge_input = torch.cat([pad_3d_unsqueeze(
            i, max_node_num, max_node_num, max_dist) for i in edge_inputs])
        attn_bias = torch.cat([pad_attn_bias_unsqueeze(
            i, max_node_num + 1) for i in attn_biases])
        attn_edge_type = torch.cat(
            [pad_edge_type_unsqueeze(i, max_node_num) for i in attn_edge_types])
        spatial_pos = torch.cat([pad_spatial_pos_unsqueeze(i, max_node_num)
                                 for i in spatial_poses])
        in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num)
                               for i in in_degrees])
        out_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num)
                                for i in out_degrees])

    if dataset == 'mimic':
        if gcn:
            return Batch(
                idx=torch.LongTensor(idxs),
                vals=vals,
                treatments=treatments,
                demographics=demographics,
                is_measured=is_measured,
                edge_index=edge_indexs,
                y=y,
                update_mask=update_mask,
                train_mask=train_mask,
                val_mask=val_mask,
                test_mask=test_mask,
                node_id=node_id
            )
        else:
            return Batch(
                idx=torch.LongTensor(idxs),
                attn_bias=attn_bias,
                attn_edge_type=attn_edge_type,
                spatial_pos=spatial_pos,
                in_degree=in_degree,
                out_degree=out_degree,
                vals=vals,
                treatments=treatments,
                demographics=demographics,
                is_measured=is_measured,
                edge_input=edge_input,
                y=y,
                update_mask=update_mask,
                train_mask=train_mask,
                val_mask=val_mask,
                test_mask=test_mask,
                dev_mask=dev_mask,
                train_dev_mask=train_dev_mask,
                node_id=node_id,
                padding_mask=padding_mask,
                mask_task=mask_task,
                adj=adjs
                # attn_mask = attn_mask
            )
    elif dataset == 'sepsis':
        return Batch(
            idx=torch.LongTensor(idxs),
            attn_bias=attn_bias,
            attn_edge_type=attn_edge_type,
            spatial_pos=spatial_pos,
            in_degree=in_degree,
            out_degree=out_degree,
            vals=valss,
            demographics=demographics,
            is_measured=None,
            edge_input=edge_input,
            y=y,
            update_mask=update_mask,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            node_id=node_id,
            adj=adjs
        )
    else:
        return Batch(
            idx=torch.LongTensor(idxs),
            attn_bias=attn_bias,
            attn_edge_type=attn_edge_type,
            spatial_pos=spatial_pos,
            in_degree=in_degree,
            out_degree=out_degree,
            x=x,
            edge_input=edge_input,
            y=y,
            update_mask=update_mask,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            node_id=node_id,
            adj=adjs
        )
