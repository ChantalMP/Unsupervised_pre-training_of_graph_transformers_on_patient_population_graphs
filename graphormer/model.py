# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# import json
import math
from collections import defaultdict

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, mean_squared_error, r2_score
from torch.nn import functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv
from transformers import BertConfig, BertModel

from graphormer.data import get_dataset
from graphormer.lr import PolynomialDecayLR


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class Graphormer(pl.LightningModule):

    def __init__(
            self,
            gcn,
            n_layers,
            num_heads,
            hidden_dim,
            dropout_rate,
            intput_dropout_rate,
            weight_decay,
            ffn_dim,
            dataset_name,
            task,
            edge_vars,
            warmup_updates,
            tot_updates,
            cross_val_split,
            label_ratio,
            peak_lr,
            end_lr,
            edge_type,
            multi_hop_max_dist,
            attention_dropout_rate,
            mlp=False,
            gat=False,
            graphsage=False,
            gin=False,
            skip_transformer=False,
            flag=False,
            flag_m=3,
            flag_step_size=1e-3,
            flag_mag=1e-3,
            run_name='debug',
            pad_mode='original',
            rotation=0,
            loss_weighting='none',
            compute_results=False,
            mask_all_tadpole=False,
            use_simple_acu_task=False,
            fold=2,
            set_id='A',
    ):
        super().__init__()
        self.save_hyperparameters()

        self.compute_results = compute_results
        self.gcn = gcn
        self.mlp = mlp
        self.gat = gat
        self.graphsage = graphsage
        self.gin = gin
        self.skip_transformer = skip_transformer
        self.num_heads = num_heads
        self.run_name = run_name
        self.rotation = rotation
        self.mask_all_tadpole = mask_all_tadpole
        self.use_simple_acu_task = use_simple_acu_task
        if cross_val_split is not None:
            self.train_losses = []
            self.train_acc = []
            self.val_losses = []
            self.train_f1 = []
            self.train_auroc = []
            self.val_f1 = []
            self.val_acc = []
            self.val_auroc = []

        if dataset_name == 'tadpole' or dataset_name == 'tadpole_class':
            # split binary and regression features
            self.acc_dict = {}
            self.best_val_score_tadpole = 0.
            self.acc_accumulation = 10
            self.acc_accumulation_count = 0
            self.log_weights_count = 0
            self.log_weights_freq = 10
            self.test_masks = []
            self.best_margin_test_acc = 0
            self.best_test_acc = 0
            self.bin_feat_encoder = nn.Embedding(512, hidden_dim // 2,
                                                 padding_idx=0)  # maximum class value after converting to unique embeddings = 481
            self.bin_feat_split = 6
            num_reg_features = 6
            self.reg_feat_encoder = nn.Linear(num_reg_features, hidden_dim // 2)  # 5 continuous features
            emb_dim = 500
            self.edge_encoder = nn.Embedding(emb_dim, num_heads, padding_idx=0)  # 1 feature a 3 values, 1 a 2, no edge, offset 3
            self.edge_type = edge_type
            if self.edge_type == 'multi_hop':
                self.edge_dis_encoder = nn.Embedding(  # max_dist_in_graph * num_heads * num_heads
                    564 * num_heads * num_heads, 1)
            self.spatial_pos_encoder = nn.Embedding(2121, num_heads, padding_idx=0)  # max_dist_in_graph -> node_count is upper_bound
            self.in_degree_encoder = nn.Embedding(  # node_count is upper bound for max number of neighbours
                564, hidden_dim, padding_idx=0)
            self.out_degree_encoder = nn.Embedding(
                564, hidden_dim, padding_idx=0)

            # epochs to stop pre-training
            if self.gcn:
                self.end_epoch = 2500
            elif self.gat:
                self.end_epoch = 3000
            elif self.gin:
                self.end_epoch = 1000
            elif self.graphsage:
                self.end_epoch = 4500
            else:  # graphormer
                self.end_epoch = 6000

        elif dataset_name == 'mimic':
            self.task = task
            if self.task == 'acu':
                acu_class_counts = [4413, 4207, 3012, 2306, 636, 260, 194, 175, 157, 64, 54, 27, 22, 19, 8, 1, 1252, 647]
                if self.use_simple_acu_task:
                    # acu_class_counts = [4597, 4271, 5414, 1273, 1899]
                    acu_class_counts = [4597, 4271, 6687, 1899]
                    # acu_class_counts = [11348, 4207, 1899]
                # acu_class_counts = [8868, 8586]

                # loss_weighting == 'log':
                self.acu_weights = torch.tensor([np.abs(1.0 / (np.log(count) + 1)) for count in acu_class_counts], device=torch.device('cuda'),
                                                dtype=torch.float)

            self.acc_dict = {}
            self.acc_accumulation = 10
            self.acc_accumulation_count = 0
            self.log_weights_count = 0
            self.log_weights_freq = 10
            self.test_masks = []
            self.not_use_dev = True
            self.mimic_treatment_masks = defaultdict(list)
            self.mimic_vals_masks = defaultdict(list)
            self.mimic_vals_final_masks = defaultdict(list)
            self.pad_mode = pad_mode
            self.best_val_score = 0

            # hidden_dim = 256
            vals_dim = 160
            vals_feature_num = 112
            treat_dim = 64
            dems_dim = 24
            age_dim = 8

            self.age_encoder = nn.Linear(1, age_dim)
            self.demographics_encoder = nn.Embedding(20, dems_dim, padding_idx=0)  # 20 for 4 dems, 32 with ethnicity and insurance
            if not self.skip_transformer:
                self.vals_upscale = nn.Linear(vals_feature_num, vals_dim)

            # linear layer to upscale, then transformer to extract the time information
            if not self.skip_transformer:
                if self.pad_mode == 'pad_emb' or self.pad_mode == "emb":
                    self.treatment_upscale = nn.Embedding(65, treat_dim)
                else:
                    self.treatment_upscale = nn.Linear(16, treat_dim)

            if not self.skip_transformer:
                self.vals_config = BertConfig(vocab_size=1, hidden_size=vals_dim, num_hidden_layers=2, num_attention_heads=8,
                                              intermediate_size=vals_dim * 4,
                                              max_position_embeddings=25)
                # 336 to be transferable to sepsis
                self.vals_transformer = BertModel(config=self.vals_config)

            if not self.skip_transformer:
                self.treat_config = BertConfig(vocab_size=1, hidden_size=treat_dim, num_hidden_layers=2, num_attention_heads=8,
                                               intermediate_size=treat_dim * 4, max_position_embeddings=25)
                self.treatment_transformer = BertModel(config=self.treat_config)

            embedding_dim = 600  # 10 for age for half edge 100, for full edge 600
            if not self.gcn and not self.mlp and not self.graphsage and not self.gat and not self.gin:
                self.edge_encoder = nn.Embedding(embedding_dim, num_heads, padding_idx=0)  # 1 feature a 3 values, 1 a 2, no edge, offset 3
                self.edge_type = edge_type
                if self.edge_type == 'multi_hop':
                    self.edge_dis_encoder = nn.Embedding(  # max_dist_in_graph * num_heads * num_heads
                        550 * num_heads * num_heads, 1)
                self.spatial_pos_encoder = nn.Embedding(2121, num_heads, padding_idx=0)  # max_dist_in_graph -> node_count is upper_bound
                self.in_degree_encoder = nn.Embedding(  # node_count is upper bound for max number of neighbours
                    550, hidden_dim, padding_idx=0)
                self.out_degree_encoder = nn.Embedding(
                    550, hidden_dim, padding_idx=0)

        elif dataset_name == 'sepsis':

            if set_id == 'AB':
                sepsis_counts = [40336, 2932]
            elif set_id == 'A':
                sepsis_counts = [20336, 1790]
            elif set_id == 'B':
                sepsis_counts = [20000, 1142]

            # loss_weighting == 'log':
            self.sepsis_weights = torch.tensor([np.abs(1.0 / (np.log(count) + 1)) for count in sepsis_counts], device=torch.device('cuda'),
                                               dtype=torch.float)

            self.best_val_score = 0
            # use empty treatments to preserve same node structure as in MIMIC
            self.vals_dim = 160
            vals_feature_num = 68
            self.treat_dim = 64  # need to access later
            dems_dim = 16
            hosp_adm_dim = 8  # additional continous demographic feature -> share demographics embedding size
            age_dim = 8
            assert hidden_dim == self.vals_dim + dems_dim + age_dim + hosp_adm_dim + self.treat_dim

            self.age_encoder = nn.Linear(1, age_dim)
            self.hosp_adm_encoder = nn.Linear(1, hosp_adm_dim)
            self.demographics_encoder = nn.Embedding(20, dems_dim, padding_idx=0)  # 20 for 4 dems, 32 with ethnicity and insurance

            # linear layer to upscale, then transformer to extract the time information
            self.vals_upscale = nn.Linear(vals_feature_num, self.vals_dim)
            self.vals_config = BertConfig(vocab_size=1, hidden_size=self.vals_dim, num_hidden_layers=2, num_attention_heads=8,
                                          intermediate_size=self.vals_dim * 4,
                                          max_position_embeddings=25)  # 25 for 24 hour task, for full time task needs to be 336 #val and test are not cropped to 96
            self.vals_transformer = BertModel(config=self.vals_config)

            embedding_dim = 600
            self.edge_encoder = nn.Embedding(embedding_dim, num_heads, padding_idx=0)
            self.edge_type = edge_type
            if self.edge_type == 'multi_hop':
                self.edge_dis_encoder = nn.Embedding(  # max_dist_in_graph * num_heads * num_heads
                    550 * num_heads * num_heads, 1)
            self.spatial_pos_encoder = nn.Embedding(2121, num_heads, padding_idx=0)  # max_dist_in_graph -> node_count is upper_bound
            self.in_degree_encoder = nn.Embedding(  # node_count is upper bound for max number of neighbours
                550, hidden_dim, padding_idx=0)
            self.out_degree_encoder = nn.Embedding(
                550, hidden_dim, padding_idx=0)

        else:
            raise NotImplementedError('dataset not implemented')

        self.input_dropout = nn.Dropout(intput_dropout_rate)
        if self.gcn:
            encoders = [GCNConv(in_channels=hidden_dim, out_channels=hidden_dim) for _ in range(n_layers)]
        elif self.gat:
            encoders = [GATConv(in_channels=hidden_dim, out_channels=hidden_dim // num_heads, heads=num_heads) for _ in range(n_layers)]
        elif self.graphsage:
            encoders = [SAGEConv(in_channels=hidden_dim, out_channels=hidden_dim) for _ in range(n_layers)]
        elif self.gin:
            encoders = [GINConv(nn=nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                                 nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))) for _ in range(n_layers)]

        elif self.mlp:
            encoders = [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)]

        else:  # default
            encoders = [EncoderLayer(hidden_dim, ffn_dim, dropout_rate, attention_dropout_rate, num_heads)
                        for _ in range(n_layers)]
        self.layers = nn.ModuleList(encoders)
        self.final_ln = nn.LayerNorm(hidden_dim)

        if dataset_name == 'tadpole':  # for now only predict discrete features (one-hot-encoding)
            if self.mask_all_tadpole:
                self.bin_out_proj_discrete = nn.Linear(hidden_dim, 208)  # possible classes per columns: 3, 19, 107, 11, 68 - sum: 208
                self.bin_out_proj_cont = nn.Linear(hidden_dim, 5)  # continuous features
                self.pred_split_idxs = [3, 22, 129, 140, 208]
            else:
                self.bin_out_proj = nn.Linear(hidden_dim, 208)  # possible classes per columns: 3, 19, 107, 11, 68 - sum: 208
                self.pred_split_idxs = [3, 22, 129, 140, 208]

        elif dataset_name == 'tadpole_class':
            self.bin_out_proj = nn.Linear(hidden_dim, 3)

        elif dataset_name == 'mimic':
            if self.task == 'pre_mask' or self.task == 'patient_prediction':
                self.bin_out_proj_vals = nn.Linear(hidden_dim, 56 * 24)
                self.bin_out_proj_treat = nn.Linear(hidden_dim, 16 * 24)
            elif self.task == 'treat_prediction':
                self.bin_out_proj = nn.Linear(hidden_dim, 14)  # 16
            elif self.task == 'multi_task_pt':
                self.bin_out_proj_vals_FM = nn.Linear(hidden_dim, 56 * 24)
                self.bin_out_proj_treat_FM = nn.Linear(hidden_dim, 16 * 24)
                self.bin_out_proj_vals_PM = nn.Linear(hidden_dim, 56 * 24)
                self.bin_out_proj_treat_PM = nn.Linear(hidden_dim, 16 * 24)
                self.bin_out_proj_vals_BM = nn.Linear(hidden_dim, 56 * 24)
                self.bin_out_proj_treat_BM = nn.Linear(hidden_dim, 16 * 24)
                self.bin_out_proj_TP = nn.Linear(hidden_dim, 14)
            else:
                output_dims = {'los': 2, 'acu': 4 if self.use_simple_acu_task else 18}  # 5
                self.bin_out_proj = nn.Linear(hidden_dim, output_dims[task])
                # acu: multi-hot encoding if label is present or not

        elif dataset_name == 'sepsis':
            self.bin_out_proj = nn.Linear(hidden_dim, 2)  # binary sepsis label

        self.graph_token = nn.Embedding(1, hidden_dim)
        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)

        dataset = get_dataset(dataset_name, cross_val_split=cross_val_split, task=task)
        self.evaluator = dataset['evaluator']
        self.metric = dataset['metric']
        self.loss_fn = dataset['loss_fn']

        self.dataset_name = dataset_name

        self.warmup_updates = warmup_updates
        self.fixed_updates = 500
        self.tot_updates = tot_updates
        self.cross_val = False if cross_val_split is None else True
        self.cross_val_split = cross_val_split if self.cross_val else fold
        self.label_ratio = label_ratio
        self.peak_lr = peak_lr
        self.end_lr = end_lr
        self.weight_decay = weight_decay
        self.multi_hop_max_dist = multi_hop_max_dist

        self.flag = flag
        self.flag_m = flag_m
        self.flag_step_size = flag_step_size
        self.flag_mag = flag_mag
        self.hidden_dim = hidden_dim
        self.automatic_optimization = not self.flag
        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, batched_data, perturb=None):
        if self.gcn or self.mlp or self.gat or self.graphsage or self.gin:
            edge_index = batched_data.edge_index
            x = batched_data.x
            if x is not None:
                x = x.unsqueeze(0)
        else:
            attn_bias, spatial_pos, x = batched_data.attn_bias, batched_data.spatial_pos, batched_data.x
            in_degree, out_degree = batched_data.in_degree, batched_data.in_degree
            edge_input, attn_edge_type = batched_data.edge_input, batched_data.attn_edge_type
            attn_mask = batched_data.attn_mask
            padding_mask = batched_data.padding_mask
        # mimic
        if self.dataset_name == 'mimic':
            vals, is_measured, treatments, demographics = batched_data.vals, batched_data.is_measured, batched_data.treatments, batched_data.demographics
            n_graph, n_node = vals.size()[:2]
        elif self.dataset_name == 'sepsis':
            vals, demographics = batched_data.vals, batched_data.demographics
            n_graph, n_node = demographics.size()[:2]
        else:
            n_graph, n_node = x.size()[:2]
        if not self.gcn and not self.mlp and not self.graphsage and not self.gat and not self.gin:
            graph_attn_bias = attn_bias.clone()
            graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(
                1, self.num_heads, 1, 1)  # [n_graph, n_head, n_node+1, n_node+1]

            # spatial pos
            # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
            spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)
            graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:,
                                            :, 1:, 1:] + spatial_pos_bias
            # reset spatial pos here
            t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
            graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
            graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t

            # edge feature
            if self.edge_type == 'multi_hop':
                spatial_pos_ = spatial_pos.clone()
                spatial_pos_[spatial_pos_ == 0] = 1  # set pad to 1
                # set 1 to 1, x > 1 to x - 1
                spatial_pos_ = torch.where(spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)
                if self.multi_hop_max_dist > 0:
                    spatial_pos_ = spatial_pos_.clamp(0, self.multi_hop_max_dist)
                    edge_input = edge_input[:, :, :, :self.multi_hop_max_dist, :]
                # [n_graph, n_node, n_node, max_dist, n_head]
                edge_input = self.edge_encoder(edge_input).mean(-2)
                max_dist = edge_input.size(-2)
                edge_input_flat = edge_input.permute(
                    3, 0, 1, 2, 4).reshape(max_dist, -1, self.num_heads)
                edge_input_flat = torch.bmm(edge_input_flat, self.edge_dis_encoder.weight.reshape(
                    -1, self.num_heads, self.num_heads)[:max_dist, :, :])
                edge_input = edge_input_flat.reshape(
                    max_dist, n_graph, n_node, n_node, self.num_heads).permute(1, 2, 3, 0, 4)
                edge_input = (edge_input.sum(-2) /
                              (spatial_pos_.float().unsqueeze(-1))).permute(0, 3, 1, 2)
            else:
                # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
                edge_input = self.edge_encoder(
                    attn_edge_type.long()).mean(-2).permute(0, 3, 1, 2)

            graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + edge_input
            graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)  # reset

        # node feature + graph token
        # different encoders for bin and regr features
        if self.dataset_name == 'mimic':
            # build node_features for MIMIC

            age_features = self.age_encoder(demographics[:, :, 0:1]).squeeze()
            dem_features = self.demographics_encoder(demographics[:, :, 1:].long()).sum(dim=-2).squeeze()

            vals_features = self.vals_upscale(vals)

            if self.pad_mode == 'pad_emb' or self.pad_mode == "emb":
                treat_features = self.treatment_upscale(treatments.int()).sum(dim=-2)
            else:
                treat_features = self.treatment_upscale(treatments)
            if self.skip_transformer:  # not compatible with gat/graphsage/gin
                vals_features_transformed = vals_features.mean(dim=2) if (
                                                                                 not self.gcn and not self.mlp) or self.task == "pre_mask" else vals_features.mean(
                    dim=1)
                treat_features_transformed = treat_features.mean(dim=2) if (
                                                                                   not self.gcn and not self.mlp) or self.task == "pre_mask" else treat_features.mean(
                    dim=1)
            else:
                if (
                        self.gcn or self.mlp or self.gat or self.graphsage or self.gin) and not self.task == "pre_mask":  # treat differently because batches are thrown together already
                    vals_features_transformed = self.vals_transformer(inputs_embeds=vals_features).last_hidden_state.mean(dim=1)
                    treat_features_transformed = self.treatment_transformer(inputs_embeds=treat_features).last_hidden_state.mean(dim=1)
                else:
                    vals_features_transformed = torch.zeros((vals_features.size(0), vals_features.size(1), vals_features.size(3)),
                                                            dtype=torch.float32, device=self.device)
                    treat_features_transformed = torch.zeros((treat_features.size(0), treat_features.size(1), treat_features.size(3)),
                                                             dtype=torch.float32, device=self.device)

                    for idx, graph in enumerate(vals_features):
                        attn_mask = None
                        vals_features_transformed[idx] = self.vals_transformer(inputs_embeds=graph, attention_mask=attn_mask).last_hidden_state.mean(
                            dim=1)

                    for idx, graph in enumerate(treat_features):
                        attn_mask = None
                        treat_features_transformed[idx] = self.treatment_transformer(inputs_embeds=graph,
                                                                                     attention_mask=attn_mask).last_hidden_state.mean(dim=1)

            # concat all features
            if len(age_features.shape) == 2 and (
                    (not self.gcn and not self.mlp and not self.graphsage and not self.gat and not self.gin) or self.task == "pre_mask"):
                age_features = age_features.unsqueeze(0)
                dem_features = dem_features.unsqueeze(0)
            node_feature = torch.cat([age_features, dem_features, vals_features_transformed, treat_features_transformed],
                                     dim=1 if (
                                                      self.gcn or self.mlp or self.graphsage or self.gat or self.gin) and not self.task == 'pre_mask' else 2)

        elif self.dataset_name == 'sepsis':
            # build node_features for sepsis
            age_features = self.age_encoder(demographics[:, :, 0:1]).squeeze()
            hosp_adm_features = self.hosp_adm_encoder(demographics[:, :, -2:-1]).squeeze()
            dem_features = self.demographics_encoder(demographics[:, :, 1:-1].long()).sum(dim=-2).squeeze()

            # first flatten, apply upscale, then unflatten again
            all_stack = torch.cat([torch.cat(elem) for elem in vals])
            all_stack = self.vals_upscale(all_stack)

            vals_features = []
            end_index = 0
            for graph in vals:
                separated = []
                for elem in graph:
                    length = elem.shape[0]
                    start_index = end_index
                    end_index = start_index + length
                    separated.append(all_stack[start_index:end_index])
                vals_features.append(separated)

            vals_features_transformed = []
            max_num_nodes = max([len(elem) for elem in vals_features])
            for idx, graph in enumerate(vals_features):
                node_seq_lengths = [node.size(0) for node in graph]
                max_len = max(node_seq_lengths)
                padded_graph = torch.stack([F.pad(node, [0, 0, 0, max_len - node.size(0)], value=np.nan) for node in graph], dim=0)
                # compute nan mask for padded_graph
                attn_mask = torch.isnan(padded_graph).max(-1).values
                padded_graph[attn_mask] = 0.0  # get rid of nan values
                attn_mask = ~attn_mask  # invert mask so tokens that should be attended are 1

                padded_graph = self.vals_transformer(inputs_embeds=padded_graph, attention_mask=attn_mask).last_hidden_state
                # mean each feature over time dimension, ignore padded values
                mean_features = []
                for idx, length in enumerate(node_seq_lengths):
                    mean_features.append(padded_graph[idx, :length].mean(dim=0))
                mean_features = torch.stack(mean_features)
                num_nodes = mean_features.size(0)
                if num_nodes < max_num_nodes:
                    new_mean_features = mean_features.new_zeros([max_num_nodes, self.vals_dim], dtype=mean_features.dtype)
                    new_mean_features[:num_nodes] = mean_features
                    mean_features = new_mean_features
                vals_features_transformed.append(mean_features)

            # need to pad here after meaning over time instead of in collator
            vals_features_transformed = torch.stack(vals_features_transformed)

            empty_treat_tensor = torch.zeros((vals_features_transformed.size(0), vals_features_transformed.size(1), self.treat_dim),
                                             device=self.device)

            if len(age_features.shape) == 2:  # batch_size of 1 -> need to add batch dimension
                age_features = age_features.unsqueeze(0)
                hosp_adm_features = hosp_adm_features.unsqueeze(0)
                dem_features = dem_features.unsqueeze(0)

            node_feature = torch.cat([age_features, hosp_adm_features, dem_features, vals_features_transformed, empty_treat_tensor], dim=2)

        else:
            bin_features = self.bin_feat_encoder(x[:, :, :self.bin_feat_split].long()).sum(dim=-2)
            reg_features = self.reg_feat_encoder(x[:, :, self.bin_feat_split:])

            node_feature = torch.cat([bin_features, reg_features], dim=2)  # [n_graph, n_node, n_hidden]
        if self.flag and perturb is not None:
            node_feature += perturb

        if not self.gcn and not self.mlp and not self.gat and not self.graphsage and not self.gin:
            node_feature = node_feature + \
                           self.in_degree_encoder(in_degree) + \
                           self.out_degree_encoder(out_degree)
            graph_token_feature = self.graph_token.weight.unsqueeze(
                0).repeat(n_graph, 1, 1)
            graph_node_feature = torch.cat(
                [graph_token_feature, node_feature], dim=1)
        else:
            graph_node_feature = node_feature

        # transformer encoder
        output = self.input_dropout(graph_node_feature)
        if self.gcn or self.graphsage or self.gin:
            for enc_layer in self.layers:
                output = enc_layer(x=output, edge_index=edge_index)
                output = F.relu(output)
        elif self.gat:
            output = output[0] if self.dataset_name == 'tadpole' or self.dataset_name == 'tadpole_class' else output
            if self.dataset_name == 'mimic' and self.task == 'pre_mask':
                bs, n_node, n_hidden = output.shape
                output = output.view(-1, 256)
            for enc_layer in self.layers:
                output = enc_layer(x=output, edge_index=edge_index)
                output = F.relu(output)
            output = output.unsqueeze(0) if self.dataset_name == 'tadpole' or self.dataset_name == 'tadpole_class' else output
            output = output.view(bs, n_node, n_hidden) if self.dataset_name == 'mimic' and self.task == 'pre_mask' else output
        elif self.mlp:
            for enc_layer in self.layers:
                output = enc_layer(output)
                output = F.relu(output)
        else:
            for enc_layer in self.layers:
                output = enc_layer(output, attn_bias=graph_attn_bias, attn_mask=attn_mask)

        output = self.final_ln(output)

        if self.dataset_name == 'tadpole':
            if self.mask_all_tadpole:
                if self.gcn or self.mlp or self.gat or self.graphsage or self.gin:
                    bin_output_cont = self.bin_out_proj_cont(output)
                    bin_output_discrete = self.bin_out_proj_discrete(output)
                else:
                    bin_output_cont = self.bin_out_proj_cont(output)[:, 1:]
                    bin_output_discrete = self.bin_out_proj_discrete(output)[:, 1:]
                return bin_output_discrete, bin_output_cont
            else:
                if self.gcn or self.mlp or self.gat or self.graphsage or self.gin:
                    bin_output = self.bin_out_proj(output)
                else:
                    bin_output = self.bin_out_proj(output)[:, 1:]
                return bin_output
        elif self.dataset_name == 'tadpole_class':
            if self.gcn or self.mlp or self.gat or self.graphsage or self.gin:
                bin_output = self.bin_out_proj(output)
            else:
                bin_output = self.bin_out_proj(output)[:, 1:]
            return bin_output

        elif self.dataset_name == 'sepsis':
            bin_output = self.bin_out_proj(output)[:, 1:]
            return bin_output

        elif self.dataset_name == 'mimic':
            not_graphormer = self.gcn or self.mlp or self.gat or self.graphsage or self.gin
            if self.task == 'pre_mask' or self.task == 'patient_prediction':

                vals_output = self.bin_out_proj_vals(output) if not_graphormer else self.bin_out_proj_vals(output)[:, 1:]
                treat_output = self.bin_out_proj_treat(output) if not_graphormer else self.bin_out_proj_treat(output)[:, 1:]
                bin_output = (vals_output, treat_output)
            elif self.task == 'multi_task_pt':
                if batched_data.mask_task == 'pre_mask':  # PM = patient masking
                    vals_output = self.bin_out_proj_vals_FM(output) if not_graphormer else self.bin_out_proj_vals_FM(output)[:, 1:]
                    treat_output = self.bin_out_proj_treat_FM(output) if not_graphormer else self.bin_out_proj_treat_FM(output)[:, 1:]
                    bin_output = (vals_output, treat_output)
                elif batched_data.mask_task == 'PM':
                    vals_output = self.bin_out_proj_vals_PM(output) if not_graphormer else self.bin_out_proj_vals_PM(output)[:, 1:]
                    treat_output = self.bin_out_proj_treat_PM(output) if not_graphormer else self.bin_out_proj_treat_PM(output)[:, 1:]
                    bin_output = (vals_output, treat_output)
                elif batched_data.mask_task == 'BM':
                    vals_output = self.bin_out_proj_vals_BM(output) if not_graphormer else self.bin_out_proj_vals_BM(output)[:, 1:]
                    treat_output = self.bin_out_proj_treat_BM(output) if not_graphormer else self.bin_out_proj_treat_BM(output)[:, 1:]
                    bin_output = (vals_output, treat_output)
                else:  # treatment prediction
                    bin_output = self.bin_out_proj_TP(output) if not_graphormer else self.bin_out_proj_TP(output)[:, 1:]
            else:  # classification or treatment prediction
                bin_output = self.bin_out_proj(output) if not_graphormer else self.bin_out_proj(output)[:, 1:]
            return bin_output

    def training_step(self, batched_data, batch_idx):
        update_mask = batched_data.update_mask
        if self.dataset_name == 'mimic':
            train_mask = batched_data.train_dev_mask if self.task.startswith(
                'pre') and not self.not_use_dev else batched_data.train_mask  # for all pretraining tasks do not use dev items
        else:
            train_mask = batched_data.train_mask

        if self.dataset_name == 'tadpole':
            if self.mask_all_tadpole:
                y_hat_bin, y_hat_cont = self(batched_data)
                y_gt_bin = batched_data.y[:, 1:6]
                y_gt_cont = batched_data.y[:, 7:]
            else:
                y_hat_bin = self(batched_data)
                y_gt_bin = batched_data.y[:, 1:6]

            loss_discrete = 0.
            start_idx = 0
            y_pred = []
            loss_count = 0
            train_mask = train_mask[0] if not self.gcn and not self.mlp and not self.gat and not self.graphsage and not self.gin else train_mask

            class_counts = {0: 3, 1: 19, 2: 107, 3: 11, 4: 68}
            for feature_idx, pred_split_idx in enumerate(self.pred_split_idxs):
                feature_idx_new = feature_idx + 1  # gender should not be predicted and is feature 0
                feat_update_mask = update_mask[train_mask][:, feature_idx_new]
                feat_pred = y_hat_bin[0, :, start_idx:pred_split_idx] if self.dataset_name == 'tadpole' else y_hat_bin[:, 0, start_idx:pred_split_idx]
                feat_gt = y_gt_bin[:, feature_idx]
                if len(feat_pred[train_mask][feat_update_mask]) > 0:
                    # calculate loss weights for current train split and feature
                    classes, counts = torch.unique(feat_gt[train_mask], return_counts=True)
                    feat_loss_weights = []
                    for c in range(class_counts[feature_idx]):
                        if c in classes:
                            feat_loss_weights.append(1 / counts[classes == c].item())
                        else:
                            feat_loss_weights.append(0.)  # not existing class

                    feat_loss_weights = torch.tensor(feat_loss_weights).to(self.device)
                    loss_feat = self.loss_fn(feat_pred[train_mask][feat_update_mask], feat_gt[train_mask][feat_update_mask].long(),
                                             weight=feat_loss_weights)
                    loss_discrete += loss_feat
                    loss_count += 1
                start_idx = pred_split_idx

                y_pred_feat = torch.argmax(feat_pred, dim=1)[train_mask]
                y_pred.append(y_pred_feat)

            loss_discrete /= loss_count

            if self.mask_all_tadpole:

                cont_update_mask = update_mask[:, 7:][train_mask]
                # compute mse loss for continuous features
                loss_cont = F.mse_loss(y_hat_cont[0][train_mask][cont_update_mask], y_gt_cont[train_mask][cont_update_mask], reduction='mean')
                self.log('train_loss_cont', loss_cont, sync_dist=True, on_epoch=True, on_step=False)
                self.log('train_loss_disc', loss_discrete, sync_dist=True, on_epoch=True, on_step=False)
                loss = (loss_discrete + loss_cont) / 2

                y_pred = [y_pred, y_hat_cont]
                y_true = [batched_data.y[:, 1:6][train_mask], batched_data.y[:, 7:][train_mask]]

                self.eval_epoch_end(
                    outputs=[{'y_pred': y_pred, 'y_true': y_true, 'update_mask': update_mask[train_mask], }],
                    split='train')

            else:
                loss = loss_discrete
                self.eval_epoch_end(
                    outputs=[{'y_pred': y_pred, 'y_true': batched_data.y[:, 1:6][train_mask], 'update_mask': update_mask[train_mask], }],
                    split='train')

        elif self.dataset_name == 'tadpole_class':
            y_hat_bin = self(batched_data)
            y_gt_bin = batched_data.y
            train_mask = train_mask[0] if not self.gcn and not self.mlp and not self.gat and not self.graphsage and not self.gin else train_mask

            # calculate loss weights for current split
            num0, num1, num2 = torch.unique(y_gt_bin[train_mask], return_counts=True)[1]
            loss_weights = [1 / num0, 1 / num1, 1 / num2]

            if self.label_ratio != 1.0:
                drop_idxs = np.load(f'data/tadpole/split/label_drop_idxs_fold{self.cross_val_split}_{self.label_ratio}_bal_sim.npy')
                drop_pos = [np.where(batched_data.node_id == drop_idx)[1].item() for drop_idx in drop_idxs]
                train_mask[drop_pos] = False

            pred = y_hat_bin[0] if self.dataset_name == 'tadpole_class' else y_hat_bin[:, 0]

            if update_mask is not None:
                loss = self.loss_fn(pred[train_mask][update_mask], y_gt_bin[train_mask][update_mask].long(),
                                    weight=torch.tensor(loss_weights).to(self.device))
            else:
                loss = self.loss_fn(pred[train_mask], y_gt_bin[train_mask].long(), weight=torch.tensor(loss_weights).to(self.device))

            y_pred = torch.argmax(pred, dim=1)

            self.eval_epoch_end(
                outputs=[{'y_pred': y_pred[train_mask], 'y_true': y_gt_bin[train_mask], 'y_scores': pred[train_mask], 'update_mask': update_mask}],
                split='train')

        elif self.dataset_name == 'mimic':

            if self.task == 'pre_mask' or self.task == 'patient_prediction' or (self.task == 'multi_task_pt' and (
                    batched_data.mask_task == 'pre_mask' or batched_data.mask_task == 'PM' or batched_data.mask_task == 'BM')):
                update_mask_vals, update_mask_treat = update_mask[0][train_mask], update_mask[1][train_mask]

                y_pred_vals, y_pred_treat = self(batched_data)
                y_pred_vals = y_pred_vals.view(y_pred_vals.shape[0], y_pred_vals.shape[1], 24, 56)
                y_pred_treat = y_pred_treat.view(y_pred_treat.shape[0], y_pred_treat.shape[1], 24, 16)
                y_true_vals, y_true_treat, _ = batched_data.y

                # apply validation mask
                y_pred_vals, y_pred_treat = y_pred_vals[train_mask][update_mask_vals], y_pred_treat[train_mask][update_mask_treat]
                y_true_vals, y_true_treat = y_true_vals[train_mask][update_mask_vals], y_true_treat[train_mask][update_mask_treat]

                # compute joint loss
                vals_loss = F.mse_loss(y_pred_vals, y_true_vals, reduction='mean')
                weight = torch.ones_like(y_true_treat, device=self.device)
                weight[y_true_treat == 0] = 1 / 4.5  # weight 0s less as they are more frequent
                treat_loss = F.binary_cross_entropy_with_logits(y_pred_treat, y_true_treat, reduction='mean', weight=weight)

                loss = vals_loss + treat_loss
                if self.task == 'multi_task_pt':
                    self.log(f'train_measurement_loss_{batched_data.mask_task}', vals_loss, sync_dist=True, on_epoch=True, on_step=False)
                    self.log(f'train_treat_loss_{batched_data.mask_task}', treat_loss, sync_dist=True, on_epoch=True, on_step=False)
                    self.log(f'train_loss_{batched_data.mask_task}', loss, sync_dist=True, on_epoch=True, on_step=False)
                else:
                    self.log(f'train_measurement_loss', vals_loss, sync_dist=True, on_epoch=True, on_step=False)
                    self.log(f'train_treat_loss', treat_loss, sync_dist=True, on_epoch=True, on_step=False)
                    self.log(f'train_loss', loss, sync_dist=True, on_epoch=True, on_step=False)

                y_pred = [y_pred_vals, torch.sigmoid(y_pred_treat) > 0.5]
                y_true = [y_true_vals, y_true_treat]

                self.eval_epoch_end(outputs=[{'y_pred': y_pred, 'y_true': y_true, 'update_mask': update_mask, 'mask_task': batched_data.mask_task}],
                                    split='train')

            elif self.task == 'treat_prediction' or (self.task == 'multi_task_pt' and batched_data.mask_task == 'TP'):
                y_gt_bin = batched_data.y[train_mask]
                pred = self(batched_data)[train_mask]
                weight = torch.ones_like(y_gt_bin, device=self.device)
                weight[y_gt_bin == 0] = 1 / 3.8
                loss = F.binary_cross_entropy_with_logits(pred, y_gt_bin, reduction='mean', weight=weight)
                if self.task == 'multi_task_pt':
                    self.log(f'train_loss_{batched_data.mask_task}', loss, sync_dist=True, on_epoch=True, on_step=False)

                y_pred = torch.sigmoid(pred) > 0.5

                if not self.compute_results:
                    self.eval_epoch_end(outputs=[
                        {'y_pred': y_pred, 'y_scores': pred, 'y_true': y_gt_bin, 'update_mask': update_mask, 'mask_task': batched_data.mask_task}],
                        split='train')

            else:
                y_hat_bin = self(batched_data)

                # calculate loss weights for current split
                if self.task == 'acu':
                    loss_weights = self.acu_weights
                else:
                    loss_weights = torch.tensor([1 / count for count in torch.unique(batched_data.y[train_mask], return_counts=True)[1]],
                                                device=self.device)

                if self.label_ratio != 1.0:
                    drop_idxs = np.load(f'data/mimic-iii-0/drop/label_drop_idxs_rot{self.rotation}_{self.label_ratio}.npy')
                    drop_pos = [np.where(batched_data.node_id.cpu().numpy() == d) for d in drop_idxs if
                                len(np.where(batched_data.node_id.cpu().numpy() == d)[0]) != 0]
                    if self.gcn or self.mlp or self.gat or self.graphsage or self.gin:
                        for index in drop_pos:
                            train_mask[index[0]] = False
                    else:
                        for index in drop_pos:
                            train_mask[index[0], index[1]] = False

                y_gt_bin = batched_data.y[train_mask]
                pred = y_hat_bin[train_mask]

                if update_mask is not None:
                    loss = self.loss_fn(pred[update_mask], y_gt_bin[update_mask].long(),
                                        weight=torch.tensor(loss_weights).to(self.device))
                else:
                    loss = self.loss_fn(pred, y_gt_bin.long(), weight=loss_weights)

                y_pred = torch.argmax(pred, dim=1)

                if not self.compute_results:
                    self.eval_epoch_end(outputs=[{'y_pred': y_pred, 'y_scores': pred, 'y_true': y_gt_bin, 'update_mask': update_mask}], split='train')

        elif self.dataset_name == 'sepsis':
            y_hat_bin = self(batched_data)

            y_gt_bin = batched_data.y[train_mask]
            pred = y_hat_bin[train_mask]

            loss = self.loss_fn(pred, y_gt_bin.long(), weight=torch.tensor(self.sepsis_weights).to(self.device))

            y_pred = torch.argmax(pred, dim=1)

            # if not self.compute_results:
            self.eval_epoch_end(outputs=[{'y_pred': y_pred, 'y_scores': pred, 'y_true': y_gt_bin, 'update_mask': update_mask}], split='train')

        if self.cross_val:
            self.train_losses.append(loss.detach().cpu().item())
        else:
            self.log('train_loss', loss, sync_dist=True, on_epoch=True, on_step=False)
        return loss

    def eval_step(self, batched_data, batch_idx, split):
        # self.compute_same_mask_performance_mimic()
        update_mask = batched_data.update_mask
        if self.dataset_name == 'mimic':
            val_mask = batched_data.dev_mask if self.task.startswith(
                'pre') and not self.not_use_dev else batched_data.val_mask  # for all pretraining tasks validate on dev set, ignore val set for now
        else:
            val_mask = batched_data.val_mask
        test_mask = batched_data.test_mask
        mask = val_mask if split == 'val' else test_mask  # never called with train

        if self.dataset_name == 'tadpole':
            mask = mask[0] if not self.gcn and not self.mlp and not self.gat and not self.graphsage and not self.gin else mask

            if self.mask_all_tadpole:
                y_pred_bin, y_pred_cont = self(batched_data)

                y_pred_bin = y_pred_bin[:, mask, :]
                y_pred_cont = y_pred_cont[:, mask, :]
                y_true_bin = batched_data.y[:, 1:6][mask]
                y_true_cont = batched_data.y[:, 7:][mask]

            else:
                y_pred_bin = self(batched_data)[:, mask, :]
                y_true_bin = batched_data.y[:, 1:6][mask]

            start_idx = 0
            y_pred = []
            loss_disc = 0.
            update_mask = update_mask[mask]
            loss_count = 0

            for feature_idx, pred_split_idx in enumerate(self.pred_split_idxs):
                feat_update_mask = update_mask[:, feature_idx + 1]
                feat_pred = y_pred_bin[0, :, start_idx:pred_split_idx] if self.dataset_name == 'tadpole' else y_pred_bin[:, 0,
                                                                                                              start_idx:pred_split_idx]
                feat_gt = y_true_bin[:, feature_idx]
                if len(feat_pred[feat_update_mask]) > 0:
                    loss_feat = self.loss_fn(feat_pred[feat_update_mask], feat_gt[feat_update_mask].long())
                    loss_disc += loss_feat
                    loss_count += 1

                y_pred_feat = torch.argmax(feat_pred, dim=1)
                y_pred.append(y_pred_feat)

                start_idx = pred_split_idx

            if self.mask_all_tadpole:
                cont_update_mask = update_mask[:, 7:]
                # compute mse loss for continuous features
                loss_cont = F.mse_loss(y_pred_cont[0][cont_update_mask], y_true_cont[cont_update_mask], reduction='mean')

                loss = (loss_disc / loss_count + loss_cont) / 2
                self.log(f'{split}_cont_loss', loss_cont, sync_dist=True, on_epoch=True, on_step=False)
                self.log(f'{split}_disc_loss', loss_disc / loss_count, sync_dist=True, on_epoch=True, on_step=False)
                self.log(f'{split}_loss', loss, sync_dist=True, on_epoch=True, on_step=False)

                y_pred = [y_pred, y_pred_cont]
                y_true = [y_true_bin, y_true_cont]

            else:
                if loss_count > 0:
                    self.log(f'{split}_loss', loss_disc / loss_count, sync_dist=True, on_epoch=True, on_step=False)
                y_true = y_true_bin

            return {
                'y_pred': y_pred,
                'y_scores': [],
                'y_true': y_true,
                'update_mask': update_mask
            }

        elif self.dataset_name == 'tadpole_class':
            y_pred_bin = self(batched_data)
            mask = mask[0] if not self.gcn and not self.mlp and not self.gat and not self.graphsage and not self.gin else mask
            y_true = batched_data.y[mask]

            pred = y_pred_bin[0][mask] if self.dataset_name == 'tadpole_class' else y_pred_bin[:, 0][
                mask]
            loss = self.loss_fn(pred, y_true.long())

            y_pred = torch.argmax(pred, dim=1)

            if self.cross_val:
                self.val_losses.append(loss.cpu().item())
            else:
                self.log(f'{split}_loss', loss, sync_dist=True, on_epoch=True, on_step=False)

        elif self.dataset_name == 'mimic':
            if self.task == 'pre_mask' or self.task == 'patient_prediction' or (self.task == 'multi_task_pt' and (
                    batched_data.mask_task == 'pre_mask' or batched_data.mask_task == 'PM' or batched_data.mask_task == 'BM')):
                update_mask_vals, update_mask_treat = update_mask[0], update_mask[1]

                y_pred_vals, y_pred_treat, y_true_vals, y_true_treat = self.get_model_prediction(item=batched_data, vals_final_mask=update_mask_vals,
                                                                                                 treat_mask=update_mask_treat, mask=mask)

                # compute joint loss
                vals_loss = F.mse_loss(y_pred_vals, y_true_vals, reduction='mean')
                treat_loss = F.binary_cross_entropy(y_pred_treat, y_true_treat, reduction='mean')

                loss = vals_loss + treat_loss
                if self.task == 'multi_task_pt':
                    self.log(f'{split}_measurement_loss_{batched_data.mask_task}', vals_loss, sync_dist=True, on_epoch=True, on_step=False)
                    self.log(f'{split}_treat_loss_{batched_data.mask_task}', treat_loss, sync_dist=True, on_epoch=True, on_step=False)
                    self.log(f'{split}_loss_{batched_data.mask_task}', loss, sync_dist=True, on_epoch=True, on_step=False)
                else:
                    self.log(f'{split}_measurement_loss', vals_loss, sync_dist=True, on_epoch=True, on_step=False)
                    self.log(f'{split}_treat_loss', treat_loss, sync_dist=True, on_epoch=True, on_step=False)
                    self.log(f'{split}_loss', loss, sync_dist=True, on_epoch=True, on_step=False)

                y_pred = [y_pred_vals, y_pred_treat > 0.5]
                y_true = [y_true_vals, y_true_treat]
                pred = None

            elif self.task == 'treat_prediction' or (self.task == 'multi_task_pt' and batched_data.mask_task == 'TP'):
                y_true = batched_data.y[mask]
                pred = self(batched_data)[mask]
                loss = F.binary_cross_entropy_with_logits(pred, y_true, reduction='mean', weight=None)
                if self.task == 'multi_task_pt':
                    self.log(f'{split}_loss_{batched_data.mask_task}', loss, sync_dist=True, on_epoch=True, on_step=False)
                else:
                    self.log(f'{split}_loss', loss, sync_dist=True, on_epoch=True, on_step=False)

                y_pred = torch.sigmoid(pred) > 0.5

            else:
                y_pred_bin = self(batched_data)

                y_true = batched_data.y[mask]
                pred = y_pred_bin[mask]

                if self.task == 'acu':
                    loss_weights = self.acu_weights
                    loss = self.loss_fn(pred, y_true.long(), weight=loss_weights)
                else:  # los
                    loss = self.loss_fn(pred, y_true.long())

                y_pred = torch.argmax(pred, dim=1)

                if self.cross_val:
                    self.val_losses.append(loss.cpu().item())
                else:
                    self.log(f'{split}_loss', loss, sync_dist=True, on_epoch=True, on_step=False)

        elif self.dataset_name == 'sepsis':
            y_pred_bin = self(batched_data)

            y_true = batched_data.y[mask]
            pred = y_pred_bin[mask]

            loss = self.loss_fn(pred, y_true.long())

            y_pred = torch.argmax(pred, dim=1)

            if self.cross_val:
                self.val_losses.append(loss.cpu().item())
            else:
                self.log(f'{split}_loss', loss, sync_dist=True, on_epoch=True, on_step=False)

        return {
            'y_pred': y_pred,
            'y_scores': pred,
            'y_true': y_true,
            'update_mask': update_mask,
            'mask_task': batched_data.mask_task,
        }

    def add_to_acc_dict(self, feature, y_true, y_pred, scores):
        if scores is not None:
            if feature not in self.acc_dict:
                self.acc_dict[feature] = {'y_true': y_true, 'y_pred': y_pred, 'scores': scores}
            else:
                self.acc_dict[feature]['y_true'] = torch.cat([self.acc_dict[feature]['y_true'], y_true])
                self.acc_dict[feature]['y_pred'] = torch.cat([self.acc_dict[feature]['y_pred'], y_pred])
                self.acc_dict[feature]['scores'] = torch.cat([self.acc_dict[feature]['scores'], scores])
        else:
            if feature not in self.acc_dict:
                self.acc_dict[feature] = {'y_true': y_true, 'y_pred': y_pred}
            else:
                self.acc_dict[feature]['y_true'] = torch.cat([self.acc_dict[feature]['y_true'], y_true])
                self.acc_dict[feature]['y_pred'] = torch.cat([self.acc_dict[feature]['y_pred'], y_pred])

    def compute_val_descriptor(self, vals):
        descriptors = np.stack([vals.mean(axis=0), vals.std(axis=0), vals.min(axis=0)[0], vals.max(axis=0)[0]])
        return descriptors

    def compute_vals_similarity(self, vals_descriptors1, vals_descriptors2):
        # per column compute similarity of time-series
        dist = torch.tensor(np.linalg.norm(vals_descriptors1 - vals_descriptors2, axis=0).mean())
        return 1 - torch.sigmoid(dist)  # in which range is this value? in test graph: 0.01 - 0.42 for half features, 0.02 - 0.4 for full features

    def compute_dem_similarity(self, node1_age, node2_age, node1_sex, node2_sex, node1_apoe, node2_apoe):
        # first un-normalize age
        node1_age = (node1_age * (89.0 - 55.0)) + 55.0
        node2_age = (node2_age * (89.0 - 55.0)) + 55.0
        age_diff = abs(node1_age - node2_age).item()
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

    def get_model_prediction(self, item, vals_final_mask, treat_mask, mask=None):
        item = item.to(self.device)
        if mask is None:
            mask = item.val_mask if self.not_use_dev else item.dev_mask
        y_pred_vals, y_pred_treat = self(item)
        y_true_vals, y_true_treat, _ = item.y

        if self.gcn or self.mlp or self.gat or self.graphsage or self.gin:
            y_pred_vals = y_pred_vals.view(len(y_true_vals), -1, 24, 56)
            y_pred_treat = y_pred_treat.view(len(y_true_treat), -1, 24, 16)
        else:
            y_pred_vals = y_pred_vals.view(y_pred_vals.shape[0], y_pred_vals.shape[1], 24, 56)
            y_pred_treat = y_pred_treat.view(y_pred_treat.shape[0], y_pred_treat.shape[1], 24, 16)

        update_mask_vals = vals_final_mask[mask]
        update_mask_treat = treat_mask[mask]

        y_pred_vals, y_pred_treat = y_pred_vals[mask][update_mask_vals], torch.sigmoid(
            y_pred_treat[mask][update_mask_treat])
        y_true_vals, y_true_treat = y_true_vals[mask][update_mask_vals], y_true_treat[mask][update_mask_treat]

        return y_pred_vals, y_pred_treat, y_true_vals, y_true_treat

    def compute_metrics(self, y_true_vals, y_true_treat, y_pred_vals, y_pred_treat):
        rmse = mean_squared_error(y_true_vals.detach().cpu(), y_pred_vals.detach().cpu(), squared=False)
        r2 = r2_score(y_true_vals.detach().cpu(), y_pred_vals.detach().cpu())
        # print(r2)
        f1 = f1_score(y_true_treat.int().detach().cpu(), y_pred_treat.detach().cpu(), average='macro')
        return rmse, f1, r2

    '''As for mimic validation performance is stable (much more validation nodes) computing this is not necessary'''

    def mimic_masking_eval(self, y_pred_vals, y_pred_treat, y_true_vals, y_true_treat, split, suffix=""):
        # for measurements
        rmse, f1, _ = self.compute_metrics(y_true_vals=y_true_vals, y_pred_vals=y_pred_vals, y_true_treat=y_true_treat,
                                           y_pred_treat=y_pred_treat)
        self.log(f'{split}_measurement_rmse{suffix}_{self.task}', rmse, sync_dist=True, on_epoch=True, on_step=False)

        # for treatments
        acc = accuracy_score(y_true=y_true_treat.cpu().numpy(), y_pred=y_pred_treat.cpu().numpy())
        self.log(f'{split}_treat{suffix}_acc_{self.task}', acc, sync_dist=True, on_epoch=True, on_step=False)
        self.log(f'{split}_treat{suffix}_f1_{self.task}', f1, sync_dist=True, on_epoch=True, on_step=False)
        self.log(f'{split}_score{suffix}_avg_{self.task}', ((1 - rmse) + f1) / 2, sync_dist=True, on_epoch=True, on_step=False)
        return rmse, f1, acc

    def mimic_treat_pred_eval(self, y_pred, y_true, split, suffix=""):
        f1 = f1_score(y_true.int().detach().cpu(), y_pred.detach().cpu(), average='macro', zero_division=1)
        acc = accuracy_score(y_true=y_true.flatten().cpu().numpy(), y_pred=y_pred.flatten().cpu().numpy())
        self.log(f'{split}_treat{suffix}_acc_{self.task}', acc, sync_dist=True, on_epoch=True, on_step=False)
        self.log(f'{split}_treat{suffix}_f1_{self.task}', f1, sync_dist=True, on_epoch=True, on_step=False)
        return f1, acc

    def eval_epoch_end(self, outputs, split):
        update_mask = outputs[0]['update_mask']

        if self.dataset_name == 'tadpole':

            if self.mask_all_tadpole:
                y_pred_cont = torch.cat([i['y_pred'][1] for i in outputs]).cpu()
                y_true_cont = torch.cat([i['y_true'][1] for i in outputs]).cpu()

                # update masks
                cont_update_mask = [update_mask[:, 7:]]

                # continous features
                rmse = mean_squared_error(y_true_cont[cont_update_mask].detach().cpu(), y_pred_cont[0][cont_update_mask].detach().cpu(),
                                          squared=False)
                self.log(f'{split}_rmse_avg', rmse, sync_dist=True, on_epoch=True, on_step=False)

            # discrete features
            start = 0
            macro_acc = 0.
            acc_margins = [None, [2, 4], [5, 15], [2], [5]]
            clinical_margins = [0, 4, 15, 2, 5]
            saving_metric = 0.0
            saving_feat_count = 0.0
            for feature_idx, pred_split_idx in enumerate(self.pred_split_idxs):
                class_count = pred_split_idx - start
                start = pred_split_idx
                if self.mask_all_tadpole:
                    y_pred_feat = torch.cat([i['y_pred'][0][feature_idx] for i in outputs])
                    y_true_feat = torch.cat([i['y_true'][0][:, feature_idx] for i in outputs])
                else:
                    y_pred_feat = torch.cat([i['y_pred'][feature_idx] for i in outputs])
                    y_true_feat = torch.cat([i['y_true'][:, feature_idx] for i in outputs])
                feat_update_mask = update_mask[:, feature_idx + 1]

                # compute "saving metric"
                if len(y_pred_feat[feat_update_mask]) > 0:
                    input_dict = {"y_true": y_true_feat[feat_update_mask], "y_pred": y_pred_feat[feat_update_mask]}
                    feat_acc = self.evaluator.eval(input_dict, margin=clinical_margins[feature_idx])['acc']
                    saving_metric += feat_acc
                    saving_feat_count += 1

                if split == 'val':
                    self.add_to_acc_dict(feature_idx, y_true_feat[feat_update_mask], y_pred_feat[feat_update_mask], scores=None)
                    if feature_idx == 0:
                        self.acc_accumulation_count += 1

                if split != 'val' or (self.acc_accumulation_count + 1) % self.acc_accumulation == 0:
                    if split == 'val':
                        input_dict_bin = {"y_true": self.acc_dict[feature_idx]['y_true'], "y_pred": self.acc_dict[feature_idx]['y_pred']}
                    else:
                        input_dict_bin = {"y_true": y_true_feat[feat_update_mask], "y_pred": y_pred_feat[feat_update_mask]}

                    if len(input_dict_bin['y_true']) > 0:
                        feat_acc = self.evaluator.eval(input_dict_bin)['acc']
                        self.log(f'{split}_acc_feat_{feature_idx}_{class_count}_classes', feat_acc, sync_dist=True, on_epoch=True, on_step=False)

                        margin = acc_margins[feature_idx]
                        if margin:
                            for m in margin:
                                margin_feat_acc = self.evaluator.eval(input_dict_bin, margin=m)['acc']
                                self.log(f'{split}_acc_feat_{feature_idx}_margin_{m}', margin_feat_acc, sync_dist=True, on_epoch=True, on_step=False)

                        macro_acc += feat_acc

            # reset accuracy dict when val accuracies were recorded
            if split == 'val' and (self.acc_accumulation_count + 1) % self.acc_accumulation == 0:
                self.acc_dict = {}

            self.log(f'{split}_macro_acc_avg_margins', saving_metric / len(self.pred_split_idxs), sync_dist=True, on_epoch=True, on_step=False)
            if split != 'val' or (self.acc_accumulation_count + 1) % self.acc_accumulation == 0:
                self.log(f'{split}_macro_acc_avg', macro_acc / len(self.pred_split_idxs), sync_dist=True, on_epoch=True, on_step=False)

            if self.current_epoch == self.end_epoch:
                torch.save(self.state_dict(), f'exps/tadpole_mask/lightning_logs/{self.run_name}/checkpoints/epoch{self.current_epoch}.pt')

        elif self.dataset_name == 'tadpole_class':
            if update_mask is not None:
                y_pred = torch.cat([i['y_pred'] for i in outputs])[update_mask]
                scores = torch.cat([i['y_scores'] for i in outputs])[update_mask]
                y_true = torch.cat([i['y_true'] for i in outputs])[update_mask]
            else:
                y_pred = torch.cat([i['y_pred'] for i in outputs])
                scores = torch.cat([i['y_scores'] for i in outputs])[update_mask]
                y_true = torch.cat([i['y_true'] for i in outputs])

            if split == 'val':
                self.add_to_acc_dict(0, y_true, y_pred, scores)
                self.acc_accumulation_count += 1

            if split != 'val' or (self.acc_accumulation_count + 1) % self.acc_accumulation == 0:
                if split == 'val':
                    input_dict_bin = {"y_true": self.acc_dict[0]['y_true'], "y_pred": self.acc_dict[0]['y_pred']}
                else:
                    input_dict_bin = {"y_true": y_true, "y_pred": y_pred}

                acc = self.evaluator.eval(input_dict_bin)['acc']
                f1 = f1_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average='macro')
                auroc = roc_auc_score(y_true.cpu().numpy(), torch.softmax(scores[0], dim=1).detach().cpu().numpy(), average='macro',
                                      multi_class='ovr')
                if self.cross_val:
                    if split == 'train':
                        self.train_acc.append(acc)
                        self.train_f1.append(f1)
                        self.train_auroc.append(auroc)
                    elif split == 'val':
                        self.val_acc.append(acc)
                        self.val_f1.append(f1)
                        self.val_auroc.append(auroc)
                else:
                    self.log(f'{split}_acc_disease_pred', acc, sync_dist=True, on_epoch=True, on_step=False)
                    self.log(f'{split}_f1_disease_pred', f1, sync_dist=True, on_epoch=True, on_step=False)
                    self.log(f'{split}_auroc_disease_pred', auroc, sync_dist=True, on_epoch=True, on_step=False)

                    if split == 'val' and auroc > self.best_val_score_tadpole:  # force model to not just be saved without being trained (for small label ratios)
                        self.best_val_score_tadpole = auroc
                        self.log(f'{split}_best_auroc_disease_pred', self.best_val_score_tadpole, sync_dist=True, on_epoch=True, on_step=False)

            # reset accuracy dict when val accuracies were recorded
            if split == 'val' and (self.acc_accumulation_count + 1) % self.acc_accumulation == 0:
                self.acc_dict = {}

        elif self.dataset_name == 'mimic':
            if self.task == 'multi_task_pt':
                # extract predictions for pre_mask
                try:
                    y_pred_vals_pre_mask = torch.cat([i['y_pred'][0] for i in outputs if i['mask_task'] == 'pre_mask']).cpu()
                    y_pred_treat_pre_mask = torch.cat([i['y_pred'][1] for i in outputs if i['mask_task'] == 'pre_mask']).cpu()
                    y_true_vals_pre_mask = torch.cat([i['y_true'][0] for i in outputs if i['mask_task'] == 'pre_mask']).cpu()
                    y_true_treat_pre_mask = torch.cat([i['y_true'][1] for i in outputs if i['mask_task'] == 'pre_mask']).cpu()
                except Exception as e:  # if no samples of this type were processed
                    y_pred_vals_pre_mask = []

                # extract predictions for patient_prediction
                try:
                    y_pred_vals_PM = torch.cat([i['y_pred'][0] for i in outputs if i['mask_task'] == 'PM']).cpu()
                    y_pred_treat_PM = torch.cat([i['y_pred'][1] for i in outputs if i['mask_task'] == 'PM']).cpu()
                    y_true_vals_PM = torch.cat([i['y_true'][0] for i in outputs if i['mask_task'] == 'PM']).cpu()
                    y_true_treat_PM = torch.cat([i['y_true'][1] for i in outputs if i['mask_task'] == 'PM']).cpu()
                except Exception as e:
                    y_pred_vals_PM = []

                # extract predictions for block wise masking
                try:
                    y_pred_vals_BM = torch.cat([i['y_pred'][0] for i in outputs if i['mask_task'] == 'BM']).cpu()
                    y_pred_treat_BM = torch.cat([i['y_pred'][1] for i in outputs if i['mask_task'] == 'BM']).cpu()
                    y_true_vals_BM = torch.cat([i['y_true'][0] for i in outputs if i['mask_task'] == 'BM']).cpu()
                    y_true_treat_BM = torch.cat([i['y_true'][1] for i in outputs if i['mask_task'] == 'BM']).cpu()
                except Exception as e:
                    y_pred_vals_BM = []

                # extract predictions for treatment prediction
                try:
                    y_pred_TP = torch.cat([i['y_pred'] for i in outputs if i['mask_task'] == 'TP'])
                    y_true_TP = torch.cat([i['y_true'] for i in outputs if i['mask_task'] == 'TP'])
                except Exception as e:
                    y_pred_TP = []

                # evaluate for all tasks where predictions were made
                if len(y_pred_vals_pre_mask) > 0:  # feature masking (FM)
                    rmse_FM, f1_FM, acc_FM = self.mimic_masking_eval(y_pred_vals_pre_mask, y_pred_treat_pre_mask, y_true_vals_pre_mask,
                                                                     y_true_treat_pre_mask, split, suffix="FM")

                if len(y_pred_vals_PM) > 0:
                    rmse_PM, f1_PM, acc_PM = self.mimic_masking_eval(y_pred_vals_PM, y_pred_treat_PM, y_true_vals_PM, y_true_treat_PM, split,
                                                                     suffix="PM")

                if len(y_pred_vals_BM) > 0:
                    rmse_BM, f1_BM, acc_BM = self.mimic_masking_eval(y_pred_vals_BM, y_pred_treat_BM, y_true_vals_BM, y_true_treat_BM, split,
                                                                     suffix="BM")

                if len(y_pred_TP) > 0:
                    f1_TP, acc_TP = self.mimic_treat_pred_eval(y_pred_TP, y_true_TP, split, suffix="TP")

                # save model if score is higher than previous best
                # only save if samples from all tasks existed
                if len(y_pred_vals_pre_mask) > 0 and len(y_pred_vals_PM) > 0 and len(y_pred_vals_BM) > 0 and len(y_pred_TP) > 0 and (
                        (1 - rmse_FM) + f1_FM) / 2 + ((1 - rmse_PM) + f1_PM) / 2 + ((1 - rmse_BM) + f1_BM) / 2 + f1_TP > self.best_val_score:
                    self.best_val_score = ((1 - rmse_FM) + f1_FM) / 2 + ((1 - rmse_PM) + f1_PM) / 2 + ((1 - rmse_BM) + f1_BM) / 2 + f1_TP
                    torch.save(self.state_dict(), f'exps/mimic_pre_mask/lightning_logs/{self.run_name}/checkpoints/best_score.pt')
                    f = open(f'exps/mimic_pre_mask/lightning_logs/{self.run_name}/checkpoints/best_score.txt', "a+")
                    f.write(f"Epoch {self.current_epoch}: {self.best_val_score}\n")
                    f.close()

            elif self.task == 'pre_mask' or self.task == 'patient_prediction':
                y_pred_vals = torch.cat([i['y_pred'][0] for i in outputs]).cpu()
                y_pred_treat = torch.cat([i['y_pred'][1] for i in outputs]).cpu()
                y_true_vals = torch.cat([i['y_true'][0] for i in outputs]).cpu()
                y_true_treat = torch.cat([i['y_true'][1] for i in outputs]).cpu()

                rmse, f1, acc = self.mimic_masking_eval(y_pred_vals, y_pred_treat, y_true_vals, y_true_treat, split)

            elif self.task == 'treat_prediction':
                y_pred = torch.cat([i['y_pred'] for i in outputs])
                y_true = torch.cat([i['y_true'] for i in outputs])
                f1, acc = self.mimic_treat_pred_eval(y_pred, y_true, split)

                if f1 > self.best_val_score:
                    self.best_val_score = f1
                    torch.save(self.state_dict(), f'exps/mimic_pre_mask/lightning_logs/{self.run_name}/checkpoints/best_score.pt')
                    f = open(f'exps/mimic_pre_mask/lightning_logs/{self.run_name}/checkpoints/best_score.txt', "a+")
                    f.write(f"Epoch {self.current_epoch}: {f1}\n")
                    f.close()

            else:
                acc = None
                f1 = None
                auroc = None
                class_acc = None
                y_pred = torch.cat([i['y_pred'] for i in outputs])
                y_scores = torch.cat([i['y_scores'] for i in outputs])
                y_true = torch.cat([i['y_true'] for i in outputs])

                input_dict_bin = {"y_true": y_true, "y_pred": y_pred}
                if self.task == 'acu':

                    y_scores_clean = torch.softmax(y_scores[:, y_true.unique().long()], dim=1)
                    acc = self.evaluator.eval(input_dict_bin)['acc']
                    f1 = f1_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average='macro')
                    # y_scores_clean = torch.softmax(input=y_scores[:, y_true.unique().long()], dim=1)  # only keep classes which are in ground truth for this batch
                    if y_scores_clean.shape[1] == 2:  # only two classes occur
                        y_scores_clean = y_scores_clean[:, 1]

                    try:
                        auroc = roc_auc_score(y_true=y_true.cpu().numpy(), y_score=y_scores_clean.detach().cpu().numpy(), average='macro',
                                              multi_class='ovr')
                    except Exception as e:
                        print(e)

                else:  # los or rea
                    acc = self.evaluator.eval(input_dict_bin)['acc']
                    f1 = f1_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average='macro')
                    try:
                        auroc = roc_auc_score(y_true.cpu().numpy(), y_scores[:, 1].detach().cpu().numpy(), average='macro')  # scores for higher class
                    except Exception as e:
                        print(e)
                    # same results with scores or probabilities
                if acc is not None:
                    self.log(f'{split}_acc_{self.task}', acc, sync_dist=True, on_epoch=True, on_step=False)
                if f1 is not None:
                    self.log(f'{split}_f1_{self.task}', f1, sync_dist=True, on_epoch=True, on_step=False)
                if auroc is not None:
                    self.log(f'{split}_auroc_{self.task}', auroc, sync_dist=True, on_epoch=True, on_step=False)

                if split == 'val' and auroc > self.best_val_score and self.current_epoch > 3:  # force model to not just be saved without being trained (for small label ratios)
                    self.best_val_score = auroc
                    if "optuna" not in self.run_name and "test" not in self.run_name:  # for tests we don't need to save the model
                        torch.save(self.state_dict(), f'exps/mimic/lightning_logs/{self.run_name}/checkpoints/best_auroc.pt')
                        f = open(f'exps/mimic/lightning_logs/{self.run_name}/checkpoints/best_auroc.txt', "a+")
                        f.write(f"Epoch {self.current_epoch}: {auroc}\n")
                        f.close()

                if split == 'val':
                    self.log(f'{split}_best_auroc_{self.task}', self.best_val_score, sync_dist=True, on_epoch=True, on_step=False)

        elif self.dataset_name == 'sepsis':
            auroc = None
            y_pred = torch.cat([i['y_pred'] for i in outputs])
            y_scores = torch.cat([i['y_scores'] for i in outputs])
            y_true = torch.cat([i['y_true'] for i in outputs])

            input_dict_bin = {"y_true": y_true, "y_pred": y_pred}

            acc = self.evaluator.eval(input_dict_bin)['acc']
            f1 = f1_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average='macro')
            try:
                auroc = roc_auc_score(y_true.cpu().numpy(), y_scores[:, 1].detach().cpu().numpy(), average='macro')  # scores for higher class
            except Exception as e:
                print(e)

            self.log(f'{split}_acc_sepsis', acc, sync_dist=True, on_epoch=True, on_step=False)
            self.log(f'{split}_f1_sepsis', f1, sync_dist=True, on_epoch=True, on_step=False)
            if auroc is not None:
                self.log(f'{split}_auroc_sepsis', auroc, sync_dist=True, on_epoch=True, on_step=False)

            if split == 'val' and f1 > self.best_val_score and self.current_epoch > 3:
                self.best_val_score = f1
                if "test" not in self.run_name:  # for tests we don't need to save the model
                    torch.save(self.state_dict(), f'exps/sepsis/lightning_logs/{self.run_name}/checkpoints/best_f1.pt')
                    f = open(f'exps/sepsis/lightning_logs/{self.run_name}/checkpoints/best_f1.txt', "a+")
                    f.write(f"Epoch {self.current_epoch}: {auroc}\n")
                    f.close()

    def validation_step(self, batched_data, batch_idx):
        return self.eval_step(batched_data=batched_data, batch_idx=batch_idx, split='val')

    def validation_epoch_end(self, outputs):
        self.eval_epoch_end(outputs=outputs, split='val')

    def test_step(self, batched_data, batch_idx):
        return self.eval_step(batched_data=batched_data, batch_idx=batch_idx, split='test')

    def test_epoch_end(self, outputs):
        self.eval_epoch_end(outputs=outputs, split='test')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad], lr=self.peak_lr, weight_decay=self.weight_decay)

        if self.dataset_name == 'mimic':
            lr_scheduler = {
                'scheduler': MultiStepLR(
                    optimizer,
                    milestones=[750, 1000],
                    gamma=0.333 if self.end_lr != self.peak_lr else 1
                ),
                'name': 'learning_rate',
                'interval': 'step',
                'frequency': 1,
            }
        elif self.dataset_name == 'sepsis':
            lr_scheduler = {
                'scheduler': MultiStepLR(
                    optimizer,
                    milestones=[],
                    gamma=0.0
                ),
                'name': 'learning_rate',
                'interval': 'step',
                'frequency': 1,
            }
        else:
            lr_scheduler = {
                'scheduler': PolynomialDecayLR(
                    optimizer,
                    warmup_updates=self.warmup_updates,
                    fixed_updates=self.fixed_updates,
                    tot_updates=self.tot_updates,
                    lr=self.peak_lr,
                    end_lr=self.end_lr,
                    power=1.0,
                ),
                'name': 'learning_rate',
                'interval': 'step',
                'frequency': 1,
            }
        return [optimizer], [lr_scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Graphormer")
        parser.add_argument('--gcn', action='store_true', default=False)
        parser.add_argument('--gat', action='store_true', default=False)
        parser.add_argument('--graphsage', action='store_true', default=False)
        parser.add_argument('--gin', action='store_true', default=False)
        parser.add_argument('--mlp', action='store_true', default=False)
        parser.add_argument('--skip_transformer', action='store_true', default=False)
        parser.add_argument('--n_layers', type=int, default=12)
        parser.add_argument('--num_heads', type=int, default=32)
        parser.add_argument('--hidden_dim', type=int, default=512)
        parser.add_argument('--ffn_dim', type=int, default=512)
        parser.add_argument('--intput_dropout_rate', type=float, default=0.1)
        parser.add_argument('--dropout_rate', type=float, default=0.1)
        parser.add_argument('--weight_decay', type=float, default=0.01)
        parser.add_argument('--attention_dropout_rate',
                            type=float, default=0.1)
        parser.add_argument('--checkpoint_path', type=str, default='')
        parser.add_argument('--pretraining_path', type=str, default='')
        parser.add_argument('--runname', type=str, default='debug')
        parser.add_argument('--warmup_updates', type=int, default=60000)
        parser.add_argument('--tot_updates', type=int, default=1000000)
        parser.add_argument('--peak_lr', type=float, default=2e-4)
        parser.add_argument('--end_lr', type=float, default=1e-9)
        parser.add_argument('--edge_type', type=str, default='multi_hop')
        parser.add_argument('--edge_vars', type=str, default='full_edge_full_node')
        parser.add_argument('--validate', action='store_true', default=False)
        parser.add_argument('--cross_val', action='store_true', default=False)
        parser.add_argument('--fold', type=int, default=2)
        parser.add_argument('--drop_val_patients', action='store_true', default=False)
        parser.add_argument('--task', type=str, default='')
        parser.add_argument('--num_graphs', type=int, default=43)
        parser.add_argument('--rotation', type=int, default=0)
        parser.add_argument('--label_ratio', type=float, default=1.0)
        parser.add_argument('--test', action='store_true', default=False)
        parser.add_argument('--flag', action='store_true')
        parser.add_argument('--flag_m', type=int, default=3)
        parser.add_argument('--flag_step_size', type=float, default=1e-3)
        parser.add_argument('--flag_mag', type=float, default=1e-3)
        parser.add_argument('--pad_mode', type=str, default='original')
        parser.add_argument('--save_last', action='store_true', default=False)
        parser.add_argument('--not_use_dev', action='store_false', default=True)
        parser.add_argument('--block_size', type=int, default=6)
        parser.add_argument('--k', type=int, default=5)
        parser.add_argument('--mask_ratio', type=float, default=0.15)
        parser.add_argument('--loss_weighting', type=str, default='none')
        parser.add_argument('--compute_results', action='store_true', default=False)
        parser.add_argument('--mask_all', action='store_true', default=False)
        parser.add_argument('--use_simple_acu_task', action='store_true', default=False)
        parser.add_argument('--set_id', type=str, default='A')
        parser.add_argument('--freeze_graphormer', action='store_true', default=False)
        parser.add_argument('--PT_transformer', action='store_true', default=True)
        return parent_parser


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None, attn_mask=None):  # q=k=v = x
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale  # normalization
        x = torch.matmul(q, k)  # [b, h, q_len, k_len] # x contains attention_scores
        if attn_bias is not None:
            x = x + attn_bias

        x = torch.softmax(x, dim=3)  # x contains attention_probs
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn] # values are multiplied with attention_probs

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None, attn_mask=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias=attn_bias, attn_mask=attn_mask)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x
