import json
import os
import statistics

import numpy as np
import torch


def generate_cross_val_split():
    idxs = np.array(range(564))
    # split in 10 parts, always one as val, rest as train
    np.random.shuffle(idxs)
    splits = np.array_split(idxs, 10)
    for idx, split in enumerate(splits):
        np.save(f'data/tadpole/split/cross_val/val_idxs_fold{idx}.npy', split)
        rest = np.array([idx for idx in idxs if idx not in split])
        np.save(f'data/tadpole/split/cross_val/train_idxs_fold{idx}.npy', rest)


def generate_stratified_cross_val_split(fold=0):
    data = torch.load(f'data/tadpole/processed/tadpole_graph_class_fold{fold}.pt')  # all data
    train_idxs = data.node_id
    labels_train = data.y
    labels0 = np.array([idx for idx, label in zip(train_idxs, labels_train) if label == 0])
    labels1 = np.array([idx for idx, label in zip(train_idxs, labels_train) if label == 1])
    labels2 = np.array([idx for idx, label in zip(train_idxs, labels_train) if label == 2])
    np.random.shuffle(labels0)
    np.random.shuffle(labels1)
    np.random.shuffle(labels2)

    splits0 = np.array_split(labels0, 10)
    splits1 = np.array_split(labels1, 10)
    splits2 = np.array_split(labels2, 10)

    for idx, (split0, split1, split2) in enumerate(zip(splits0, splits1, splits2)):
        val_split = np.concatenate([split0, split1, split2])
        np.save(f'data/tadpole/split/cross_val/val_idxs_fold{idx}_strat.npy', val_split)
        rest = np.array([idx for idx in train_idxs if idx not in val_split])
        np.save(f'data/tadpole/split/cross_val/train_idxs_fold{idx}_strat.npy', rest)


def compute_mean_and_std(values):
    mean = statistics.mean(values)
    std = statistics.stdev(values)
    print(mean, std)


# which training nodes should not be used for updating the model in the missing label case
# currently for fold 2
def generate_labels_to_drop():
    train_idxs = np.load('data/tadpole/split/cross_val/train_idxs_fold2.npy')
    existing_label_ratios = [0.1, 0.25, 0.5, 0.75, 0.9]
    for ratio in existing_label_ratios:
        label_idxs = np.random.choice(train_idxs, size=round(len(train_idxs) * (1 - ratio)))
        np.save(f'data/tadpole/split/label_drop_idxs_fold2_{ratio}.npy', label_idxs)


def generate_labels_to_drop_same_samples(fold):
    train_idxs = np.load(f'data/tadpole/split/cross_val/train_idxs_fold{fold}.npy')
    np.random.shuffle(train_idxs)
    splits = np.array_split(train_idxs, 20)  # create splits of 5% of samples
    label_drop_idxs_09 = np.concatenate([splits[i] for i in range(0, 2)])  # drop 10%
    label_drop_idxs_075 = np.concatenate([splits[i] for i in range(0, 5)])  # drop 25%
    label_drop_idxs_05 = np.concatenate([splits[i] for i in range(0, 10)])  # drop 50%
    label_drop_idxs_025 = np.concatenate([splits[i] for i in range(0, 15)])  # drop 75%
    label_drop_idxs_01 = np.concatenate([splits[i] for i in range(0, 18)])  # drop 90%
    label_drop_idxs_005 = np.concatenate([splits[i] for i in range(0, 19)])  # drop 95%

    np.save(f'data/tadpole/split/label_drop_idxs_fold2_0.9_same_samples.npy', label_drop_idxs_09)
    np.save(f'data/tadpole/split/label_drop_idxs_fold2_0.75_same_samples.npy', label_drop_idxs_075)
    np.save(f'data/tadpole/split/label_drop_idxs_fold2_0.5_same_samples.npy', label_drop_idxs_05)
    np.save(f'data/tadpole/split/label_drop_idxs_fold2_0.25_same_samples.npy', label_drop_idxs_025)
    np.save(f'data/tadpole/split/label_drop_idxs_fold2_0.1_same_samples.npy', label_drop_idxs_01)
    np.save(f'data/tadpole/split/label_drop_idxs_fold2_0.05_same_samples.npy', label_drop_idxs_005)


def generate_labels_to_drop_balanced_tadpole(fold):
    data = torch.load(f'data/tadpole/processed/tadpole_graph_class_drop_val_train_fold{fold}_sim.pt')
    train_idxs = data.node_id
    labels_train = data.y
    labels0 = [idx for idx, label in zip(train_idxs, labels_train) if label == 0]
    labels1 = [idx for idx, label in zip(train_idxs, labels_train) if label == 1]
    labels2 = [idx for idx, label in zip(train_idxs, labels_train) if label == 2]

    for ratio in [0.01, 0.05, 0.1, 0.5]:
        drop0 = np.random.choice(labels0, size=round(len(labels0) * (1 - ratio)), replace=False)
        drop1 = np.random.choice(labels1, size=round(len(labels1) * (1 - ratio)), replace=False)
        drop2 = np.random.choice(labels2, size=round(len(labels2) * (1 - ratio)), replace=False)
        drop = np.concatenate([drop0, drop1, drop2])
        np.save(f'data/tadpole/split/label_drop_idxs_fold{fold}_{ratio}_bal_sim.npy', drop)


def generate_labels_to_drop_mimic(rotation):
    # currently for los
    with open(f'data/mimic-iii-0/rotations/train_patients_rot_{rotation}.json') as f:
        train_idxs = json.load(f)

    for ratio in [0.01, 0.1, 0.05, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        label_idxs = np.random.choice(train_idxs, size=round(len(train_idxs) * (1 - ratio)), replace=False)
        # if folders are not created, create them
        if not os.path.exists(f'data/mimic-iii-0/drop/los'):
            os.makedirs(f'data/mimic-iii-0/drop/los')
        np.save(f'data/mimic-iii-0/drop/los/label_drop_idxs_rot{rotation}_{ratio}.npy', label_idxs)


def print_class_imbalances(fold=0):
    data = torch.load(f'data/tadpole/processed/tadpole_graph_class_drop_val_train_fold{fold}.pt')
    data_all = torch.load(f'data/tadpole/processed/tadpole_graph_class_drop_val_val_fold{fold}.pt')
    labels_train = data.y
    tcount0, tcount1, tcount2 = torch.unique(labels_train, return_counts=True)[1]
    labels_all = data_all.y
    acount0, acount1, acount2 = torch.unique(labels_all, return_counts=True)[1]
    labels_val = data_all.y[data_all.val_mask]
    vcount0, vcount1, vcount2 = torch.unique(labels_val, return_counts=True)[1]

    print(f"Fold: {fold}: \n")
    # print(f"Distribution All: 0: {acount0}, {acount0/len(labels_all)} 1: {acount1}, {acount1/len(labels_all)} 2: {acount2}, {acount2/len(labels_all)}")
    print(
        f"Distribution Train: 0: {tcount0}, {tcount0 / len(labels_train)} 1: {tcount1}, {tcount1 / len(labels_train)} 2: {tcount2}, {tcount2 / len(labels_train)}")
    # print(f"Distribution Val: 0: {vcount0}, {vcount0/len(labels_val)} 1: {vcount1}, {vcount1/len(labels_val)} 2: {vcount2}, {vcount2/len(labels_val)}")

    for ratio in []:
        drop_idxs = np.load(f'data/tadpole/split/label_drop_idxs_fold2_{ratio}_bal.npy')
        drop_pos = [np.where(data.node_id == drop_idx)[0].item() for drop_idx in drop_idxs]
        keep_pos = [i for i in range(len(labels_train)) if i not in drop_pos]
        labels_ratio = labels_train[keep_pos]
        count0, count1, count2 = torch.unique(labels_ratio, return_counts=True)[1]
        print(
            f"Distribution Ratio {ratio}: 0: {count0}, {count0 / len(labels_ratio)} 1: {count1}, {count1 / len(labels_ratio)} 2: {count2}, {count2 / len(labels_ratio)}")


def summarize_acu_task(y):
    # HOME + medical care: 0, 11, 8
    y[y == 0] = -1
    y[y == 11] = -1
    y[y == 8] = -1

    # HOME: 1, 9
    y[y == 1] = 0
    y[y == 9] = 0

    y[y == -1] = 1

    # CARE FACILITY: 13,3,10,2,15,12
    y[y == 13] = 2
    y[y == 3] = 2
    y[y == 10] = 2
    y[y == 2] = 2
    y[y == 15] = 2
    y[y == 12] = 2
    # HOSPITAL: 5,6,7,4,14
    y[y == 5] = 2
    y[y == 6] = 2
    y[y == 7] = 2
    y[y == 4] = 2
    y[y == 14] = 2

    # DEATH
    y[y == 16] = 3
    y[y == 17] = 3

    return y


if __name__ == '__main__':
    pass
