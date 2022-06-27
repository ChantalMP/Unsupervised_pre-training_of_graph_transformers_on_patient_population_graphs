import json
import os
import random

import numpy as np
from tqdm import tqdm


def create_subgraphs_mimic_random(rotation=1):
    # load patient id lists for all splits in this rotation with json
    with open(f'data/mimic-iii-0/rotations/train_patients_rot_{rotation}.json', 'r') as f:
        train_patients = json.load(f)
    with open(f'data/mimic-iii-0/rotations/val_patients_rot_{rotation}.json', 'r') as f:
        val_patients = json.load(f)
    with open(f'data/mimic-iii-0/rotations/test_patients_rot_{rotation}.json', 'r') as f:
        test_patients = json.load(f)

    all_patients = {'train': [], 'val': [], 'test': []}
    for split in ['val', 'train', 'test']:
        for folder in tqdm(os.listdir(f'data/mimic-iii-0/{split}')):
            if not folder.startswith('patient'):
                continue
            patient_id = folder.split('_')[1]

            if int(patient_id) in train_patients:
                real_split = 'train'
            elif int(patient_id) in val_patients:
                real_split = 'val'
            elif int(patient_id) in test_patients:
                real_split = 'test'
            else:
                raise ValueError(f'{patient_id} not in any split')

            all_patients[real_split].append([patient_id, real_split, split])

    # 21.800 patients, to form batches of 500 patients split in 43 groups (506 patients per group)
    # shuffle all patients list randomly
    random.shuffle(all_patients['train'])
    random.shuffle(all_patients['val'])
    random.shuffle(all_patients['test'])
    train_splits = np.array_split(all_patients['train'], 43)
    val_splits = np.array_split(all_patients['val'], 43)
    test_splits = np.array_split(all_patients['test'], 43)

    patients_subsets = []
    for train_set, val_set, test_set in zip(train_splits, val_splits, test_splits):
        patients_subsets.append(np.concatenate([train_set, val_set, test_set]))

    # create dir data/mimic-iii-0/train/raw/rot{rotation}/ if it not exists
    if not os.path.exists(f'data/mimic-iii-0/train/raw/rot{rotation}'):
        os.makedirs(f'data/mimic-iii-0/train/raw/rot{rotation}')
    # create dir data/mimic-iii-0/val/raw/rot{rotation}/ if it not exists
    if not os.path.exists(f'data/mimic-iii-0/val/raw/rot{rotation}'):
        os.makedirs(f'data/mimic-iii-0/val/raw/rot{rotation}')
    # create dir data/mimic-iii-0/test/raw/rot{rotation}/ if it not exists
    if not os.path.exists(f'data/mimic-iii-0/test/raw/rot{rotation}'):
        os.makedirs(f'data/mimic-iii-0/test/raw/rot{rotation}')

    for idx, subset in tqdm(enumerate(patients_subsets)):
        # create mask of size len(subset) were first and 50 elements are False
        graph_list = [s.tolist() for idx, s in enumerate(subset) if s[1] == 'train']
        with open(f'data/mimic-iii-0/train/raw/rot{rotation}/random_graph_subset_{idx}.json', 'w') as f:
            json.dump(graph_list, f)
        graph_list = [s.tolist() for idx, s in enumerate(subset) if s[1] == 'train' or s[1] == 'val']
        with open(f'data/mimic-iii-0/val/raw/rot{rotation}/random_graph_subset_{idx}.json', 'w') as f:
            json.dump(graph_list, f)
        graph_list = [s.tolist() for idx, s in enumerate(subset) if s[1] == 'train' or s[1] == 'test' or s[1] == 'val']
        with open(f'data/mimic-iii-0/test/raw/rot{rotation}/random_graph_subset_{idx}.json', 'w') as f:
            json.dump(graph_list, f)


if __name__ == '__main__':
    create_subgraphs_mimic_random()
