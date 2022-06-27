import json
import os
import random

import numpy as np
from tqdm import tqdm


def create_train_val_test_split(set='A'):
    patients = []
    filepath = f"data/sepsis/training_set{set}/"  # path to training set A
    for filename in os.listdir(filepath):
        if filename.endswith(".psv"):
            patient = filename.split("p")[1]
            patient = patient.split(".")[0]
            patients.append(patient)

    # shuffle patients
    random.shuffle(patients)
    # split into train, val, test
    train_patients = patients[:int(len(patients) * 0.8)]
    val_patients = patients[int(len(patients) * 0.8):int(len(patients) * 0.9)]
    test_patients = patients[int(len(patients) * 0.9):]

    # save train split to json
    with open(f"data/sepsis/train_set{set}.json", "w") as outfile:
        json.dump(train_patients, outfile)
    # save val split to json
    with open(f"data/sepsis/val_set{set}.json", "w") as outfile:
        json.dump(val_patients, outfile)
    # save test split to json
    with open(f"data/sepsis/test_set{set}.json", "w") as outfile:
        json.dump(test_patients, outfile)


def create_subgraphs_random(set='A'):
    # load patient id lists for all splits in this rotation with json
    with open(f'data/sepsis/train_set{set}.json', 'r') as f:
        train_patients = json.load(f)
    with open(f'data/sepsis/val_set{set}.json', 'r') as f:
        val_patients = json.load(f)
    with open(f'data/sepsis/test_set{set}.json', 'r') as f:
        test_patients = json.load(f)

    # 20336 patients for A, 20000 for B , to form batches of 500 patients split in 40 groups (508 (A)/ 500 (B) patients per group)
    # shuffle all patients list randomly
    random.shuffle(train_patients)
    random.shuffle(val_patients)
    random.shuffle(test_patients)
    train_splits = np.array_split(train_patients, 40)
    val_splits = np.array_split(val_patients, 40)
    test_splits = np.array_split(test_patients, 40)

    # create dir data/mimic-iii-0/train/raw/rot{rotation}/ if it not exists
    if not os.path.exists(f'data/sepsis/train/raw/training_set{set}'):
        os.makedirs(f'data/sepsis/train/raw/training_set{set}')
    if not os.path.exists(f'data/sepsis/val/raw/training_set{set}'):
        os.makedirs(f'data/sepsis/val/raw/training_set{set}')
    if not os.path.exists(f'data/sepsis/test/raw/training_set{set}'):
        os.makedirs(f'data/sepsis/test/raw/training_set{set}')

    for idx in tqdm(range(len(train_splits))):
        train_list = [list((x, 'train')) for x in train_splits[idx]]
        val_list = [list((x, 'val')) for x in val_splits[idx]]
        test_list = [list((x, 'test')) for x in test_splits[idx]]

        with open(f'data/sepsis/train/raw/training_set{set}/random_graph_subset_{idx}.json', 'w') as f:
            json.dump(train_list, f)

        graph_list = train_list + val_list
        with open(f'data/sepsis/val/raw/training_set{set}/random_graph_subset_{idx}.json', 'w') as f:
            json.dump(graph_list, f)

        graph_list = train_list + val_list + test_list
        with open(f'data/sepsis/test/raw/training_set{set}/random_graph_subset_{idx}.json', 'w') as f:
            json.dump(graph_list, f)


def create_val_test_graphs(set='A'):
    # load patient id lists for all splits in this rotation with json
    with open(f'data/sepsis/val_set{set}.json', 'r') as f:
        val_patients = json.load(f)
    with open(f'data/sepsis/test_set{set}.json', 'r') as f:
        test_patients = json.load(f)

    # A: 2034 val/test patients -> 4 graphs a 508 nodes, B: 2000 val/test patients -> 4 graphs a 500 nodes
    # shuffle all patients list randomly
    random.shuffle(val_patients)
    random.shuffle(test_patients)
    val_splits = np.array_split(val_patients, 4)
    test_splits = np.array_split(test_patients, 4)

    for idx in tqdm(range(len(val_splits))):
        val_list = [list((x, 'val')) for x in val_splits[idx]]
        with open(f'data/sepsis/val/raw/training_set{set}/random_val_graph_subset_{idx}.json', 'w') as f:
            json.dump(val_list, f)

    for idx in tqdm(range(len(test_splits))):
        test_list = [list((x, 'test')) for x in test_splits[idx]]
        with open(f'data/sepsis/test/raw/training_set{set}/random_test_graph_subset_{idx}.json', 'w') as f:
            json.dump(test_list, f)


def create_cross_val_graphs(set='A'):
    # load patient id lists for all splits in this rotation with json
    with open(f'data/sepsis/train_set{set}.json', 'r') as f:
        train_patients = json.load(f)
    with open(f'data/sepsis/val_set{set}.json', 'r') as f:
        val_patients = json.load(f)
    with open(f'data/sepsis/test_set{set}.json', 'r') as f:
        test_patients = json.load(f)

    # 20336 patients for A, 20000 for B , to form batches of 500 patients split in 40 groups (508 (A)/ 500 (B) patients per group)
    # shuffle all patients list randomly
    patients = train_patients + val_patients + test_patients
    random.shuffle(patients)

    splits = np.array_split(patients, 40)

    # create dir data/mimic-iii-0/train/raw/rot{rotation}/ if it not exists
    if not os.path.exists(f'data/sepsis/raw/cross_val_{set}'):
        os.makedirs(f'data/sepsis/raw/cross_val_{set}')

    for idx in tqdm(range(len(splits))):
        patient_list = [list((x, 'train')) for x in splits[idx]]

        with open(f'data/sepsis/raw/cross_val_{set}/random_graph_subset_{idx}.json', 'w') as f:
            json.dump(patient_list, f)


if __name__ == '__main__':
    # create_train_val_test_split(set='B')
    # create_subgraphs_random(set='B')
    # create_cross_val_graphs(set='B')
    pass
