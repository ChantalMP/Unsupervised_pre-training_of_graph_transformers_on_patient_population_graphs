import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd


def calc_sepsis_data_imbalance():
    filepath_A = f"data/sepsis/training_setA/"
    filepath_B = f"data/sepsis/training_setB/"

    total = 0
    septic = 0
    for i, filename in enumerate(os.listdir(filepath_A)):
        if i % 1000 == 0:
            print(i)
        if filename.endswith(".psv"):
            patient = filename.split("p")[1]
            patient = patient.split(".")[0]
            sepsis = pd.read_csv(f'{filepath_A}p{patient}.psv', sep="|")['SepsisLabel']
            total += 1
            if sepsis[sepsis == 1].sum() > 0:
                septic += 1

    print("Percentage of septic patients A: ", septic / total, septic, total)

    totalB = 0
    septicB = 0
    for i, filename in enumerate(os.listdir(filepath_B)):
        if i % 1000 == 0:
            print(i)
        if filename.endswith(".psv"):
            patient = filename.split("p")[1]
            patient = patient.split(".")[0]
            sepsis = pd.read_csv(f'{filepath_B}p{patient}.psv', sep="|")['SepsisLabel']
            totalB += 1
            if sepsis[sepsis == 1].sum() > 0:
                septicB += 1

    print("Percentage of septic patients B: ", septicB / totalB, septicB, totalB)

    print("Percentage of septic patients: ", (septic + septicB) / (total + totalB), septic + septicB, total + totalB)


def get_mean_dataset_values_sepsis(set='A'):
    values = defaultdict(list)
    value_means = defaultdict(list)
    value_stds = defaultdict(list)
    filepath = f"data/sepsis/training_set{set}/"

    with open(f'data/sepsis/train_set{set}.json') as f:
        train_set = json.load(f)
    for i, filename in enumerate(os.listdir(filepath)):
        if filename.endswith(".psv"):
            patient = filename.split("p")[1]  # want only training set mean/var for normalization
            patient = patient.split(".")[0]
            if patient in train_set:
                vals = pd.read_csv(f'{filepath}p{patient}.psv', sep="|").drop(
                    columns=['Hour', 'age', 'gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS', 'SepsisLabel', 'Patient_ID', 'sepsis_bin'])
                is_measured = np.load(f'{filepath}is_measuredp{patient}.npy')

                for idx, col in enumerate(vals.columns):
                    vals[col] = vals[col].mask(is_measured[:, idx] == False, np.nan)
                    values[col].extend(vals[col].dropna())

                # get for age and HospAdmTime
                dems = pd.read_csv(f'{filepath}p{patient}.psv', sep="|")[['age', 'HospAdmTime']]
                values['age'].append(dems['age'][0])
                values['HospAdmTime'].append(dems['HospAdmTime'][0])

    # convert all lists in value_means to overall mean
    for col in values:
        value_means[col] = np.mean(values[col])
        value_stds[col] = np.std(values[col])

    # for age and HospAdmTime
    value_means['age'] = np.mean(values['age'])
    value_stds['age'] = np.std(values['age'])
    value_means['HospAdmTime'] = np.mean(values['HospAdmTime'])
    value_stds['HospAdmTime'] = np.std(values['HospAdmTime'])

    # save means to file
    with open(f'data/sepsis/value_means{set}.json', 'w') as f:
        json.dump(value_means, f)
    with open(f'data/sepsis/value_stds{set}.json', 'w') as f:
        json.dump(value_stds, f)


if __name__ == '__main__':
    calc_sepsis_data_imbalance()
