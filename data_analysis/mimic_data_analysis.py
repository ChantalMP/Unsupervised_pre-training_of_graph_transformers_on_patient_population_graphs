import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def get_original_ages():
    # get all patient ids
    # in patients.csv Date of birth (DOB) is saved together with Subject ID
    # in admissions admission time is saved together with Subject ID and HADM_ID
    # use HADM_ID to get all used patient, then get their admission time and use Subject ID to get DOB

    # read admissions.csv
    admissions = pd.read_csv('data/mimic-iii-0/ADMISSIONS.csv')
    # read patients.csv
    patients = pd.read_csv('data/mimic-iii-0/PATIENTS.csv')
    # red all_patients.json ids: ['subject_id', 'hadm_id', 'icustay_id', 'Fold']
    all_patients = pd.read_json('data/mimic-iii-0/all_patients.json')
    hadm_and_icu_ids = all_patients[[1, 2]]
    # rename column 1 to HADM_ID and column 2 to ICUSTAY_ID
    hadm_and_icu_ids.columns = ['HADM_ID', 'ICUSTAY_ID']
    # from admissions get all rows where HADM_ID is in hadm_ids
    admissions = admissions[admissions['HADM_ID'].isin(hadm_and_icu_ids['HADM_ID'])]
    # append column 2 from hadm_and_icu_ids to admissions, merge based on HADM_ID
    admissions = admissions.merge(hadm_and_icu_ids, on='HADM_ID')
    admittime = admissions[['HADM_ID', 'ICUSTAY_ID', 'SUBJECT_ID', 'ADMITTIME']]
    # from patients get all rows with subject_id in dob['SUBJECT_ID']
    patients = patients[patients['SUBJECT_ID'].isin(admittime['SUBJECT_ID'])]
    # append DOB column from patients to admittime
    admittime = admittime.merge(patients[['SUBJECT_ID', 'DOB']], on='SUBJECT_ID')
    # for all values in column ADMITTIME split on '-' and take the first element and convert to int
    admittime['ADMITTIME'] = admittime['ADMITTIME'].apply(lambda x: int(x.split('-')[0]))
    # same for DOB
    admittime['DOB'] = admittime['DOB'].apply(lambda x: int(x.split('-')[0]))

    # add column 'age' to admittime which is the difference between ADMITTIME and DOB
    admittime['age'] = admittime['ADMITTIME'] - admittime['DOB']
    # set all ages which have value 300 to 91.4
    admittime.loc[admittime['age'] >= 300, 'age'] = 91.4
    return admittime
    # print minimum, maximum and unique values with counts in age column
    # print(admittime['age'].min(), admittime['age'].max(), admittime['age'].unique(), admittime['age'].value_counts())
    # get mean and std of age
    # print(admittime['age'].mean(), admittime['age'].std())


def calc_treat_imbalance():
    zero_count = 0
    one_count = 0
    for idx, patient in tqdm(enumerate(os.listdir('data/mimic-iii-0/train'))):
        if 'patient' not in patient:  # skip non-patient files
            continue
        ts_treatment = pd.read_csv('data/mimic-iii-0/train/' + patient + '/' + 'ts_treatment.csv')[:24]
        one_count += np.count_nonzero(ts_treatment.values)
        zero_count += np.count_nonzero(ts_treatment.values == 0)

    print(zero_count, one_count)


def calc_treat_imbalance_binary():
    zero_count = 0
    one_count = 0
    per_feature_zero_count = defaultdict(int)
    per_feature_one_count = defaultdict(int)
    for idx, patient in tqdm(enumerate(os.listdir('data/mimic-iii-0/train'))):
        if 'patient' not in patient:  # skip non-patient files
            continue
        ts_treatment = pd.read_csv('data/mimic-iii-0/train/' + patient + '/' + 'ts_treatment.csv')
        ts_treatment_binary = torch.tensor(np.any(ts_treatment.values, axis=0).astype(np.float32))
        one_count += np.count_nonzero(ts_treatment_binary)
        zero_count += np.count_nonzero(ts_treatment_binary == 0)
        for i in range(14):  # 16
            per_feature_zero_count[i] += np.count_nonzero(ts_treatment_binary[i] == 0)
            per_feature_one_count[i] += np.count_nonzero(ts_treatment_binary[i] == 1)

    print(zero_count, one_count)
    print(per_feature_zero_count)
    print(per_feature_one_count)


def calc_acu_imbalance():
    counts = defaultdict(int)
    labels = []
    for idx, patient in tqdm(enumerate(os.listdir('data/mimic-iii-0/train'))):
        if 'patient' not in patient:  # skip non-patient files
            continue
        acu_label = pd.read_csv('data/mimic-iii-0/train/' + patient + '/' + 'Final Acuity Outcome.csv')['Final Acuity Outcome'][0]
        counts[acu_label] += 1
        labels.append(acu_label)

    for idx, patient in tqdm(enumerate(os.listdir('data/mimic-iii-0/val'))):
        if 'patient' not in patient:  # skip non-patient files
            continue
        acu_label = pd.read_csv('data/mimic-iii-0/val/' + patient + '/' + 'Final Acuity Outcome.csv')['Final Acuity Outcome'][0]
        counts[acu_label] += 1
        labels.append(acu_label)

    for idx, patient in tqdm(enumerate(os.listdir('data/mimic-iii-0/test'))):
        if 'patient' not in patient:  # skip non-patient files
            continue
        acu_label = pd.read_csv('data/mimic-iii-0/test/' + patient + '/' + 'Final Acuity Outcome.csv')['Final Acuity Outcome'][0]
        counts[acu_label] += 1
        labels.append(acu_label)

    print(counts)
    # calculate and print occurance in percentage of each label
    sample_num = sum(counts.values())
    for label in range(18):
        count = counts[label]
        ratio = count / sample_num
        print(f'{label}: {ratio}')

    # print(labels)
    # class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=np.array(labels))
    class_weights_lin = [1 / count for count in np.unique(labels, return_counts=True)[1]]
    class_weights_log = [np.abs(1.0 / (np.log(count) + 1)) for count in np.unique(labels, return_counts=True)[1]]


def calc_icd_imbalance():  # 12411 -> 70%
    zeros = defaultdict(int)
    ones = defaultdict(int)
    for idx, patient in tqdm(enumerate(os.listdir('data/mimic-iii-0/train'))):
        if 'patient' not in patient:  # skip non-patient files
            continue
        icds = pd.read_csv('data/mimic-iii-0/train/' + patient + '/' + 'static_tasks_binary_multilabel.csv').values[0, 1:-1]
        if np.isnan(icds).any():
            if np.isnan(icds).sum() != 18:
                print("WARNING: NOT ALL ICD CODES ARE NONE:", np.isnan(icds).sum())
            continue
        if icds.sum() == 0:
            icds[-1] = 1
        for label in range(18):
            if icds[label] == 0:
                zeros[label] += 1
            elif icds[label] == 1:
                ones[label] += 1

    print(zeros)
    print(ones)


def calc_rea_imbalance():
    zero_count = 0
    one_count = 0
    for idx, patient in tqdm(enumerate(os.listdir('data/mimic-iii-0/train'))):
        if 'patient' not in patient:  # skip non-patient files
            continue
        rea_label = pd.read_csv('data/mimic-iii-0/train/' + patient + '/' + 'static_tasks_binary_multilabel.csv')['Readmission 30'][0]
        if rea_label == 0:
            zero_count += 1
        else:
            one_count += 1

    print("train: ")
    print(zero_count, one_count)
    print(zero_count / (zero_count + one_count), one_count / (zero_count + one_count))

    zero_count = 0
    one_count = 0
    for idx, patient in tqdm(enumerate(os.listdir('data/mimic-iii-0/val'))):
        if 'patient' not in patient:  # skip non-patient files
            continue
        rea_label = pd.read_csv('data/mimic-iii-0/val/' + patient + '/' + 'static_tasks_binary_multilabel.csv')['Readmission 30'][0]
        if rea_label == 0:
            zero_count += 1
        else:
            one_count += 1

    print("val: ")
    print(zero_count, one_count)
    print(zero_count / (zero_count + one_count), one_count / (zero_count + one_count))

    zero_count = 0
    one_count = 0
    for idx, patient in tqdm(enumerate(os.listdir('data/mimic-iii-0/test'))):
        if 'patient' not in patient:  # skip non-patient files
            continue
        rea_label = pd.read_csv('data/mimic-iii-0/test/' + patient + '/' + 'static_tasks_binary_multilabel.csv')['Readmission 30'][0]
        if rea_label == 0:
            zero_count += 1
        else:
            one_count += 1

    print("test: ")
    print(zero_count, one_count)
    print(zero_count / (zero_count + one_count), one_count / (zero_count + one_count))


def calc_los_imbalance():
    zero_count = 0
    one_count = 0
    for idx, patient in tqdm(enumerate(os.listdir('data/mimic-iii-0/train'))):
        if 'patient' not in patient:  # skip non-patient files
            continue
        los_label = pd.read_csv('data/mimic-iii-0/train/' + patient + '/' + 'static_tasks_binary_multilabel.csv')['Long LOS'][0]
        if los_label == 0:
            zero_count += 1
        else:
            one_count += 1

    for idx, patient in tqdm(enumerate(os.listdir('data/mimic-iii-0/val'))):
        if 'patient' not in patient:  # skip non-patient files
            continue
        los_label = pd.read_csv('data/mimic-iii-0/val/' + patient + '/' + 'static_tasks_binary_multilabel.csv')['Long LOS'][0]
        if los_label == 0:
            zero_count += 1
        else:
            one_count += 1

    for idx, patient in tqdm(enumerate(os.listdir('data/mimic-iii-0/test'))):
        if 'patient' not in patient:  # skip non-patient files
            continue
        los_label = pd.read_csv('data/mimic-iii-0/test/' + patient + '/' + 'static_tasks_binary_multilabel.csv')['Long LOS'][0]
        if los_label == 0:
            zero_count += 1
        else:
            one_count += 1

    print(zero_count, one_count)
    print(zero_count / (zero_count + one_count), one_count / (zero_count + one_count))


if __name__ == '__main__':
    pass
