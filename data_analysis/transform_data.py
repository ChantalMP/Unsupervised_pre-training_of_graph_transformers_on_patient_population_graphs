import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from data_analysis.mimic_data_analysis import get_original_ages


def transform_tadpole():
    df = pd.read_csv('data/tadpole/raw/tadpole-preprocessed.csv')

    # drop information columns
    for col in [' ', 'PTID', 'VISCODE', 'D1', 'D2', 'COLPROT', 'ORIGPROT', 'SITE', 'EXAMDATE', 'VERSION_BAIPETNMRC_09_12_16',
                'LONIUID_BAIPETNMRC_09_12_16']:
        df = df.drop(col, axis=1)

    # transform real-valued columns for the start set of values

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col])
        elif df[col].str.contains('Pass').any():
            df[col] = df[col].mask(df[col] == 'Fail', 0)
            df[col] = df[col].mask(df[col] == 'Pass', 1)

    # handle string valued columns
    # DX_bl: CN - 1, LMCI - 2, AD - 3
    df['DX_bl'] = df['DX_bl'].mask(df['DX_bl'] == 'CN', 0)
    df['DX_bl'] = df['DX_bl'].mask(df['DX_bl'] == 'SMC', 0)
    df['DX_bl'] = df['DX_bl'].mask(df['DX_bl'] == 'LMCI', 1)
    df['DX_bl'] = df['DX_bl'].mask(df['DX_bl'] == 'EMCI', 1)
    df['DX_bl'] = df['DX_bl'].mask(df['DX_bl'] == 'AD', 2)

    # PTETHCAT: 'Hisp/Latino' - 1, 'Not Hisp/Latino' - 2, 'Unknown' - 3
    df['PTETHCAT'] = df['PTETHCAT'].mask(df['PTETHCAT'] == 'Hisp/Latino', 0)
    df['PTETHCAT'] = df['PTETHCAT'].mask(df['PTETHCAT'] == 'Not Hisp/Latino', 1)
    df['PTETHCAT'] = df['PTETHCAT'].mask(df['PTETHCAT'] == 'Unknown', 2)

    # PTRACCAT: 'Am Indian/Alaskan' - 1, 'Asian' - 2, 'Black' - 3, 'Hawaiian/Other PI' - 4, 'More than one' - 5, 'White' - 6, 'Unknown' - 7
    df['PTRACCAT'] = df['PTRACCAT'].mask(df['PTRACCAT'] == 'White', 0)
    df['PTRACCAT'] = df['PTRACCAT'].mask(df['PTRACCAT'] == 'Am Indian/Alaskan', 1)
    df['PTRACCAT'] = df['PTRACCAT'].mask(df['PTRACCAT'] == 'Asian', 2)
    df['PTRACCAT'] = df['PTRACCAT'].mask(df['PTRACCAT'] == 'Black', 3)
    df['PTRACCAT'] = df['PTRACCAT'].mask(df['PTRACCAT'] == 'Hawaiian/Other PI', 4)
    df['PTRACCAT'] = df['PTRACCAT'].mask(df['PTRACCAT'] == 'More than one', 5)
    df['PTRACCAT'] = df['PTRACCAT'].mask(df['PTRACCAT'] == 'Unknown', 6)

    # PTMARRY: 'Divorced' - 1, 'Married' - 2, 'Never married' - 3, 'Widowed' - 4, 'Unknown' - 5
    df['PTMARRY'] = df['PTMARRY'].mask(df['PTMARRY'] == 'Divorced', 0)
    df['PTMARRY'] = df['PTMARRY'].mask(df['PTMARRY'] == 'Married', 1)
    df['PTMARRY'] = df['PTMARRY'].mask(df['PTMARRY'] == 'Never married', 2)
    df['PTMARRY'] = df['PTMARRY'].mask(df['PTMARRY'] == 'Widowed', 3)
    df['PTMARRY'] = df['PTMARRY'].mask(df['PTMARRY'] == 'Unknown', 4)

    # FLDSTRENG_bl: '1.5 Tesla MRI' - 1, '3 Tesla MRI' - 2
    df['FLDSTRENG_bl'] = df['FLDSTRENG_bl'].mask(df['FLDSTRENG_bl'] == '1.5 Tesla MRI', 0)
    df['FLDSTRENG_bl'] = df['FLDSTRENG_bl'].mask(df['FLDSTRENG_bl'] == '3 Tesla MRI', 1)

    df_copy = df.copy()
    df['PTGENDER'] = df['PTGENDER'].mask(df_copy['PTGENDER'] == 1, 0)
    df['PTGENDER'] = df['PTGENDER'].mask(df_copy['PTGENDER'] == 2, 1)

    # discretize columns with fixed number of continuous values
    for col in ['APOE4', 'CDRSB', 'ADAS11', 'MMSE', 'RAVLT_immediate']:
        values = np.unique(df[col].dropna().values)
        min_val = values.min()
        min_dist = (values[1:] - np.roll(values, 1)[1:]).min()  # compare each elements to neighbours and find minimal distance
        col_copy = df[col].copy()
        for idx, value in enumerate(values):
            label = round((value - min_val) / min_dist)
            df[col] = df[col].mask(col_copy == value, label)

    # add id column
    df.insert(0, 'node_ID', range(0, len(df)))

    df.to_csv('data/tadpole/raw/tadpole_numerical.csv', index=False)


def get_mean_dataset_values():
    value_means = defaultdict(list)
    for folder in tqdm(os.listdir('data/mimic-iii-0/train')):
        # if folder is not a directory , skip it
        if not os.path.isdir('data/mimic-iii-0/train/' + folder) or not folder.startswith('patient'):
            continue
        ts_vals = pd.read_csv('data/mimic-iii-0/train/' + folder + '/' + 'ts_vals.csv')
        ts_is_measured = pd.read_csv('data/mimic-iii-0/train/' + folder + '/' + 'ts_is_measured.csv')

        for col in ts_vals.columns:
            is_measured = ts_is_measured[col.replace("'mean'", "'time_since_measured'")]
            ts_vals[col] = ts_vals[col].mask(is_measured == 0.0, np.nan)
            value_means[col].append(ts_vals[col].dropna().mean())

    # convert all lists in value_means to overall mean
    for col in value_means:
        value_means[col] = np.nanmean(value_means[col])

    # save means to file
    with open('data/mimic-iii-0/value_means.json', 'w') as f:
        json.dump(value_means, f)


def transform_sepsis(set='A'):
    filepath = f"data/sepsis/training_set{set}/"
    for filename in os.listdir(filepath):
        if filename.endswith(".psv"):
            with open(filepath + filename) as openfile:
                patient = filename.split("p")[1]
                patient = patient.split(".")[0]

                file = pd.read_csv(openfile, sep="|")
                file['Patient_ID'] = patient

                file = file.reset_index()
                file = file.rename(columns={"index": "Hour"})

                # rename columns overlapping with MIMIC III to MIMIC names
                name_map = {
                    "HR": "('heart rate', 'mean')",
                    "Temp": "('temperature', 'mean')",
                    "SBP": "('systolic blood pressure', 'mean')",
                    "DBP": "('diastolic blood pressure', 'mean')",
                    "Resp": "('respiratory rate', 'mean')",
                    "EtCO2": "('co2 (etco2, pco2, etc.)', 'mean')",
                    "HCO3": "('bicarbonate', 'mean')",
                    "FiO2": "('fraction inspired oxygen', 'mean')",
                    "pH": "('ph', 'mean')",
                    "PaCO2": "('partial pressure of carbon dioxide', 'mean')",
                    "SaO2": "('oxygen saturation', 'mean')",
                    "BUN": "('blood urea nitrogen', 'mean')",
                    "Calcium": "('calcium', 'mean')",
                    "Chloride": "('chloride', 'mean')",
                    "Creatinine": "('creatinine', 'mean')",
                    "Glucose": "('glucose', 'mean')",
                    "Lactate": "('lactate', 'mean')",
                    "Magnesium": "('magnesium', 'mean')",
                    "Phosphate": "('phosphate', 'mean')",
                    "Potassium": "('potassium', 'mean')",
                    "Hct": "('hematocrit', 'mean')",
                    "Hgb": "('hemoglobin', 'mean')",
                    "PTT": "('partial thromboplastin time', 'mean')",
                    "Platelets": "('platelets', 'mean')",
                    "Age": "age",
                    "Gender": "gender"
                }

                file = file.rename(columns=name_map)

                # generate is_measured mask
                vals = file.drop(columns=['Hour', 'age', 'gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS', 'SepsisLabel', 'Patient_ID'])
                is_measured = ~np.isnan(vals.values)

                # linear interpolation for missing features
                for col in file.columns:
                    file[col] = file[col].interpolate(method='linear', limit_area='inside')  # fill with linear interpolation
                    file[col] = file[col].fillna(method='ffill')  # set values at end to last known value
                    file[col] = file[col].fillna(method='bfill')  # fill values at beginning with first known value

                # generate column with binary sepsis label
                file['sepsis_bin'] = file['SepsisLabel'].values.max()

                # save adapted file
                file.to_csv(f'data/sepsis/training_set{set}/' + filename, sep='|', index=False)
                np.save(f'data/sepsis/training_set{set}/is_measured' + filename.split('.')[0], is_measured)


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


def transform_mimic():
    age_df = get_original_ages()
    value_means = json.load(open('data/mimic-iii-0/value_means.json'))
    # iterate through all folder in data/mimic-iii-0/test, use tqdm to show progress
    for folder in tqdm(os.listdir('data/mimic-iii-0/test')):
        # if folder is not a directory , skip it
        if not os.path.isdir('data/mimic-iii-0/test/' + folder) or not folder.startswith('patient'):
            continue
        # extract patient id from folder name
        patient_id = folder.split('_')[1]
        # extract data and static prediction task labels, for now ignore rolling tasks

        treatments = pd.read_csv('data/mimic-iii-0/test/' + folder + '/' + 'ts_treatment.csv')
        if 'DNR Ordered_1' in treatments.columns:
            treatments = treatments.drop(['DNR Ordered_1', 'Comfort Measures Ordered_1'], axis=1)
            treatments.to_csv('data/mimic-iii-0/test/' + folder + '/' + 'ts_treatment.csv', index=False)

        statics = pd.read_csv('data/mimic-iii-0/test/' + folder + '/' + 'statics.csv')
        ts_vals = pd.read_csv('data/mimic-iii-0/test/' + folder + '/' + 'ts_vals.csv')
        ts_is_measured = pd.read_csv('data/mimic-iii-0/test/' + folder + '/' + 'ts_is_measured.csv')

        # for all columns in ts_vals , replace 'nan' with last measured value or zero if none was measured
        for col in ts_vals.columns:
            # ts_vals[col] = ts_vals[col].fillna(method='ffill') # fill forward
            is_measured = ts_is_measured[col.replace("'mean'", "'time_since_measured'")]
            ts_vals[col] = ts_vals[col].mask(is_measured == 0.0, np.nan)
            # if column has only nan values, replace with mean value
            if ts_vals[col].isnull().all():
                ts_vals[col] = value_means[col]
            ts_vals[col] = ts_vals[col].interpolate(method='linear', limit_area='inside')  # fill with linear interpolation
            ts_vals[col] = ts_vals[col].fillna(method='ffill')  # set values at end to last known value
            ts_vals[col] = ts_vals[col].fillna(method='bfill')  # fill with first known value whatever is left in beginning of seqs
        ts_vals.to_csv('data/mimic-iii-0/test/' + folder + '/' + 'ts_vals_linear_imputed.csv', index=False)

        # get age from row where patient id is equal to current patient id
        real_age = age_df[age_df['ICUSTAY_ID'] == int(patient_id)]['age'].values[0]
        # create pandas dataframe with columns 'gender', 'ethnicity', 'insurance', 'admission_type', 'first_careunit'
        new_statics = pd.DataFrame(columns=['age', 'gender', 'ethnicity', 'insurance', 'admission_type', 'first_careunit'])
        value_row = defaultdict(int)
        value_row['age'] = real_age
        for i in range(1, 3):
            if statics[f'gender_{i}'].values[0] == 1:
                # add gender column to new_statics df and add value 1
                value_row['gender'] = i - 1  # 1 to 0, 2 to 1

        for i in range(5):
            if statics[f'ethnicity_{i}'].values[0] == 1:
                value_row['ethnicity'] = i

        for i in range(1, 6):
            if statics[f'insurance_{i}'].values[0] == 1:
                value_row['insurance'] = i - 1

        # same for admission_type 1-3
        for i in range(1, 4):
            if statics[f'admission_type_{i}'].values[0] == 1:
                value_row['admission_type'] = i - 1

        # same for first_careunit 1-5
        for i in range(1, 6):
            if statics[f'first_careunit_{i}'].values[0] == 1:
                value_row['first_careunit'] = i - 1

        new_statics = new_statics.append(value_row, ignore_index=True)
        # save to statics.csv
        new_statics.to_csv('data/mimic-iii-0/test/' + folder + '/' + 'statics.csv', index=False)


if __name__ == '__main__':
    # get_mean_dataset_values()
    # transform_mimic()
    # transform_sepsis(set='B')
    # get_mean_dataset_values_sepsis(set='B')
    pass
