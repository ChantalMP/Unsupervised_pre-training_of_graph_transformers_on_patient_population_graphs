import os

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


if __name__ == '__main__':
    # get_max_stay_time()
    # get_stay_time_distribution()
    calc_sepsis_data_imbalance()
