import numpy as np
import torch


def create_treat_mask_mimic(item, mask_ratio=0.15, block_size=6):
    num_mask_columns = round(14 * mask_ratio)
    binary_treatment_mask = np.zeros_like(item.treatments, dtype=np.bool)
    random_starts = np.random.randint(0, 25 - block_size, size=item.treatments.shape[0] * num_mask_columns)
    for p_i in range(item.treatments.shape[0]):
        starts = random_starts[num_mask_columns * p_i:num_mask_columns * p_i + num_mask_columns]
        col_indices = np.random.choice(14, size=num_mask_columns, replace=False)  # do not mask DNR and CMO
        for s, c_i in zip(starts, col_indices):
            binary_treatment_mask[p_i, s:s + block_size, c_i] = True
    return binary_treatment_mask


def create_patient_mask_mimic(item, mask_ratio=0.15):
    binary_treatment_mask = np.zeros_like(item.treatments, dtype=np.bool)
    drop_idxs = np.random.choice(len(binary_treatment_mask), size=int(len(binary_treatment_mask) * mask_ratio), replace=False)
    binary_treatment_mask[drop_idxs, :, :16] = True

    valid_mask_vals = item.is_measured.bool()  # torch.ones_like(item.is_measured) # which values were actually measured and not interpolated, for test all values are valid (also predict interpolated values)
    binary_vals_mask = np.zeros_like(item.is_measured, dtype=np.bool)
    binary_vals_mask_big = np.zeros_like(item.vals, dtype=np.bool)
    binary_mask_mask_big = np.zeros_like(item.vals, dtype=np.bool)

    binary_vals_mask[drop_idxs] = True
    binary_vals_mask_big[drop_idxs, :, 0::2] = True
    binary_mask_mask_big[drop_idxs, :, 1::2] = True

    final_vals_mask = torch.logical_and(valid_mask_vals, torch.from_numpy(binary_vals_mask))

    return binary_treatment_mask, binary_vals_mask_big, binary_mask_mask_big, final_vals_mask


def create_vals_mask_mimic(item, mask_ratio=0.15, block_size=6):
    num_mask_columns = round(56 * mask_ratio)
    valid_mask_vals = item.is_measured.bool()  # torch.ones_like(item.is_measured) # which values were actually measured and not interpolated, for test all values are valid (also predict interpolated values)
    binary_vals_mask = np.zeros_like(item.is_measured, dtype=np.bool)
    binary_vals_mask_big = np.zeros_like(item.vals, dtype=np.bool)
    binary_mask_mask_big = np.zeros_like(item.vals, dtype=np.bool)
    random_starts = np.random.randint(0, 25 - block_size, size=item.vals.shape[0] * num_mask_columns)
    for p_i in range(item.vals.shape[0]):
        starts = random_starts[num_mask_columns * p_i:num_mask_columns * p_i + num_mask_columns]
        col_indices = np.random.choice(56, size=num_mask_columns, replace=False)
        for s, c_i in zip(starts, col_indices):
            binary_vals_mask[p_i, s:s + block_size, c_i] = True
            binary_vals_mask_big[p_i, s:s + block_size, c_i * 2] = True
            binary_mask_mask_big[p_i, s:s + block_size, c_i * 2 + 1] = True

    final_vals_mask = torch.logical_and(valid_mask_vals, torch.from_numpy(binary_vals_mask))

    return binary_vals_mask_big, binary_mask_mask_big, final_vals_mask
