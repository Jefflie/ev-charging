import csv
from collections import defaultdict
import numpy as np
from scipy.stats import skew, kurtosis

import numpy as np

def numpy_skew(x, axis=0):
    mean = np.mean(x, axis=axis, keepdims=True)
    centered = x - mean
    m2 = np.mean(centered**2, axis=axis)
    m3 = np.mean(centered**3, axis=axis)
    return m3 / (m2 ** 1.5)

def numpy_kurtosis(x, axis=0):
    mean = np.mean(x, axis=axis, keepdims=True)
    centered = x - mean
    m2 = np.mean(centered**2, axis=axis)
    m4 = np.mean(centered**4, axis=axis)
    return (m4 / (m2 ** 2)) - 3

def split_csv_by_columns(csv_file_path, output_prefix):
    with open(csv_file_path, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)

    if len(data) < 2:
        print("Not enough rows to split.")
        return

    rows = data[1:]
    start_idx = 0
    group_id = 0
    current_key = tuple(rows[0][5:10])

    for i in range(1, len(rows)):
        new_key = tuple(rows[i][5:10])
        if new_key != current_key:
            group_rows = rows[start_idx:i]
            output_path = f"{output_prefix}{group_id}.csv"
            with open(output_path, "w", newline='') as f_out:
                csv.writer(f_out).writerows(group_rows)
            print(f"Saved {len(group_rows)} rows to {output_path}")
            group_id += 1
            start_idx = i
            current_key = new_key

    group_rows = rows[start_idx:]
    output_path = f"{output_prefix}{group_id}.csv"
    with open(output_path, "w", newline='') as f_out:
        csv.writer(f_out).writerows(group_rows)
    print(f"Saved {len(group_rows)} rows to {output_path}")


def split_rows_train_test(rows, test_ratio=0.2):
    """
    Splits contiguous rows into train and test by taking the last test_ratio of each group as test.
    """
    groups = defaultdict(list)
    for row in rows:
        groups[row[6]].append(row)  # group by attack type (column 7)

    train_rows, test_rows = [], []
    for attack, group in groups.items():
        n = len(group)
        n_test = max(1, int(test_ratio * n))
        train_rows.extend(group[:-n_test])
        test_rows.extend(group[-n_test:])

    return train_rows, test_rows


def extract_data_sequence_n(rows, save_file_path, num_steps, n):
    """
    Applies sliding-window windowing with size num_steps, and writes num_steps rows to save_file_path.
    """
    parsed = [(row[1:5], row[6]) for row in rows]   # (measurements, attack_type)
    output_rows = []

    i = 0
    while i < len(parsed):
        current_attack = parsed[i][1]
        group = []
        while i < len(parsed) and parsed[i][1] == current_attack:
            group.append(parsed[i])
            i += 1

        if len(group) < num_steps:
            continue

        for j in range(len(group) - num_steps + 1):
            window = group[j:j + num_steps]
            flat = [float(x) for meas, _ in window for x in meas]
            output_rows.append(flat + [n])

    with open(save_file_path, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerows(output_rows)
    print(f"Saved time windows to {save_file_path}")


def extract_data_sequence_features(csv_file_path, save_file_path, num_steps):
    """
    Extracts features (mean, variance, skewness, kurtosis) from time-windowed rows for each measurement (across num_steps), and writes to save_file_path.
    """
    data = []
    with open(csv_file_path, newline='') as f:
        for row in csv.reader(f):
            data.append(row)

    num_cols = len(data[0])
    num_features = num_cols - 1
    if num_features % num_steps != 0:
        raise ValueError(f"{num_features} features not divisible by {num_steps} steps.")

    num_measurements = num_features // num_steps
    features = np.array([[float(x) for x in row[:-1]] for row in data])
    labels = [row[-1] for row in data]
    groups = [[m + s * num_measurements for s in range(num_steps)] for m in range(num_measurements)]
            
    output_rows = []

    for i in range(len(data)):
        row_features = []

        for g in groups:
            vals = features[i, g]
            mean_val = np.mean(vals)
            var_val = np.var(vals, ddof=0)
            # Skewness and kurtosis
            if var_val < 1e-6:  # Avoid division by zero in skewness/kurtosis
                skew_val = 0.0
                kurt_val = 0.0
            else:
                skew_val = skew(vals)
                # skew_val_numpy = numpy_skew(vals)
                kurt_val = kurtosis(vals)
                # kurt_val_numpy = numpy_kurtosis(vals)
                # if abs(skew_val_numpy - skew_val) > 1e-6 or abs(kurt_val_numpy - kurt_val) > 1e-6:
                #     print(f"Different values: {skew_val_numpy - skew_val}, {kurt_val_numpy - kurt_val}")
            row_features.extend([mean_val, var_val, skew_val, kurt_val])

        row_features.append(labels[i])
        output_rows.append(row_features)

    with open(save_file_path, "w", newline='') as f:
        csv.writer(f).writerows(output_rows)
    print(f"Saved features to {save_file_path}")


def merge_and_modify_csvs(input_files, output_file, mapping):
    """
    Merges multiple CSV files into one, changes value in last column based on  mapping dictionary.
    """
    merged_rows = []
    for file in input_files:
        with open(file, newline='') as f:
            for row in csv.reader(f):
                old_val = row[-1]
                if old_val in mapping:
                    row[-1] = mapping[old_val]
                    merged_rows.append(row)
                else:
                    print(f"Unexpected label '{old_val}' in {file}. Row not merged.")
                

    with open(output_file, "w", newline='') as f:
        csv.writer(f).writerows(merged_rows)
    print(f"Merged {len(input_files)} files into {output_file}")


if __name__ == "__main__":
    NUM_STEPS = 20
    TEST_RATIO = 0.2
    SEED = 42

    mapping = {
        "0": "1", "1": "1",
        "2": "0", "3": "2",
        "4": "0", "5": "2",
        "6": "4", "7": "4",
        "8": "3", "9": "3",
    }

    # Step 1: split raw CSV into groups by testing condition
    split_csv_by_columns('EVSE-B-PowerCombined.csv', 'group_')

    # Step 2: for each group, split raw rows into train and test BEFORE windowing, then window each split separately
    for n in range(10):
        with open(f'group_{n}.csv', newline='', encoding='utf-8') as f:
            all_rows = list(csv.reader(f))

        train_rows, test_rows = split_rows_train_test(all_rows, test_ratio=TEST_RATIO)

        # Window each split independently---no leakage across the boundary
        extract_data_sequence_n(train_rows, f'windowed_train_{n}.csv', NUM_STEPS, n)
        extract_data_sequence_n(test_rows,  f'windowed_test_{n}.csv',  NUM_STEPS, n)

        # Feature extraction on each windowed split
        extract_data_sequence_features(f'windowed_train_{n}.csv', f'features_train_{n}.csv', NUM_STEPS)
        extract_data_sequence_features(f'windowed_test_{n}.csv',  f'features_test_{n}.csv',  NUM_STEPS)

    # Step 3: merge all groups, applying label mapping
    merge_and_modify_csvs([f'features_train_{n}.csv' for n in range(10)], 'merged1_4thorder_train.csv', mapping)
    merge_and_modify_csvs([f'features_test_{n}.csv'  for n in range(10)], 'merged1_4thorder_test.csv',  mapping)

    # # For raw-measurement datasets (no feature extraction)
    merge_and_modify_csvs([f'windowed_train_{n}.csv' for n in range(10)], 'merged2_4thorder_train.csv', mapping)
    merge_and_modify_csvs([f'windowed_test_{n}.csv'  for n in range(10)], 'merged2_4thorder_test.csv',  mapping)