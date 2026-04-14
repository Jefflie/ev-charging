import csv
from collections import defaultdict
import numpy as np
import time

def split_csv_by_columns(csv_file_path, output_prefix):
    """
    Splits a CSV into multiple CSV files based on changes in columns 6-10 (1-based indexing).
    Starting from the second row, rows are grouped together until any value in columns 6-10 changes.
    """

    with open(csv_file_path, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)

    if len(data) < 2:
        print("Not enough rows to split.")
        return

    rows = data[1:]  # start from second row (skip header)

    # Group rows based on columns 6-10
    start_idx = 0
    group_id = 0
    current_key = tuple(rows[0][5:10])  # columns 6-10 are indices 5-9 (0-based)

    for i in range(1, len(rows)):
        new_key = tuple(rows[i][5:10])
        if new_key != current_key:
            # Save the current group
            group_rows = rows[start_idx:i]
            output_path = f"{output_prefix}{group_id}.csv"
            with open(output_path, "w", newline='') as f_out:
                writer = csv.writer(f_out)
                writer.writerows(group_rows)
            print(f"Saved {len(group_rows)} rows to {output_path}")
            
            # Start a new group
            group_id += 1
            start_idx = i
            current_key = new_key

    # Save the final group
    group_rows = rows[start_idx:]
    output_path = f"{output_prefix}{group_id}.csv"

    with open(output_path, "w", newline='') as f_out:
        writer = csv.writer(f_out)
        writer.writerows(group_rows)

    print(f"Saved {len(group_rows)} rows to {output_path}")

def extract_data_sequence_n(csv_file_path, save_file_path, num_steps, n):
    """
    """

    # Read all rows into memory
    rows = []
    with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            measurements = row[1:5]   # columns 2–5
            attack_type = row[6]      # column 7
            rows.append((measurements, attack_type))

    output_rows = []

    # Process rows by attack group
    i = 0
    while i < len(rows):
        current_attack = rows[i][1]
        group = []

        # Collect consecutive rows with the same attack type
        while i < len(rows) and rows[i][1] == current_attack:
            group.append(rows[i])
            i += 1

        k = len(group)
        if k < num_steps:
            continue  # not enough timesteps for a sequence

        # Generate (k-4) sliding windows within this group
        for j in range(k - num_steps + 1):
            window = group[j:j+num_steps]

            # Flatten measurements (20 features)
            measurements_flat = []
            for meas, _ in window:
                measurements_flat.extend([float(x) for x in meas])
            label = n
            output_rows.append(measurements_flat + [label])

    with open(save_file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(output_rows)

    print(f"Saved time windows to {save_file_path}")

def extract_data_sequence_features(csv_file_path, save_file_path, num_steps):
    """
    Reads a CSV with N rows and M+1 columns (M features + 1 label), where M = num_steps * num_measurements.
    For each measurement type, computes summary features (mean, variance, range) across the num_steps time steps.

    Output CSV has shape N * (3*num_measurements + 1).
    """

    data = []
    with open(csv_file_path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)

    num_rows = len(data)
    num_cols = len(data[0])

    # ensure we have at least one label
    if num_cols < 2:
        raise ValueError("Input file must have at least one feature column and one label column.")

    num_features = num_cols - 1  # all except label

    # check divisibility
    if num_features % num_steps != 0:
        raise ValueError(
            f"Number of feature columns ({num_features}) is not divisible by num_steps ({num_steps})."
        )

    num_measurements = num_features // num_steps

    # convert to numeric arrays
    features = np.array([list(map(float, row[:-1])) for row in data])
    labels = [row[-1] for row in data]

    # Get column indices by measurement type
    # e.g., num_steps=5, num_measurements=4
    # groups = [[0,4,8,12,16], [1,5,9,13,17], [2,6,10,14,18], [3,7,11,15,19]]
    col_idx = [[m + s * num_measurements for s in range(num_steps)] for m in range(num_measurements)]

    # Compute features
    output_rows = []
    # start = time.time()
    for i in range(num_rows):
        row_features = []
        for c in col_idx:
            vals = features[i, c]

            mean_val = np.mean(vals)
            var_val = np.var(vals, ddof=0)
            range_val = np.max(vals) - np.min(vals)

            row_features.extend([mean_val, var_val, range_val])
        row_features.append(labels[i])
        output_rows.append(row_features)
    # end = time.time()
    # print((end - start)/num_rows)

    # Write to output csv
    with open(save_file_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(output_rows)

    print(f"Saved features to {save_file_path}")

def merge_and_modify_csv(input_files, output_file, mapping):
    """
    Merges multiple CSV files into one by stacking them top-to-bottom.
    Before merging, replaces values in the last column according to a label mapping.
    """
    
    merged_rows = []

    for file in input_files:
        with open(file, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                old_val = row[-1]
                if old_val in mapping:
                    row[-1] = mapping[old_val]
                else:
                    print("error in row")
                merged_rows.append(row)

    # Write to output csv
    with open(output_file, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(merged_rows)

    print(f"Merged {len(input_files)} csv files into {output_file} with label mapping applied")

def stratified_split_csv(file_path, test_ratio=0.2):
    """
    Splits a CSV dataset into train and test sets, preserving class proportions.
    The last column is assumed to be the class label.
    """

    # Read entire CSV
    with open(file_path, 'r', newline='') as f:
        reader = csv.reader(f)
        data = [row for row in reader]

    # Group rows by class (last column)
    class_groups = defaultdict(list)
    for row in data:
        label = row[-1]
        class_groups[label].append(row)

    train_rows, test_rows = [], []

    # Stratified split per class
    for label, rows in class_groups.items():
        rows = np.array(rows)
        n_total = len(rows)
        n_test = max(1, int(test_ratio * n_total))  # at least one sample if possible
        idx = np.random.permutation(n_total)
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]
        test_rows.extend(rows[test_idx])
        train_rows.extend(rows[train_idx])

    # Save the new CSVs
    train_path = file_path.replace('.csv', '_train.csv')
    test_path = file_path.replace('.csv', '_test.csv')

    with open(train_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(train_rows)

    with open(test_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(test_rows)

    print(f"Saved training set in {train_path}")
    print(f"Saved test set in {test_path}")

    return train_path, test_path

if __name__ == "__main__":
    """
    The pipeline from EVSE-B-PowerCombined.csv to merged1_train.csv, merged1_test.csv is:
    grouping by n, 20-step windowing, feature extraction, merging with label mapping, and finally stratified splitting

    To get the raw time series measurements (merged2_train.csv and merged2_test.csv), skip feature extraction step and merge right after windowing. 
    """

    # Separate original dataset into different groups whenever the testing condition changes
    split_csv_by_columns('EVSE-B-PowerCombined.csv', 'group_')

    # Perform windowing and feature extraction on each group individually
    # This avoids having our time window span discontinuous measurement times 
    for n in range(10):
        extract_data_sequence_n(csv_file_path=f'group_{n}.csv', save_file_path=f'dataset_5class_timeseries_20_{n}.csv', num_steps=20, n=n)
        extract_data_sequence_features(csv_file_path=f'dataset_5class_timeseries_20_{n}.csv', save_file_path=f'dataset_5class_timeseries_20_features_{n}.csv', num_steps=20)

    # Labeling for each group to desired cyberattack class: {"none": 0, "DoS": 1, "cryptojacking": 2, "backdoor": 3, "recon": 4}
    mapping = {"0": "1",
               "1": "1",
               "2": "0",
               "3": "0", 
                "4": "2",
                "5": "2", 
                "6": "4", 
                "7": "4", 
                "8": "3", 
                "9": "3"}
    
    # Merge each group using the correct label 
    merge_and_modify_csv([f'dataset_5class_timeseries_20_features_{n}.csv' for n in range(10)], "merged1.csv", mapping)

    # 80-20 train-test split of time series features in merged1_train.csv and merged1_test.csv 
    stratified_split_csv("merged1.csv", test_ratio=0.2)

    # To get dataset with raw measurements, merge pre-feature extraction
    merge_and_modify_csv([f'dataset_5class_timeseries_20_{n}.csv' for n in range(10)], "merged2.csv", mapping)
    stratified_split_csv("merged2.csv", test_ratio=0.2)
