#!/usr/bin/env python3
"""
generate_qml_dataset.py
-------------------------
This script generates a larger dataset for QML classification by randomly
sampling from two source .npy files.

It creates a specified number of feature vectors for each class, where each
vector is composed of a small, random sample of values from the source data.
The final, combined dataset is saved as a CSV file.

Usage:
    python generate_qml_dataset.py \
        --low_file soillow_scaled.npy \
        --high_file soilhigh_scaled.npy \
        --output_file qml_dataset.csv \
        --feature_size 8 \
        --num_samples 1000
"""

import argparse
import numpy as np
import pandas as pd

def create_dataset_from_files(low_file, high_file, output_file, num_samples, feature_size):
    """
    Creates and saves a dataset by randomly sampling from source .npy files.

    Args:
        low_file (str): Path to the .npy file for the "low" class (label 0).
        high_file (str): Path to the .npy file for the "high" class (label 1).
        output_file (str): Path to save the final .csv file.
        num_samples (int): The number of data points to generate for each class.
        feature_size (int): The number of features (CE values) per data point.
    """
    try:
        soillow_data = np.load(low_file)
        soilhigh_data = np.load(high_file)
        print(f"Successfully loaded source files: '{low_file}' and '{high_file}'.")
    except FileNotFoundError as e:
        print(f"FATAL ERROR: Could not load the source .npy files. {e}")
        print("Please ensure the file paths are correct.")
        return

    def _create_features(data_array, label):
        """Helper to generate features for one class."""
        feature_list = []
        for _ in range(num_samples):
            indices = np.random.choice(len(data_array), size=feature_size, replace=True)
            feature_list.append(data_array[indices])
        df = pd.DataFrame(feature_list)
        df.columns = [f'feature_{i}' for i in range(feature_size)]
        df['label'] = label
        return df

    print(f"Generating {num_samples} samples for the 'low' class (label 0)...")
    df_low = _create_features(soillow_data, 0)
    
    print(f"Generating {num_samples} samples for the 'high' class (label 1)...")
    df_high = _create_features(soilhigh_data, 1)

    print("Combining and shuffling data...")
    final_df = pd.concat([df_low, df_high]).sample(frac=1).reset_index(drop=True)
    
    try:
        final_df.to_csv(output_file, index=False)
        print(f"\n--- SUCCESS ---")
        print(f"Successfully generated '{output_file}' with {len(final_df)} total samples.")
        print(f"Each sample has {feature_size} features.")
    except IOError as e:
        print(f"\n--- FAILURE ---")
        print(f"Could not write to output file '{output_file}'. Error: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a dataset for QML classification.")
    parser.add_argument("--low_file", type=str, required=True, help="Path to the 'low' class .npy file.")
    parser.add_argument("--high_file", type=str, required=True, help="Path to the 'high' class .npy file.")
    parser.add_argument("--output_file", type=str, default="qml_dataset.csv", help="Name of the output CSV file.")
    parser.add_argument("--feature_size", type=int, default=9, help="Number of features per sample (and qubits).")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to generate per class.")
    
    args = parser.parse_args()
    
    create_dataset_from_files(
        low_file=args.low_file,
        high_file=args.high_file,
        output_file=args.output_file,
        num_samples=args.num_samples,
        feature_size=args.feature_size
    )