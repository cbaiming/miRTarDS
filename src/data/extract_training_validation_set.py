import pandas as pd
import numpy as np
import os

# ============================================================================
# Configuration Parameters
# ============================================================================
RANDOM_SEED = 42              # Random seed for reproducibility
VALIDATION_SPLIT_RATIO = 0.1  # Proportion of data for validation set
INPUT_FILE_PATH = 'MTI/concat_MTIs_with_counts.txt'
OUTPUT_FILE_PATH = 'MTI/miRTarDS_train_valid.csv'
# ============================================================================

# Set random seeds for reproducibility
np.random.seed(RANDOM_SEED)

# Read the dataset from CSV file
df = pd.read_csv(INPUT_FILE_PATH)

# Create binary labels: 1 for experimental validation (TarBase/miRTarBase), 0 for predicted only
# Label = 1 if support type contains either 'TarBase' or 'miRTarBase', otherwise 0
df['label'] = (df['support type'].str.contains('TarBase') | 
               df['support type'].str.contains('miRTarBase')).astype(int)

# Separate positive (experimentally validated) and negative (predicted only) samples
experiment_rows = df[df['label'] == 1]  # Positive samples
predicted_rows = df[df['label'] == 0]   # Negative samples

# Balance the dataset by downsampling negative samples to match positive samples count
# This ensures equal representation of both classes in training
sampled_predicted_rows = predicted_rows.sample(
    n=len(experiment_rows), 
    random_state=RANDOM_SEED
)

# Combine positive and downsampled negative samples
combined_df = pd.concat([experiment_rows, sampled_predicted_rows], ignore_index=True)

# Shuffle the combined dataset to randomize sample order
shuffled_df = combined_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

# Separate shuffled dataset into positive and negative samples for stratified split
positive_samples = shuffled_df[shuffled_df['label'] == 1]
negative_samples = shuffled_df[shuffled_df['label'] == 0]

# Calculate validation set size (10% of total samples)
valid_size = int(VALIDATION_SPLIT_RATIO * len(shuffled_df))

# Stratified split: Maintain class balance in validation set
# Ensure equal number of positive and negative samples in validation set
valid_positive_size = valid_size // 2
valid_negative_size = valid_size - valid_positive_size

# Randomly select samples for validation set
valid_positive = positive_samples.sample(n=valid_positive_size, random_state=RANDOM_SEED)
valid_negative = negative_samples.sample(n=valid_negative_size, random_state=RANDOM_SEED)
valid_df = pd.concat([valid_positive, valid_negative], ignore_index=True)

# Remaining samples form the training set
train_positive = positive_samples.drop(valid_positive.index)
train_negative = negative_samples.drop(valid_negative.index)
train_df = pd.concat([train_positive, train_negative], ignore_index=True)

# Add split identifiers for each sample
train_df['split'] = 'train'
valid_df['split'] = 'valid'

# Combine training and validation sets into final dataset
final_df = pd.concat([train_df, valid_df], ignore_index=True)

# Display dataset statistics for verification
print(f"Total instances: {len(shuffled_df)}")
print(f"Training instances: {len(train_df)}")
print(f"Validation instances: {len(valid_df)}")
print(f"Positive training instances: {len(train_df[train_df['label'] == 1])}")
print(f"Negative training instances: {len(train_df[train_df['label'] == 0])}")
print(f"Positive validation instances: {len(valid_df[valid_df['label'] == 1])}")
print(f"Negative validation instances: {len(valid_df[valid_df['label'] == 0])}")

# Save the processed dataset to CSV
final_df.to_csv(OUTPUT_FILE_PATH, index=False)

print(f"\nDataset successfully processed and saved to: {OUTPUT_FILE_PATH}")
print(f"Random seed: {RANDOM_SEED}, Validation split: {VALIDATION_SPLIT_RATIO}")