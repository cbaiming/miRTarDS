"""
Script: KNN-based Sample Matching for Imbalanced Dataset Construction
Purpose: Construct balanced dataset for miRNA-target interaction classification
         by matching experimental samples (positive class) with predicted samples 
         (negative class) using K-Nearest Neighbors (KNN) algorithm.
         
Input: Tab-separated file containing miRNA-target interaction data with columns:
       - Required columns: 'support type', 'miR_disease_num', 'gene_disease_num'
       - File path: 'MTI/concat_MTIs_with_counts.txt'
       
Output:
      1. Balanced dataset: 'MTI/miRTarDS_KNN_matching.csv'
      2. Visualization: 'MTI/Disease Entry Distribution KNN.png'
      
Methodology:
      - Label experimental evidence (TarBase/miRTarBase) as positive samples (1)
      - Label predicted interactions as negative samples (0)
      - Use KNN to find most similar predicted samples for each experimental sample
      - Combine matched pairs and split into train/validation sets
      - Ensure class balance in both training and validation sets
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List, Dict, Any

# ============================
# CONFIGURATION PARAMETERS
# ============================

# File paths
INPUT_FILE_PATH: str = 'MTI/concat_MTIs_with_counts.txt'
OUTPUT_FILE_PATH: str = 'MTI/miRTarDS_train_valid_KNN_extract.csv'
VISUALIZATION_PATH: str = 'MTI/Disease Entry Distribution KNN.png'

# Data labeling parameters
POSITIVE_KEYWORDS: List[str] = ['TarBase', 'miRTarBase']
POSITIVE_LABEL: int = 1
NEGATIVE_LABEL: int = 0

# Feature configuration
FEATURE_COLUMNS: List[str] = ['miR_disease_num', 'gene_disease_num']

# KNN matching parameters
KNN_N_NEIGHBORS: int = 1
KNN_ALGORITHM: str = 'ball_tree'
RANDOM_STATE: int = 42

# Data split parameters
VALIDATION_SIZE: float = 0.1
TEST_SIZE: float = 0.0  # No test split in current implementation
STRATIFY_SPLIT: bool = True

# Visualization parameters
PLOT_FIGSIZE: Tuple[int, int] = (10, 5)
POSITIVE_MARKER: str = 'x'
NEGATIVE_MARKER: str = 'o'
POSITIVE_COLOR: str = 'red'
NEGATIVE_COLOR: str = 'blue'
PLOT_ALPHA: float = 0.5


def load_and_label_data(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load data from file and label samples as positive or negative.
    
    Args:
        file_path: Path to input data file
        
    Returns:
        Tuple containing:
            - experiment_df: DataFrame with positive samples
            - predicted_df: DataFrame with negative samples
            
    Raises:
        FileNotFoundError: If input file doesn't exist
        KeyError: If required columns are missing
    """
    # Load data
    df = pd.read_csv(file_path, sep='\t')
    
    # Validate required columns exist
    required_columns = ['support type'] + FEATURE_COLUMNS
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")
    
    # Create labels based on support type
    label_condition = df['support type'].str.contains('|'.join(POSITIVE_KEYWORDS))
    df['label'] = np.where(label_condition, POSITIVE_LABEL, NEGATIVE_LABEL)
    
    # Split into positive and negative sets
    experiment_df = df[df['label'] == POSITIVE_LABEL].copy()
    predicted_df = df[df['label'] == NEGATIVE_LABEL].copy()
    
    print(f"Total samples: {len(df)}")
    print(f"Positive samples: {len(experiment_df)}")
    print(f"Negative samples: {len(predicted_df)}")
    print(f"Class ratio: {len(predicted_df)/len(experiment_df):.2f}:1")
    
    return experiment_df, predicted_df


def standardize_features(
    positive_df: pd.DataFrame, 
    negative_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Standardize features for distance-based matching.
    
    Args:
        positive_df: DataFrame with positive samples
        negative_df: DataFrame with negative samples
        
    Returns:
        Tuple containing:
            - pos_features_scaled: Standardized positive features
            - neg_features_scaled: Standardized negative features  
            - scaler: Fitted StandardScaler object
    """
    scaler = StandardScaler()
    
    # Fit scaler on negative samples (larger set)
    scaler.fit(negative_df[FEATURE_COLUMNS])
    
    # Transform both positive and negative features
    pos_features_scaled = scaler.transform(positive_df[FEATURE_COLUMNS])
    neg_features_scaled = scaler.transform(negative_df[FEATURE_COLUMNS])
    
    return pos_features_scaled, neg_features_scaled, scaler


def knn_match_samples(
    positive_features: np.ndarray,
    negative_features: np.ndarray,
    negative_df: pd.DataFrame,
    n_neighbors: int = KNN_N_NEIGHBORS
) -> pd.DataFrame:
    """
    Find K-nearest neighbors in negative samples for each positive sample.
    
    Args:
        positive_features: Feature matrix for positive samples
        negative_features: Feature matrix for negative samples
        negative_df: Original DataFrame of negative samples
        n_neighbors: Number of nearest neighbors to find
        
    Returns:
        DataFrame with matched negative samples
    """
    # Initialize and fit KNN model
    knn_model = NearestNeighbors(
        n_neighbors=n_neighbors,
        algorithm=KNN_ALGORITHM
    )
    knn_model.fit(negative_features)
    
    # Find nearest neighbors
    distances, indices = knn_model.kneighbors(positive_features)
    
    # Extract matched samples
    selected_indices = indices.flatten()
    matched_negatives = negative_df.iloc[selected_indices].copy()
    
    # Remove duplicates (unlikely but possible)
    matched_negatives = matched_negatives.drop_duplicates()
    
    print(f"Matched {len(matched_negatives)} negative samples "
          f"to {len(positive_features)} positive samples")
    
    return matched_negatives


def create_balanced_dataset(
    positive_df: pd.DataFrame,
    matched_negatives_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Combine positive and matched negative samples into balanced dataset.
    
    Args:
        positive_df: Original positive samples
        matched_negatives_df: Negative samples matched via KNN
        
    Returns:
        Combined and shuffled DataFrame
    """
    # Combine positive and matched negative samples
    combined_df = pd.concat(
        [positive_df, matched_negatives_df],
        ignore_index=True
    )
    
    # Shuffle the dataset
    combined_df = combined_df.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)
    
    print(f"Balanced dataset created with {len(combined_df)} samples")
    print(f"Class distribution: {combined_df['label'].value_counts().to_dict()}")
    
    return combined_df


def split_dataset_stratified(
    df: pd.DataFrame,
    validation_size: float = VALIDATION_SIZE,
    stratify: bool = STRATIFY_SPLIT
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into training and validation sets with stratified sampling.
    
    Args:
        df: Input DataFrame to split
        validation_size: Proportion of data for validation set
        stratify: Whether to use stratified sampling
        
    Returns:
        Tuple containing training and validation DataFrames
    """
    if stratify:
        # Use sklearn's train_test_split for stratified sampling
        train_df, valid_df = train_test_split(
            df,
            test_size=validation_size,
            stratify=df['label'],
            random_state=RANDOM_STATE
        )
    else:
        # Fallback to random sampling
        train_df, valid_df = train_test_split(
            df,
            test_size=validation_size,
            random_state=RANDOM_STATE
        )
    
    # Add split labels
    train_df = train_df.copy()
    valid_df = valid_df.copy()
    train_df['split'] = 'train'
    valid_df['split'] = 'valid'
    
    print(f"Training set: {len(train_df)} samples")
    print(f"Validation set: {len(valid_df)} samples")
    print(f"Training class distribution: {train_df['label'].value_counts().to_dict()}")
    print(f"Validation class distribution: {valid_df['label'].value_counts().to_dict()}")
    
    return train_df, valid_df


def visualize_matching_results(
    positive_df: pd.DataFrame,
    matched_negatives_df: pd.DataFrame,
    output_path: str = VISUALIZATION_PATH
) -> None:
    """
    Create visualization comparing original and matched sample distributions.
    
    Args:
        positive_df: Original positive samples
        matched_negatives_df: Matched negative samples
        output_path: Path to save visualization
    """
    plt.figure(figsize=PLOT_FIGSIZE)
    
    # Plot matched negative samples
    plt.scatter(
        matched_negatives_df['miR_disease_num'],
        matched_negatives_df['gene_disease_num'],
        alpha=PLOT_ALPHA,
        label='Matched Negative Samples',
        c=NEGATIVE_COLOR,
        marker=NEGATIVE_MARKER
    )
    
    # Plot original positive samples
    plt.scatter(
        positive_df['miR_disease_num'],
        positive_df['gene_disease_num'],
        alpha=PLOT_ALPHA,
        label='Original Positive Samples',
        c=POSITIVE_COLOR,
        marker=POSITIVE_MARKER
    )
    
    plt.legend()
    plt.title("KNN-based Sample Matching Results")
    plt.xlabel("miRNA Disease Association Count")
    plt.ylabel("Gene Disease Association Count")
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Visualization saved to: {output_path}")


def main() -> None:
    """
    Main execution function orchestrating the complete pipeline.
    """
    print("=" * 60)
    print("KNN-based Sample Matching Pipeline")
    print("=" * 60)
    
    try:
        # Step 1: Load and label data
        print("\n1. Loading and labeling data...")
        experiment_df, predicted_df = load_and_label_data(INPUT_FILE_PATH)
        
        # Step 2: Standardize features
        print("\n2. Standardizing features...")
        pos_features_scaled, neg_features_scaled, scaler = standardize_features(
            experiment_df, predicted_df
        )
        
        # Step 3: KNN matching
        print("\n3. Performing KNN matching...")
        matched_negatives_df = knn_match_samples(
            pos_features_scaled,
            neg_features_scaled,
            predicted_df,
            n_neighbors=KNN_N_NEIGHBORS
        )
        
        # Step 4: Create balanced dataset
        print("\n4. Creating balanced dataset...")
        balanced_df = create_balanced_dataset(experiment_df, matched_negatives_df)
        
        # Step 5: Split into train/validation sets
        print("\n5. Splitting dataset...")
        train_df, valid_df = split_dataset_stratified(
            balanced_df,
            validation_size=VALIDATION_SIZE,
            stratify=STRATIFY_SPLIT
        )
        
        # Step 6: Combine splits and save
        print("\n6. Saving results...")
        final_df = pd.concat([train_df, valid_df], ignore_index=True)
        final_df.to_csv(OUTPUT_FILE_PATH, sep='\t', index=False)
        print(f"Final dataset saved to: {OUTPUT_FILE_PATH}")
        
        # Step 7: Create visualization
        print("\n7. Creating visualization...")
        visualize_matching_results(experiment_df, matched_negatives_df)
        
        print("\n" + "=" * 60)
        print("Pipeline completed successfully!")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"Error: Input file not found - {e}")
    except KeyError as e:
        print(f"Error: Missing required column - {e}")
    except Exception as e:
        print(f"Error: Unexpected error occurred - {e}")
        raise


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(RANDOM_STATE)
    
    # Execute main pipeline
    main()