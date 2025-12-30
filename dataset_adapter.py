"""
Dataset Adapter for integrated_eeg_dataset.npz format.

This module provides compatibility layer between the integrated EEG dataset
and the existing data preprocessing pipeline.
"""

import numpy as np
import logging
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)


def load_integrated_eeg_dataset(
    data_path: str,
    binary_classification: bool = False,
    classes_to_keep: list = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and adapt the integrated_eeg_dataset.npz format.
    
    Dataset structure:
    - X_raw: shape (101916, 128, 19) - Raw EEG signals (samples, timepoints, channels)
    - y_labels: shape (101916, 3) - Labels with 3 columns
    - X_features: shape (101916, 76) - Extracted features
    
    Classes in dataset:
    1. AD-Auditory (30.08%) - Alzheimer's Disease Auditory
    2. ADFSU (2.44%) - Alzheimer's Disease Follow-up
    3. ADFTD (62.58%) - Alzheimer's Disease Frontotemporal Dementia
    4. ADSZ (0.62%) - Alzheimer's Disease Seizure
    5. APAVA-19 (4.27%) - Alzheimer's Disease variant
    
    Args:
        data_path: Path to the .npz file
        binary_classification: If True, convert to binary (AD vs non-AD)
        classes_to_keep: List of class names to keep (None = keep all)
    
    Returns:
        Tuple of (data, labels)
        - data: shape (n_samples, n_channels, n_timepoints)
        - labels: shape (n_samples,) - integer labels starting from 0
    """
    logger.info(f"Loading integrated EEG dataset from {data_path}")
    
    # Load the NPZ file
    dataset = np.load(data_path)
    
    # Extract raw EEG signals: (samples, timepoints, channels) -> (samples, channels, timepoints)
    X_raw = dataset['X_raw']
    logger.info(f"Original X_raw shape: {X_raw.shape}")
    
    # Transpose to match expected format: (samples, channels, timepoints)
    X_data = np.transpose(X_raw, (0, 2, 1))
    logger.info(f"Transposed to: {X_data.shape}")
    
    # Extract diagnosis labels (third column)
    y_labels_raw = dataset['y_labels'][:, 2]
    
    # Get unique classes
    unique_classes = np.unique(y_labels_raw)
    logger.info(f"Found {len(unique_classes)} unique classes: {unique_classes}")
    
    # Count samples per class
    for cls in unique_classes:
        count = np.sum(y_labels_raw == cls)
        percentage = (count / len(y_labels_raw)) * 100
        logger.info(f"  {cls}: {count} samples ({percentage:.2f}%)")
    
    # Filter classes if specified
    if classes_to_keep is not None:
        logger.info(f"Filtering to keep only classes: {classes_to_keep}")
        mask = np.isin(y_labels_raw, classes_to_keep)
        X_data = X_data[mask]
        y_labels_raw = y_labels_raw[mask]
        logger.info(f"After filtering: {len(X_data)} samples")
    
    # Convert to numeric labels
    if binary_classification:
        logger.info("Converting to binary classification (AD vs non-AD)")
        # All classes are AD-related, so for true binary we'd need healthy controls
        # For now, we'll use the largest class (ADFTD) as class 0, others as class 1
        y_labels = np.where(y_labels_raw == 'ADFTD', 0, 1)
        logger.info(f"Binary distribution - Class 0: {np.sum(y_labels == 0)}, Class 1: {np.sum(y_labels == 1)}")
    else:
        # Multi-class: create mapping from string labels to integers
        class_mapping = {cls: idx for idx, cls in enumerate(sorted(unique_classes))}
        logger.info(f"Class mapping: {class_mapping}")
        
        y_labels = np.array([class_mapping[label] for label in y_labels_raw], dtype=np.int64)
    
    # Convert to float32 for memory efficiency
    X_data = X_data.astype(np.float32)
    
    logger.info(f"Final data shape: {X_data.shape}")
    logger.info(f"Final labels shape: {y_labels.shape}")
    logger.info(f"Data range: [{X_data.min():.3f}, {X_data.max():.3f}]")
    logger.info(f"Labels range: [{y_labels.min()}, {y_labels.max()}]")
    
    return X_data, y_labels


def get_class_info(data_path: str) -> Dict[str, Any]:
    """
    Get information about classes in the dataset.
    
    Args:
        data_path: Path to the .npz file
    
    Returns:
        Dictionary with class information
    """
    dataset = np.load(data_path)
    y_labels_raw = dataset['y_labels'][:, 2]
    
    unique_classes = np.unique(y_labels_raw)
    class_counts = {}
    
    for cls in unique_classes:
        count = np.sum(y_labels_raw == cls)
        percentage = (count / len(y_labels_raw)) * 100
        class_counts[str(cls)] = {
            'count': int(count),
            'percentage': float(percentage)
        }
    
    return {
        'total_samples': len(y_labels_raw),
        'num_classes': len(unique_classes),
        'classes': list(unique_classes),
        'class_distribution': class_counts
    }


if __name__ == "__main__":
    # Test the adapter
    import yaml
    
    logging.basicConfig(level=logging.INFO)
    
    # Change dataset path to root folder
    data_path = "./integrated_eeg_dataset.npz"
    
    print("\n" + "="*80)
    print("DATASET INFORMATION")
    print("="*80)
    
    info = get_class_info(data_path)
    print(f"\nTotal samples: {info['total_samples']:,}")
    print(f"Number of classes: {info['num_classes']}")
    print(f"\nClasses:")
    for cls in info['classes']:
        dist = info['class_distribution'][cls]
        print(f"  {cls:20s}: {dist['count']:6,} samples ({dist['percentage']:5.2f}%)")
    
    print("\n" + "="*80)
    print("LOADING DATASET (Multi-class)")
    print("="*80)
    
    X, y = load_integrated_eeg_dataset(data_path, binary_classification=False)
    print(f"\nX shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"y unique values: {np.unique(y)}")
    
    print("\n" + "="*80)
    print("LOADING DATASET (Binary)")
    print("="*80)
    
    X_bin, y_bin = load_integrated_eeg_dataset(data_path, binary_classification=True)
    print(f"\nX shape: {X_bin.shape}")
    print(f"y shape: {y_bin.shape}")
    print(f"y unique values: {np.unique(y_bin)}")
    print(f"Binary distribution - Class 0: {np.sum(y_bin == 0):,}, Class 1: {np.sum(y_bin == 1):,}")
