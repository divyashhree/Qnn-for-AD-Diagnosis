"""
Data preprocessing module for EEG signal processing.

This module handles loading, filtering, normalization, and segmentation of EEG data
from the Kaggle Alzheimer's disease dataset.
"""

import os
from typing import Tuple, Optional, List, Dict, Any
import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import butter, filtfilt
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EEGPreprocessor:
    """
    EEG signal preprocessing class for bandpass filtering and normalization.
    """

    def __init__(
        self,
        sampling_rate: int = 128,
        lowcut: float = 0.5,
        highcut: float = 50.0,
        filter_order: int = 5
    ):
        """
        Initialize EEG preprocessor.

        Args:
            sampling_rate: Sampling frequency in Hz
            lowcut: Lower cutoff frequency for bandpass filter (Hz)
            highcut: Upper cutoff frequency for bandpass filter (Hz)
            filter_order: Order of Butterworth filter
        """
        self.sampling_rate = sampling_rate
        self.lowcut = lowcut
        self.highcut = highcut
        self.filter_order = filter_order

    def bandpass_filter(self, data: np.ndarray) -> np.ndarray:
        """
        Apply Butterworth bandpass filter to EEG data.

        Args:
            data: EEG data array of shape (n_channels, n_samples) or (n_samples,)

        Returns:
            Filtered EEG data with same shape as input
        """
        nyquist = 0.5 * self.sampling_rate
        low = self.lowcut / nyquist
        high = self.highcut / nyquist

        b, a = butter(self.filter_order, [low, high], btype='band')

        # Handle both 1D and 2D arrays
        if data.ndim == 1:
            filtered_data = filtfilt(b, a, data)
        else:
            filtered_data = np.zeros_like(data)
            for i in range(data.shape[0]):
                filtered_data[i] = filtfilt(b, a, data[i])

        return filtered_data

    def normalize(self, data: np.ndarray, method: str = 'zscore') -> np.ndarray:
        """
        Normalize EEG data using z-score normalization.

        Args:
            data: EEG data array of shape (n_channels, n_samples)
            method: Normalization method ('zscore', 'minmax')

        Returns:
            Normalized EEG data
        """
        if method == 'zscore':
            # Z-score normalization per channel
            mean = np.mean(data, axis=-1, keepdims=True)
            std = np.std(data, axis=-1, keepdims=True)
            std = np.where(std == 0, 1.0, std)  # Avoid division by zero
            normalized_data = (data - mean) / std
        elif method == 'minmax':
            # Min-max normalization
            min_val = np.min(data, axis=-1, keepdims=True)
            max_val = np.max(data, axis=-1, keepdims=True)
            range_val = max_val - min_val
            range_val = np.where(range_val == 0, 1.0, range_val)
            normalized_data = (data - min_val) / range_val
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return normalized_data

    def remove_artifacts(
        self,
        data: np.ndarray,
        threshold: float = 5.0
    ) -> np.ndarray:
        """
        Remove artifacts from EEG data using threshold-based detection.

        Args:
            data: EEG data array
            threshold: Z-score threshold for artifact detection

        Returns:
            Data with artifacts replaced by interpolation
        """
        # Calculate z-scores
        mean = np.mean(data, axis=-1, keepdims=True)
        std = np.std(data, axis=-1, keepdims=True)
        std = np.where(std == 0, 1.0, std)
        z_scores = np.abs((data - mean) / std)

        # Identify artifacts
        artifacts = z_scores > threshold

        # Replace artifacts with interpolated values
        if np.any(artifacts):
            for i in range(data.shape[0]):
                if np.any(artifacts[i]):
                    artifact_indices = np.where(artifacts[i])[0]
                    clean_indices = np.where(~artifacts[i])[0]
                    if len(clean_indices) > 0:
                        data[i, artifact_indices] = np.interp(
                            artifact_indices,
                            clean_indices,
                            data[i, clean_indices]
                        )

        return data


class EEGDataset(Dataset):
    """
    PyTorch Dataset for EEG data with windowing and preprocessing.
    """

    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        window_size: int,
        overlap: float = 0.5,
        preprocessor: Optional[EEGPreprocessor] = None
    ):
        """
        Initialize EEG Dataset.

        Args:
            data: EEG data array of shape (n_samples, n_channels, n_timepoints)
            labels: Labels array of shape (n_samples,)
            window_size: Window size in number of samples
            overlap: Overlap ratio between windows (0.0 to 1.0)
            preprocessor: EEGPreprocessor instance for signal processing
        """
        self.data = data
        self.labels = labels
        self.window_size = window_size
        self.overlap = overlap
        self.preprocessor = preprocessor

        # Calculate stride for windowing
        self.stride = int(window_size * (1 - overlap))

        # Generate windows
        self.windows, self.window_labels = self._create_windows()

    def _create_windows(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create overlapping windows from continuous EEG data.

        Returns:
            Tuple of (windowed_data, window_labels)
        """
        all_windows = []
        all_labels = []

        for i in range(len(self.data)):
            signal_data = self.data[i]
            label = self.labels[i]

            # Calculate number of windows for this sample
            n_timepoints = signal_data.shape[-1]
            n_windows = (n_timepoints - self.window_size) // self.stride + 1

            for j in range(n_windows):
                start_idx = j * self.stride
                end_idx = start_idx + self.window_size

                window = signal_data[:, start_idx:end_idx]

                # Apply preprocessing if available
                if self.preprocessor is not None:
                    window = self.preprocessor.bandpass_filter(window)
                    window = self.preprocessor.remove_artifacts(window)
                    window = self.preprocessor.normalize(window)

                all_windows.append(window)
                all_labels.append(label)

        windows = np.array(all_windows, dtype=np.float32)
        window_labels = np.array(all_labels, dtype=np.int64)

        return windows, window_labels

    def __len__(self) -> int:
        """Return number of windows in dataset."""
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single window and its label.

        Args:
            idx: Index of the window

        Returns:
            Tuple of (window_tensor, label_tensor)
        """
        window = self.windows[idx]
        label = self.window_labels[idx]

        # Convert to PyTorch tensors
        window_tensor = torch.from_numpy(window).float()
        label_tensor = torch.tensor(label, dtype=torch.long)

        # Handle NaN values
        if torch.isnan(window_tensor).any():
            logger.warning(f"NaN values found in window {idx}, replacing with zeros")
            window_tensor = torch.nan_to_num(window_tensor, nan=0.0)

        return window_tensor, label_tensor


def load_eeg_data(
    data_path: str,
    config: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load EEG data from various file formats.

    Args:
        data_path: Path to data directory or file
        config: Configuration dictionary

    Returns:
        Tuple of (data, labels)
        - data: shape (n_samples, n_channels, n_timepoints)
        - labels: shape (n_samples,)
    """
    logger.info(f"Loading EEG data from {data_path}")

    # Check if path exists
    if not os.path.exists(data_path):
        logger.warning(f"Data path {data_path} does not exist. Creating dummy data for demonstration.")
        return _create_dummy_data(config)

    # Try loading different file formats
    if os.path.isfile(data_path):
        if data_path.endswith('.npy'):
            data = np.load(data_path)
        elif data_path.endswith('.npz'):
            loaded = np.load(data_path)
            data = loaded['data']
            labels = loaded['labels']
            return data, labels
        elif data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
            # Assume last column is label, rest are features
            labels = df.iloc[:, -1].values
            data = df.iloc[:, :-1].values
            # Reshape to (n_samples, n_channels, n_timepoints)
            n_samples = len(data)
            n_channels = config['data']['num_channels']
            n_timepoints = len(data[0]) // n_channels
            data = data.reshape(n_samples, n_channels, n_timepoints)
            return data, labels
    else:
        # Directory of files
        logger.warning(f"Directory loading not implemented. Creating dummy data.")
        return _create_dummy_data(config)

    # Default: create dummy data if format not recognized
    logger.warning(f"Unrecognized data format. Creating dummy data for demonstration.")
    return _create_dummy_data(config)


def _create_dummy_data(config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create dummy EEG data for testing and demonstration.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (data, labels)
    """
    logger.info("Creating dummy EEG data for demonstration")

    n_samples = 100
    n_channels = config['data']['num_channels']
    sampling_rate = config['data']['sampling_rate']
    duration = 10  # seconds
    n_timepoints = sampling_rate * duration

    # Generate synthetic EEG-like data
    data = []
    labels = []

    for i in range(n_samples):
        # Generate random EEG-like signal with multiple frequency components
        t = np.linspace(0, duration, n_timepoints)
        channels = []

        for ch in range(n_channels):
            # Mix of different frequency bands typical in EEG
            alpha = np.sin(2 * np.pi * 10 * t) * np.random.randn()  # 10 Hz alpha
            beta = np.sin(2 * np.pi * 20 * t) * np.random.randn()   # 20 Hz beta
            noise = np.random.randn(n_timepoints) * 0.1
            channel_signal = alpha + beta + noise
            channels.append(channel_signal)

        data.append(np.array(channels))

        # Binary labels (0: Healthy, 1: Alzheimer's)
        labels.append(i % 2)

    data = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    logger.info(f"Created dummy data: {data.shape}, labels: {labels.shape}")
    return data, labels


def create_dataloaders(
    config: Dict[str, Any],
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.

    Args:
        config: Configuration dictionary
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load data
    data_path = config['data']['dataset_path']
    data, labels = load_eeg_data(data_path, config)

    logger.info(f"Loaded data shape: {data.shape}, labels shape: {labels.shape}")

    # Create preprocessor
    preprocessor = EEGPreprocessor(
        sampling_rate=config['data']['sampling_rate'],
        lowcut=config['data']['bandpass_filter']['low'],
        highcut=config['data']['bandpass_filter']['high']
    )

    # Calculate window size in samples
    window_size_samples = int(
        config['data']['window_size'] * config['data']['sampling_rate']
    )

    # Create dataset
    dataset = EEGDataset(
        data=data,
        labels=labels,
        window_size=window_size_samples,
        overlap=config['data']['overlap'],
        preprocessor=preprocessor
    )

    logger.info(f"Total windows created: {len(dataset)}")

    # Split dataset
    train_ratio = config['data']['train_split']
    val_ratio = config['data']['val_split']
    test_ratio = config['data']['test_split']

    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )

    logger.info(f"Dataset splits - Train: {train_size}, Val: {val_size}, Test: {test_size}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test data preprocessing
    import yaml

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    train_loader, val_loader, test_loader = create_dataloaders(config)

    # Test batch
    for batch_data, batch_labels in train_loader:
        print(f"Batch data shape: {batch_data.shape}")
        print(f"Batch labels shape: {batch_labels.shape}")
        print(f"Data range: [{batch_data.min():.3f}, {batch_data.max():.3f}]")
        break
