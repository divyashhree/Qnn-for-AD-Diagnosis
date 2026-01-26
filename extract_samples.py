"""
Extract 100 sample EEG files from the dataset for demo purposes.
"""

import numpy as np
import os
from pathlib import Path

def extract_samples():
    """Extract 100 random samples from the integrated dataset."""
    
    # Load the dataset
    print("Loading integrated_eeg_dataset.npz...")
    data = np.load('D:\\Ongoing\\Quantum\\Qnn-for-AD-Diagnosis\\integrated_eeg_dataset.npz', allow_pickle=True)
    
    X_data = data['X_raw']
    y_labels = data['y_labels']
    
    print(f"Total samples: {len(X_data)}")
    print(f"Classes: {np.unique(y_labels)}")
    
    # Create samples directory
    samples_dir = Path('demo_samples')
    samples_dir.mkdir(exist_ok=True)
    
    # Get 25 samples from each of 4 classes
    classes = ['AD-Auditory', 'ADFTD', 'ADFSU', 'APAVA-19']
    samples_per_class = 25
    
    sample_info = []
    
    for class_name in classes:
        class_indices = np.where(y_labels == class_name)[0]
        
        if len(class_indices) >= samples_per_class:
            selected_indices = np.random.choice(class_indices, samples_per_class, replace=False)
        else:
            selected_indices = class_indices
        
        print(f"\nExtracting {len(selected_indices)} samples for class: {class_name}")
        
        for i, idx in enumerate(selected_indices):
            sample = X_data[idx]
            label = y_labels[idx]
            
            # Transpose to (19, 128) if needed
            if sample.shape == (128, 19):
                sample = sample.T
            
            # Save as .npy file
            filename = f"{class_name}_{i+1:03d}.npy"
            filepath = samples_dir / filename
            np.save(filepath, sample)
            
            sample_info.append({
                'filename': filename,
                'label': label,
                'shape': sample.shape
            })
    
    # Save metadata
    metadata = {
        'total_samples': len(sample_info),
        'classes': classes,
        'samples_per_class': samples_per_class,
        'sample_shape': X_data[0].shape,
        'samples': sample_info
    }
    
    np.save(samples_dir / 'metadata.npy', metadata, allow_pickle=True)
    
    print(f"\n✓ Extracted {len(sample_info)} samples to {samples_dir}/")
    print(f"✓ Sample shape: {X_data[0].shape} (19 channels, 128 timepoints)")
    
    return samples_dir

if __name__ == '__main__':
    extract_samples()
