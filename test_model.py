"""
Quick diagnostic script to test model components.
Run this before full training to catch issues early.
"""

import torch
import numpy as np
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
'''
def test_data():
    """Test 1: Validate data"""
    print("\n" + "="*80)
    print("TEST 1: DATA VALIDATION")
    print("="*80)
    
    try:
        data = np.load('./integrated_eeg_dataset.npz', allow_pickle=True)
        
        # Load features and labels
        X_train = data['X_features']
        y_train = data['y_labels']
        
        print(f"✓ Data loaded successfully")
        print(f"  X_features shape: {X_train.shape}")
        print(f"  y_labels shape: {y_train.shape}")
        print(f"  y_labels type: {y_train.dtype}")
        
        # Check if labels are strings - if so, convert to integers
        if y_train.dtype == object or y_train.dtype.kind == 'U':
            print(f"  Labels are strings, converting to integers...")
            unique_labels = np.unique(y_train)
            print(f"  Unique labels: {unique_labels}")
            
            # Create label mapping
            label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            print(f"  Label mapping: {label_to_idx}")
            
            # Convert to integers
            y_train_int = np.array([label_to_idx[label] for label in y_train])
            y_train = y_train_int
        
        # Convert to tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        
        # Check for issues
        has_nan = torch.isnan(X_train).any()
        has_inf = torch.isinf(X_train).any()
        
        print(f"  NaN values: {'❌ YES' if has_nan else '✓ NO'}")
        print(f"  Inf values: {'❌ YES' if has_inf else '✓ NO'}")
        print(f"  X Min: {X_train.min():.6f}")
        print(f"  X Max: {X_train.max():.6f}")
        print(f"  X Mean: {X_train.mean():.6f}")
        print(f"  X Std: {X_train.std():.6f}")
        print(f"  Label range: {y_train.min()} to {y_train.max()}")
        
        if has_nan or has_inf:
            print("\n❌ DATA HAS ISSUES - Clean it first!")
            return False
        else:
            print("\n✓ Data looks good!")
            return True
            
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return False
 '''   
def test_data():
    """Test 1: Validate data"""
    print("\n" + "="*80)
    print("TEST 1: DATA VALIDATION")
    print("="*80)
    
    try:
        data = np.load('./integrated_eeg_dataset.npz', allow_pickle=True)
        
        # Load features and labels
        X_train = data['X_features']
        y_train = data['y_labels']
        
        print(f"✓ Data loaded successfully")
        print(f"  X_features shape: {X_train.shape}")
        print(f"  y_labels shape: {y_train.shape}")
        print(f"  y_labels type: {y_train.dtype}")
        
        # If labels are 2D, inspect and choose the right column
        if len(y_train.shape) == 2:
            print(f"\n  Labels are 2D with {y_train.shape[1]} columns")
            print(f"  First 5 rows of labels:")
            for i in range(min(5, len(y_train))):
                print(f"    {y_train[i]}")
            
            # Ask which column to use (or automatically select)
            # For now, let's check which column has the actual class labels
            print(f"\n  Checking each column:")
            for col_idx in range(y_train.shape[1]):
                col_data = y_train[:, col_idx]
                unique_vals = np.unique(col_data)
                print(f"    Column {col_idx}: {len(unique_vals)} unique values - {unique_vals[:10]}")
            
            # Automatically select the column with AD-related labels
            # Usually the last column has the actual diagnosis
            for col_idx in range(y_train.shape[1]):
                col_data = y_train[:, col_idx]
                unique_vals = np.unique(col_data)
                # Check if this column has AD-related labels
                if any('AD' in str(val) for val in unique_vals):
                    print(f"\n  ✓ Using column {col_idx} (contains AD labels)")
                    y_train = col_data
                    break
            else:
                # If no AD labels found, use the last column
                print(f"\n  Using last column by default")
                y_train = y_train[:, -1]
        
        # Now convert string labels to integers
        if y_train.dtype == object or y_train.dtype.kind == 'U':
            print(f"\n  Converting string labels to integers...")
            unique_labels = np.unique(y_train)
            print(f"  Unique labels ({len(unique_labels)}): {unique_labels[:20]}...")  # Show first 20
            
            # Create label mapping
            label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            
            # Convert to integers
            y_train_int = np.array([label_to_idx[label] for label in y_train])
            y_train = y_train_int
            
            print(f"  Mapped to integer range: 0 to {len(unique_labels)-1}")
        
        # Convert to tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        
        print(f"\n  Final data shapes:")
        print(f"    X: {X_train.shape}")
        print(f"    y: {y_train.shape}")
        
        # Check for issues
        has_nan = torch.isnan(X_train).any()
        has_inf = torch.isinf(X_train).any()
        
        print(f"\n  Data quality checks:")
        print(f"    NaN values: {'❌ YES' if has_nan else '✓ NO'}")
        print(f"    Inf values: {'❌ YES' if has_inf else '✓ NO'}")
        print(f"    X Min: {X_train.min():.6f}")
        print(f"    X Max: {X_train.max():.6f}")
        print(f"    X Mean: {X_train.mean():.6f}")
        print(f"    X Std: {X_train.std():.6f}")
        print(f"    Label range: {y_train.min()} to {y_train.max()}")
        print(f"    Number of classes: {len(torch.unique(y_train))}")
        
        if has_nan or has_inf:
            print("\n❌ DATA HAS ISSUES - Clean it first!")
            return False
        else:
            print("\n✓ Data looks good!")
            return True
            
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_classical_model():
    """Test 2: Classical model"""
    print("\n" + "="*80)
    print("TEST 2: CLASSICAL MODEL")
    print("="*80)
    
    try:
        from classical_model import create_classical_model
        
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        model = create_classical_model(config)
        print("✓ Model created")
        
        # Create dummy input
        batch_size = 4
        num_channels = config['data']['num_channels']
        seq_len = int(config['data']['window_size'] * config['data']['sampling_rate'])
        x = torch.randn(batch_size, num_channels, seq_len)
        
        print(f"  Input shape: {x.shape}")
        
        # Forward pass
        output = model(x)
        
        has_nan = torch.isnan(output).any()
        has_inf = torch.isinf(output).any()
        
        print(f"  Output shape: {output.shape}")
        print(f"  NaN in output: {'❌ YES' if has_nan else '✓ NO'}")
        print(f"  Inf in output: {'❌ YES' if has_inf else '✓ NO'}")
        print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
        
        if has_nan or has_inf:
            print("\n❌ CLASSICAL MODEL PRODUCES NaN/Inf!")
            return False
        else:
            print("\n✓ Classical model works!")
            return True
            
    except Exception as e:
        print(f"❌ Error in classical model: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_quantum_layer():
    """Test 3: Quantum layer"""
    print("\n" + "="*80)
    print("TEST 3: QUANTUM LAYER")
    print("="*80)
    
    try:
        from quantum_layer import create_quantum_layer
        
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        quantum_layer = create_quantum_layer(config)
        print("✓ Quantum layer created")
        
        # Create dummy input
        batch_size = 4
        input_dim = config['classical_model']['dense_output']
        x = torch.randn(batch_size, input_dim) * 0.1  # Small values
        
        print(f"  Input shape: {x.shape}")
        
        # Forward pass
        output = quantum_layer(x)
        
        has_nan = torch.isnan(output).any()
        has_inf = torch.isinf(output).any()
        
        print(f"  Output shape: {output.shape}")
        print(f"  NaN in output: {'❌ YES' if has_nan else '✓ NO'}")
        print(f"  Inf in output: {'❌ YES' if has_inf else '✓ NO'}")
        print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
        
        if has_nan or has_inf:
            print("\n❌ QUANTUM LAYER PRODUCES NaN/Inf!")
            return False
        else:
            print("\n✓ Quantum layer works!")
            return True
            
    except Exception as e:
        print(f"❌ Error in quantum layer: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hybrid_model():
    """Test 4: Full hybrid model"""
    print("\n" + "="*80)
    print("TEST 4: HYBRID MODEL (with quantum)")
    print("="*80)
    
    try:
        # Import the updated hybrid_model with use_quantum flag
        from hybrid_model_test import create_hybrid_model
        
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Test without quantum first
        print("\n--- Testing WITHOUT quantum ---")
        model_no_q = create_hybrid_model(config, use_quantum=False)
        
        batch_size = 4
        num_channels = config['data']['num_channels']
        seq_len = int(config['data']['window_size'] * config['data']['sampling_rate'])
        x = torch.randn(batch_size, num_channels, seq_len)
        
        output = model_no_q(x)
        has_nan = torch.isnan(output).any()
        print(f"  NaN in output: {'❌ YES' if has_nan else '✓ NO'}")
        
        if has_nan:
            print("❌ Classical-only model has NaN!")
            return False
        
        # Test with quantum
        print("\n--- Testing WITH quantum ---")
        model_with_q = create_hybrid_model(config, use_quantum=True)
        output = model_with_q(x)
        has_nan = torch.isnan(output).any()
        print(f"  NaN in output: {'❌ YES' if has_nan else '✓ NO'}")
        
        if has_nan:
            print("❌ Hybrid model with quantum has NaN!")
            return False
        else:
            print("\n✓ Full hybrid model works!")
            return True
            
    except Exception as e:
        print(f"❌ Error in hybrid model: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "#"*80)
    print("# MODEL DIAGNOSTIC TEST SUITE")
    print("#"*80)
    
    results = {
        'Data': test_data(),
        'Classical Model': test_classical_model(),
        'Quantum Layer': test_quantum_layer(),
        'Hybrid Model': test_hybrid_model()
    }
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*80)
    if all_passed:
        print("🎉 ALL TESTS PASSED! You can start training.")
    else:
        print("⚠️  SOME TESTS FAILED! Fix the issues before training.")
        print("\nRecommended actions:")
        if not results['Data']:
            print("  1. Clean your data (remove NaN/Inf, normalize)")
        if not results['Classical Model']:
            print("  2. Fix classical model initialization")
        if not results['Quantum Layer']:
            print("  3. Fix quantum layer (lower init values, check encoding)")
        if not results['Hybrid Model']:
            print("  4. Check hybrid model connections")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()