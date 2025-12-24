"""
Example script demonstrating how to use the hybrid quantum-classical model.

This script shows a simple end-to-end example of:
1. Loading configuration
2. Creating data loaders
3. Building the hybrid model
4. Running a quick training demo
5. Evaluating the model
"""

import yaml
import torch
import warnings
warnings.filterwarnings('ignore')

from data_preprocessing import create_dataloaders
from hybrid_model import create_hybrid_model, count_parameters
from train import Trainer
from evaluate import ModelEvaluator

def main():
    """Run a simple demo of the hybrid quantum-classical model."""
    
    print("="*80)
    print("Hybrid Quantum-Classical Neural Network Demo")
    print("Alzheimer's Disease Classification from EEG Data")
    print("="*80)
    
    # Load configuration
    print("\n1. Loading configuration...")
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Override settings for quick demo
    config['data']['num_workers'] = 0
    config['data']['batch_size'] = 8
    config['training']['epochs'] = 3
    config['logging']['tensorboard'] = True
    config['logging']['save_plots'] = True
    config['training']['early_stopping']['enabled'] = False
    
    print(f"   ✓ Configuration loaded")
    print(f"   - Training for {config['training']['epochs']} epochs")
    print(f"   - Batch size: {config['data']['batch_size']}")
    print(f"   - Quantum qubits: {config['quantum_layer']['n_qubits']}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n2. Setup device: {device}")
    if device.type == 'cuda':
        print(f"   - GPU: {torch.cuda.get_device_name(0)}")
        print(f"   - Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create data loaders
    print("\n3. Creating data loaders...")
    train_loader, val_loader, test_loader = create_dataloaders(config, seed=42)
    print(f"   ✓ Data loaders created")
    print(f"   - Train batches: {len(train_loader)}")
    print(f"   - Validation batches: {len(val_loader)}")
    print(f"   - Test batches: {len(test_loader)}")
    
    # Create model
    print("\n4. Building hybrid quantum-classical model...")
    model = create_hybrid_model(config, use_ensemble=False)
    total_params, trainable_params = count_parameters(model)
    print(f"   ✓ Model created")
    print(f"   - Total parameters: {total_params:,}")
    print(f"   - Trainable parameters: {trainable_params:,}")
    
    # Display model architecture
    print("\n5. Model Architecture:")
    print("   Classical Component (BiLSTM + Attention):")
    print("      - Input: 19 EEG channels")
    print("      - BiLSTM Layer 1: 128 hidden units (bidirectional)")
    print("      - BiLSTM Layer 2: 64 hidden units (bidirectional)")
    print("      - Multi-Head Attention: 4 heads")
    print("      - Dropout: 0.3")
    print("      - Output: 16 features")
    print("\n   Quantum Component (VQC):")
    print(f"      - Qubits: {config['quantum_layer']['n_qubits']}")
    print(f"      - Layers: {config['quantum_layer']['n_layers']}")
    print(f"      - Encoding: {config['quantum_layer']['encoding']}")
    print("      - Gates: RX, RY, RZ, CNOT")
    print(f"      - Output: {config['hybrid_model']['num_classes']} classes")
    
    # Create trainer
    print("\n6. Initializing trainer...")
    trainer = Trainer(model, config, device)
    print("   ✓ Trainer initialized")
    print("   - Optimizer: Adam")
    print("   - Loss: CrossEntropyLoss")
    print("   - Scheduler: ReduceLROnPlateau")
    print("   - Gradient clipping: Enabled")
    
    # Train
    print("\n7. Starting training...")
    print("="*80)
    try:
        trainer.train(train_loader, val_loader)
    except KeyboardInterrupt:
        print("\n\n   Training interrupted by user")
    
    # Evaluate
    print("\n8. Evaluating model...")
    print("="*80)
    evaluator = ModelEvaluator(model, config, device)
    
    test_metrics = evaluator.evaluate(test_loader, split_name='test')
    val_metrics = evaluator.evaluate(val_loader, split_name='val')
    
    # Summary
    print("\n" + "="*80)
    print("DEMO COMPLETED - Results Summary")
    print("="*80)
    
    print("\nValidation Set:")
    print(f"   Accuracy:  {val_metrics['accuracy']:.4f} ({val_metrics['accuracy']*100:.2f}%)")
    print(f"   Precision: {val_metrics['precision']:.4f}")
    print(f"   Recall:    {val_metrics['recall']:.4f}")
    print(f"   F1-Score:  {val_metrics['f1_score']:.4f}")
    print(f"   ROC AUC:   {val_metrics['roc_auc']:.4f}")
    
    print("\nTest Set:")
    print(f"   Accuracy:  {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
    print(f"   Precision: {test_metrics['precision']:.4f}")
    print(f"   Recall:    {test_metrics['recall']:.4f}")
    print(f"   F1-Score:  {test_metrics['f1_score']:.4f}")
    print(f"   ROC AUC:   {test_metrics['roc_auc']:.4f}")
    
    print("\nGenerated Files:")
    print("   - Checkpoints: ./checkpoints/best_checkpoint.pth")
    print("   - Training plots: ./plots/training_history.png")
    print("   - Confusion matrices: ./plots/test_confusion_matrix.png")
    print("   - TensorBoard logs: ./logs/")
    
    print("\nTo view training progress in TensorBoard:")
    print("   $ tensorboard --logdir=./logs")
    print("\n" + "="*80)
    print("Demo completed successfully! 🎉")
    print("="*80)

if __name__ == "__main__":
    main()
