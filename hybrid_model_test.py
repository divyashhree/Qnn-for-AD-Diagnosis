"""
Hybrid Model with Debug Mode to disable quantum layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import logging

from classical_model import BiLSTMWithAttention, create_classical_model
from quantum_layer import HybridQuantumLayer, create_quantum_layer

logger = logging.getLogger(__name__)


class HybridModel(nn.Module):
    """
    Hybrid Quantum-Classical Neural Network with debug mode.
    """

    def __init__(
        self,
        classical_model: BiLSTMWithAttention,
        quantum_layer: HybridQuantumLayer,
        num_classes: int,
        dropout: float = 0.3,
        use_quantum: bool = True  # NEW: Flag to enable/disable quantum
    ):
        """
        Initialize Hybrid Model.

        Args:
            classical_model: BiLSTM with Attention model
            quantum_layer: Hybrid quantum layer
            num_classes: Number of output classes
            dropout: Dropout probability for output layer
            use_quantum: Whether to use quantum layer (False for debugging)
        """
        super(HybridModel, self).__init__()

        self.classical_model = classical_model
        self.quantum_layer = quantum_layer
        self.num_classes = num_classes
        self.use_quantum = use_quantum  # NEW

        # Additional classical layers after quantum processing
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(num_classes, num_classes)

        # NEW: If not using quantum, add a bridge layer from classical to output
        if not use_quantum:
            self.classical_to_output = nn.Sequential(
                nn.Linear(classical_model.output_dim, num_classes),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

        # Batch normalization for stability
        self.batch_norm = nn.BatchNorm1d(num_classes)

        logger.info(f"Initialized HybridModel with {num_classes} output classes")
        logger.info(f"Quantum layer enabled: {use_quantum}")

    def forward(
        self,
        x: torch.Tensor,
        return_intermediate: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through hybrid model.

        Args:
            x: Input tensor of shape (batch_size, num_channels, seq_len)
            return_intermediate: Whether to return intermediate features

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Check for NaN in input
        if torch.isnan(x).any():
            logger.warning("NaN detected in input, replacing with zeros")
            x = torch.nan_to_num(x, nan=0.0)

        # Classical feature extraction
        classical_features = self.classical_model(x)

        # Check for NaN after classical model
        if torch.isnan(classical_features).any():
            logger.warning("NaN detected after classical model, replacing with zeros")
            classical_features = torch.nan_to_num(classical_features, nan=0.0)

        # NEW: Choose quantum or classical-only path
        if self.use_quantum:
            # Original quantum path
            quantum_features = self.quantum_layer(classical_features)

            # Check for NaN after quantum layer
            if torch.isnan(quantum_features).any():
                logger.warning("NaN detected after quantum layer, replacing with zeros")
                quantum_features = torch.nan_to_num(quantum_features, nan=0.0)

            # Batch normalization
            if quantum_features.shape[0] > 1:
                quantum_features = self.batch_norm(quantum_features)

            # Output layer with dropout
            output = self.dropout(quantum_features)
            output = self.output_layer(output)
        else:
            # Classical-only path (bypass quantum layer)
            logger.debug("Using classical-only path (quantum disabled)")
            quantum_features = self.classical_to_output(classical_features)
            
            # Batch normalization
            if quantum_features.shape[0] > 1:
                quantum_features = self.batch_norm(quantum_features)
            
            output = quantum_features

        # Final NaN check
        if torch.isnan(output).any():
            logger.warning("NaN detected in final output, replacing with zeros")
            output = torch.nan_to_num(output, nan=0.0)

        # Clamp output to prevent extreme values
        output = torch.clamp(output, -10.0, 10.0)

        if return_intermediate:
            return output, classical_features, quantum_features

        return output


def create_hybrid_model(config: dict, use_ensemble: bool = False, 
                       use_quantum: bool = True) -> nn.Module:
    """
    Create hybrid quantum-classical model from configuration.

    Args:
        config: Configuration dictionary
        use_ensemble: Whether to use ensemble of quantum circuits
        use_quantum: Whether to use quantum layer (NEW)

    Returns:
        HybridModel instance
    """
    # Create classical model
    classical_model = create_classical_model(config)

    if use_ensemble:
        raise NotImplementedError("Ensemble mode not supported in debug mode")
    else:
        # Create quantum layer (even if not using it, to avoid errors)
        quantum_layer = create_quantum_layer(config)

        # Create hybrid model with quantum enable/disable flag
        model = HybridModel(
            classical_model=classical_model,
            quantum_layer=quantum_layer,
            num_classes=config['hybrid_model']['num_classes'],
            dropout=config['classical_model']['lstm_layers'][0]['dropout'],
            use_quantum=use_quantum  # NEW
        )

    return model


if __name__ == "__main__":
    # Test both modes
    import yaml

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    print("Testing Classical-Only Mode (Quantum Disabled)")
    print("=" * 80)
    model_classical = create_hybrid_model(config, use_quantum=False)
    
    batch_size = 4
    num_channels = config['data']['num_channels']
    seq_len = int(config['data']['window_size'] * config['data']['sampling_rate'])
    x = torch.randn(batch_size, num_channels, seq_len)
    
    output = model_classical(x)
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    print("\n" + "=" * 80)
    print("Testing Hybrid Mode (Quantum Enabled)")
    print("=" * 80)
    model_quantum = create_hybrid_model(config, use_quantum=True)
    output = model_quantum(x)
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")