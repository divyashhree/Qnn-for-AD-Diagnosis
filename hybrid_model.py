"""
Hybrid Quantum-Classical Neural Network Model.

This module combines the classical BiLSTM-Attention model with the quantum layer
to create a hybrid model for Alzheimer's disease classification.
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
    Hybrid Quantum-Classical Neural Network for EEG-based AD classification.

    Architecture:
    1. Classical BiLSTM with Multi-Head Attention for temporal feature extraction
    2. Quantum layer for quantum feature transformation
    3. Classical output layer for final classification
    """

    def __init__(
        self,
        classical_model: BiLSTMWithAttention,
        quantum_layer: HybridQuantumLayer,
        num_classes: int,
        dropout: float = 0.3
    ):
        """
        Initialize Hybrid Model.

        Args:
            classical_model: BiLSTM with Attention model
            quantum_layer: Hybrid quantum layer
            num_classes: Number of output classes
            dropout: Dropout probability for output layer
        """
        super(HybridModel, self).__init__()

        self.classical_model = classical_model
        self.quantum_layer = quantum_layer
        self.num_classes = num_classes

        # Additional classical layers after quantum processing
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(num_classes, num_classes)

        # Batch normalization for stability
        self.batch_norm = nn.BatchNorm1d(num_classes)

        logger.info(f"Initialized HybridModel with {num_classes} output classes")

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
            If return_intermediate=True, returns tuple (output, classical_features, quantum_features)
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

        # Quantum processing
        quantum_features = self.quantum_layer(classical_features)

        # Check for NaN after quantum layer
        if torch.isnan(quantum_features).any():
            logger.warning("NaN detected after quantum layer, replacing with zeros")
            quantum_features = torch.nan_to_num(quantum_features, nan=0.0)

        # Batch normalization
        if quantum_features.shape[0] > 1:  # Batch norm requires batch_size > 1
            quantum_features = self.batch_norm(quantum_features)

        # Output layer with dropout
        output = self.dropout(quantum_features)
        output = self.output_layer(output)

        # Final NaN check
        if torch.isnan(output).any():
            logger.warning("NaN detected in final output, replacing with zeros")
            output = torch.nan_to_num(output, nan=0.0)

        # Clamp output to prevent extreme values
        output = torch.clamp(output, -10.0, 10.0)

        if return_intermediate:
            return output, classical_features, quantum_features

        return output


class HybridModelWithEnsemble(nn.Module):
    """
    Enhanced hybrid model with ensemble of quantum circuits for robustness.
    """

    def __init__(
        self,
        classical_model: BiLSTMWithAttention,
        num_classes: int,
        n_quantum_circuits: int = 3,
        n_qubits: int = 10,
        n_layers: int = 3,
        dropout: float = 0.3
    ):
        """
        Initialize Hybrid Model with Ensemble.

        Args:
            classical_model: Classical model component
            num_classes: Number of output classes
            n_quantum_circuits: Number of quantum circuits in ensemble
            n_qubits: Number of qubits per circuit
            n_layers: Number of variational layers
            dropout: Dropout probability
        """
        super(HybridModelWithEnsemble, self).__init__()

        self.classical_model = classical_model
        self.num_classes = num_classes
        self.n_quantum_circuits = n_quantum_circuits

        # Create ensemble of quantum layers
        self.quantum_ensemble = nn.ModuleList([
            HybridQuantumLayer(
                input_dim=classical_model.output_dim,
                output_dim=num_classes,
                n_qubits=n_qubits,
                n_layers=n_layers,
                use_classical_preprocessing=False
            )
            for _ in range(n_quantum_circuits)
        ])

        # Attention weights for ensemble
        self.ensemble_attention = nn.Parameter(
            torch.ones(n_quantum_circuits) / n_quantum_circuits
        )

        # Output processing
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(num_classes, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ensemble hybrid model.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        # Classical feature extraction
        classical_features = self.classical_model(x)

        # Process through quantum ensemble
        quantum_outputs = []
        for quantum_layer in self.quantum_ensemble:
            output = quantum_layer(classical_features)
            quantum_outputs.append(output)

        # Stack and weight ensemble outputs
        quantum_outputs = torch.stack(quantum_outputs, dim=0)  # (n_circuits, batch_size, num_classes)

        # Apply attention weights
        attention_weights = F.softmax(self.ensemble_attention, dim=0)
        weighted_output = torch.sum(
            quantum_outputs * attention_weights.view(-1, 1, 1),
            dim=0
        )

        # Output layer
        output = self.dropout(weighted_output)
        output = self.output_layer(output)

        return output


def create_hybrid_model(config: dict, use_ensemble: bool = False) -> nn.Module:
    """
    Create hybrid quantum-classical model from configuration.

    Args:
        config: Configuration dictionary
        use_ensemble: Whether to use ensemble of quantum circuits

    Returns:
        HybridModel or HybridModelWithEnsemble instance
    """
    # Create classical model
    classical_model = create_classical_model(config)

    if use_ensemble:
        # Create ensemble hybrid model
        model = HybridModelWithEnsemble(
            classical_model=classical_model,
            num_classes=config['hybrid_model']['num_classes'],
            n_quantum_circuits=3,
            n_qubits=config['quantum_layer']['n_qubits'],
            n_layers=config['quantum_layer']['n_layers'],
            dropout=config['classical_model']['lstm_layers'][0]['dropout']
        )
    else:
        # Create quantum layer
        quantum_layer = create_quantum_layer(config)

        # Create hybrid model
        model = HybridModel(
            classical_model=classical_model,
            quantum_layer=quantum_layer,
            num_classes=config['hybrid_model']['num_classes'],
            dropout=config['classical_model']['lstm_layers'][0]['dropout']
        )

    return model


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters in model.

    Args:
        model: PyTorch model

    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


if __name__ == "__main__":
    # Test hybrid model
    import yaml

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Create model
    model = create_hybrid_model(config, use_ensemble=False)
    print(model)
    print("\n" + "="*80 + "\n")

    # Count parameters
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test forward pass
    batch_size = 4
    num_channels = config['data']['num_channels']
    seq_len = int(config['data']['window_size'] * config['data']['sampling_rate'])

    x = torch.randn(batch_size, num_channels, seq_len)
    print(f"\nInput shape: {x.shape}")

    # Forward pass
    output = model(x)
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")

    # Test with return_intermediate
    output, classical_feat, quantum_feat = model(x, return_intermediate=True)
    print(f"\nClassical features shape: {classical_feat.shape}")
    print(f"Quantum features shape: {quantum_feat.shape}")
    print(f"Final output shape: {output.shape}")

    # Test softmax probabilities
    probs = F.softmax(output, dim=1)
    print(f"\nProbabilities sum: {probs.sum(dim=1)}")
    print(f"Probabilities:\n{probs}")
