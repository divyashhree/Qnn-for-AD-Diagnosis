"""
Quantum layer implementation using PennyLane.

This module implements a Variational Quantum Circuit (VQC) layer for integration
with classical neural networks in a hybrid quantum-classical model.
"""

from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
import logging

logger = logging.getLogger(__name__)


class QuantumLayer(nn.Module):
    """
    Variational Quantum Circuit (VQC) layer using PennyLane.

    This layer encodes classical data into quantum states and performs
    parameterized quantum operations to extract quantum features.
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        input_dim: int,
        output_dim: int,
        encoding: str = "amplitude",
        diff_method: str = "best",
        device_name: str = "default.qubit"
    ):
        """
        Initialize Quantum Layer.

        Args:
            n_qubits: Number of qubits in the quantum circuit
            n_layers: Number of variational layers
            input_dim: Dimension of classical input
            output_dim: Dimension of quantum output (number of measurements)
            encoding: Encoding method ("amplitude" or "angle")
            diff_method: Differentiation method for gradients
            device_name: PennyLane device name
        """
        super(QuantumLayer, self).__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.encoding = encoding
        self.diff_method = diff_method

        # Create quantum device
        self.dev = qml.device(device_name, wires=n_qubits)

        # Create quantum circuit
        self.qnode = qml.QNode(
            self._quantum_circuit,
            self.dev,
            interface="torch",
            diff_method=diff_method
        )

        # Calculate number of parameters needed
        # Each layer has: 3 rotations per qubit + entangling gates
        self.n_params = n_qubits * 3 * n_layers

        # Initialize quantum parameters
        self.q_params = nn.Parameter(
            torch.randn(self.n_params) * 0.1
        )

        # If input_dim doesn't match encoding requirements, add linear projection
        if encoding == "amplitude":
            # Amplitude encoding requires 2^n_qubits dimensions
            required_dim = 2 ** n_qubits
            if input_dim != required_dim:
                self.input_projection = nn.Linear(input_dim, required_dim)
            else:
                self.input_projection = None
        elif encoding == "angle":
            # Angle encoding requires n_qubits dimensions
            if input_dim != n_qubits:
                self.input_projection = nn.Linear(input_dim, n_qubits)
            else:
                self.input_projection = None
        else:
            raise ValueError(f"Unknown encoding method: {encoding}")

        # Output projection to match desired output dimension
        self.output_projection = nn.Linear(n_qubits, output_dim)

        logger.info(f"Initialized QuantumLayer with {n_qubits} qubits, "
                   f"{n_layers} layers, {self.n_params} parameters")

    def _amplitude_encoding(self, inputs: torch.Tensor) -> None:
        """
        Encode classical data using amplitude encoding.

        Args:
            inputs: Classical input tensor (must have 2^n_qubits dimensions)
        """
        # Normalize inputs to create valid quantum state
        norm = torch.sqrt(torch.sum(inputs ** 2))
        if norm > 1e-8:
            inputs = inputs / norm
        else:
            inputs = torch.zeros_like(inputs)
            inputs[0] = 1.0

        # Amplitude encoding
        qml.AmplitudeEmbedding(
            features=inputs,
            wires=range(self.n_qubits),
            normalize=True,
            pad_with=0.0
        )

    def _angle_encoding(self, inputs: torch.Tensor) -> None:
        """
        Encode classical data using angle encoding.

        Args:
            inputs: Classical input tensor (must have n_qubits dimensions)
        """
        # Angle encoding: each input feature controls rotation of one qubit
        qml.AngleEmbedding(
            features=inputs,
            wires=range(self.n_qubits),
            rotation='Y'
        )

    def _variational_layer(self, params: torch.Tensor, layer_idx: int) -> None:
        """
        Apply one layer of variational quantum circuit.

        Args:
            params: Quantum parameters
            layer_idx: Index of current layer
        """
        # Parameterized rotations for each qubit
        param_offset = layer_idx * self.n_qubits * 3

        for i in range(self.n_qubits):
            idx = param_offset + i * 3
            qml.RX(params[idx], wires=i)
            qml.RY(params[idx + 1], wires=i)
            qml.RZ(params[idx + 2], wires=i)

        # Entangling layer using CNOT gates
        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])

        # Connect last qubit to first for circular entanglement
        if self.n_qubits > 1:
            qml.CNOT(wires=[self.n_qubits - 1, 0])

    def _quantum_circuit(
        self,
        inputs: torch.Tensor,
        params: torch.Tensor
    ) -> List[float]:
        """
        Define the quantum circuit.

        Args:
            inputs: Classical input data
            params: Quantum circuit parameters

        Returns:
            List of expectation values from measurements
        """
        # Data encoding
        if self.encoding == "amplitude":
            self._amplitude_encoding(inputs)
        elif self.encoding == "angle":
            self._angle_encoding(inputs)

        # Variational layers
        for layer in range(self.n_layers):
            self._variational_layer(params, layer)

        # Measurements: expectation value of Pauli-Z for each qubit
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through quantum layer.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        batch_size = x.shape[0]

        # Check for NaN values in input
        if torch.isnan(x).any():
            logger.warning("NaN values detected in quantum layer input, replacing with zeros")
            x = torch.nan_to_num(x, nan=0.0)

        # Clamp extreme values to prevent numerical instability
        x = torch.clamp(x, -10.0, 10.0)

        # Project input if necessary
        if self.input_projection is not None:
            x = self.input_projection(x)

        # Process each sample in the batch
        outputs = []
        for i in range(batch_size):
            try:
                # Run quantum circuit
                measurement = self.qnode(x[i], self.q_params)

                # Convert to tensor
                if isinstance(measurement, list):
                    measurement = torch.stack([
                        m if isinstance(m, torch.Tensor) else torch.tensor(m)
                        for m in measurement
                    ])
                elif not isinstance(measurement, torch.Tensor):
                    measurement = torch.tensor(measurement)

                outputs.append(measurement)

            except Exception as e:
                logger.error(f"Error in quantum circuit execution: {e}")
                # Return zeros in case of error
                outputs.append(torch.zeros(self.n_qubits))

        # Stack outputs
        output = torch.stack(outputs)

        # Ensure correct dtype (float32)
        if output.dtype != torch.float32:
            output = output.float()

        # Check for NaN in quantum output
        if torch.isnan(output).any():
            logger.warning("NaN detected in quantum circuit output, replacing with zeros")
            output = torch.nan_to_num(output, nan=0.0)

        # Project to desired output dimension
        output = self.output_projection(output)

        # Final NaN check
        if torch.isnan(output).any():
            logger.warning("NaN detected in quantum layer output, replacing with zeros")
            output = torch.nan_to_num(output, nan=0.0)

        return output


class HybridQuantumLayer(nn.Module):
    """
    Hybrid quantum layer with classical pre/post-processing.

    This wrapper adds classical layers before and after the quantum circuit
    for better integration and performance.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_qubits: int = 10,
        n_layers: int = 3,
        encoding: str = "amplitude",
        use_classical_preprocessing: bool = True
    ):
        """
        Initialize Hybrid Quantum Layer.

        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output features
            n_qubits: Number of qubits
            n_layers: Number of variational layers
            encoding: Encoding method
            use_classical_preprocessing: Whether to use classical preprocessing
        """
        super(HybridQuantumLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_classical_preprocessing = use_classical_preprocessing

        # Classical preprocessing
        if use_classical_preprocessing:
            self.pre_net = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 16)
            )
            quantum_input_dim = 16
        else:
            self.pre_net = None
            quantum_input_dim = input_dim

        # Quantum layer
        self.quantum_layer = QuantumLayer(
            n_qubits=n_qubits,
            n_layers=n_layers,
            input_dim=quantum_input_dim,
            output_dim=output_dim,
            encoding=encoding
        )

        # Classical post-processing
        self.post_net = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through hybrid quantum layer.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        # Classical preprocessing
        if self.pre_net is not None:
            x = self.pre_net(x)

        # Check for NaN
        if torch.isnan(x).any():
            logger.warning("NaN detected before quantum layer, replacing with zeros")
            x = torch.nan_to_num(x, nan=0.0)

        # Quantum processing
        x = self.quantum_layer(x)

        # Classical post-processing
        x = self.post_net(x)

        # Final check
        if torch.isnan(x).any():
            logger.warning("NaN detected after quantum post-processing, replacing with zeros")
            x = torch.nan_to_num(x, nan=0.0)

        return x


def create_quantum_layer(config: dict) -> HybridQuantumLayer:
    """
    Create quantum layer from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        HybridQuantumLayer instance
    """
    quantum_config = config['quantum_layer']
    classical_config = config['classical_model']

    layer = HybridQuantumLayer(
        input_dim=classical_config['dense_output'],
        output_dim=config['hybrid_model']['num_classes'],
        n_qubits=quantum_config['n_qubits'],
        n_layers=quantum_config['n_layers'],
        encoding=quantum_config['encoding'],
        use_classical_preprocessing=False  # Already preprocessed by classical model
    )

    return layer


if __name__ == "__main__":
    # Test quantum layer
    import yaml

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    quantum_layer = create_quantum_layer(config)
    print(quantum_layer)

    # Test forward pass
    batch_size = 4
    input_dim = config['classical_model']['dense_output']

    x = torch.randn(batch_size, input_dim)
    output = quantum_layer(x)

    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")

    # Count parameters
    total_params = sum(p.numel() for p in quantum_layer.parameters())
    trainable_params = sum(p.numel() for p in quantum_layer.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
