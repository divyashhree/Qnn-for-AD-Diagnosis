"""
Classical neural network model with BiLSTM and Multi-Head Attention.

This module implements the classical component of the hybrid quantum-classical model
for EEG-based Alzheimer's disease classification.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism for sequence data.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        """
        Initialize Multi-Head Attention.

        Args:
            embed_dim: Dimension of input embeddings
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super(MultiHeadAttention, self).__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear projections
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of Multi-Head Attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            mask: Optional attention mask

        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, embed_dim = x.size()

        # Linear projections and reshape for multi-head attention
        Q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, V)

        # Reshape and apply output projection
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        output = self.out_linear(context)

        return output, attention_weights


class BiLSTMWithAttention(nn.Module):
    """
    Bidirectional LSTM with Multi-Head Attention for EEG sequence processing.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list,
        num_attention_heads: int = 4,
        dropout: float = 0.3,
        output_dim: int = 16
    ):
        """
        Initialize BiLSTM with Attention model.

        Args:
            input_size: Number of input features (EEG channels)
            hidden_sizes: List of hidden sizes for each LSTM layer
            num_attention_heads: Number of attention heads
            dropout: Dropout probability
            output_dim: Output dimension for quantum layer input
        """
        super(BiLSTMWithAttention, self).__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.output_dim = output_dim

        # Create LSTM layers
        self.lstm_layers = nn.ModuleList()
        current_input_size = input_size

        for i, hidden_size in enumerate(hidden_sizes):
            lstm = nn.LSTM(
                input_size=current_input_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
                dropout=0.0  # We'll use separate dropout layers
            )
            self.lstm_layers.append(lstm)
            current_input_size = hidden_size * 2  # *2 for bidirectional

        # Dropout layers
        self.dropout_layers = nn.ModuleList([
            nn.Dropout(dropout) for _ in range(len(hidden_sizes))
        ])
        self.batch_norm_layers = nn.ModuleList([
            nn.BatchNorm1d(hidden_size * 2)  # *2 for bidirectional
            for hidden_size in hidden_sizes
        ])

        # Multi-Head Attention
        attention_dim = hidden_sizes[-1] * 2  # *2 for bidirectional
        self.attention = MultiHeadAttention(
            embed_dim=attention_dim,
            num_heads=num_attention_heads,
            dropout=dropout
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(attention_dim)

        # Dense layers for dimensionality reduction
        self.fc1 = nn.Linear(attention_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)
        self.dropout_fc = nn.Dropout(dropout)

        # Activation function
        self.relu = nn.ReLU()

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights using Xavier initialization."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data, gain=0.5)  # Add gain < 1
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data, gain=0.5)  # Add gain < 1
            elif 'bias' in name:
                param.data.fill_(0)
            # Add forget gate bias = 1 for LSTM stability
            if 'bias_ih' in name or 'bias_hh' in name:
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)  # Forget gate bias
            elif 'weight' in name and isinstance(self._get_module_from_name(name), nn.Linear):
                nn.init.xavier_uniform_(param.data, gain=0.5)

    def _get_module_from_name(self, name: str):
        """Helper to get module from parameter name."""
        parts = name.split('.')
        module = self
        for part in parts[:-1]:
            module = getattr(module, part)
        return module

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through BiLSTM with Attention.

        Args:
            x: Input tensor of shape (batch_size, num_channels, seq_len)
            return_attention: Whether to return attention weights

        Returns:
            Output tensor of shape (batch_size, output_dim)
            If return_attention=True, returns tuple (output, attention_weights)
        """
        # Input shape: (batch_size, num_channels, seq_len)
        # Transpose to (batch_size, seq_len, num_channels) for LSTM
        x = x.transpose(1, 2)
        x = F.layer_norm(x, x.shape[-1:])
        x = torch.clamp(x, -5.0, -5.0)  # Initial clamping to avoid extreme values

        # Check for NaN values
        if torch.isnan(x).any():
            logger.warning("NaN values detected in input, replacing with zeros")
            x = torch.nan_to_num(x, nan=0.0)

        # Pass through LSTM layers
        for i, (lstm, dropout,bn) in enumerate(zip(self.lstm_layers, self.dropout_layers, self.batch_norm_layers)):
            x, (h_n, c_n) = lstm(x)
            x = x.transpose(1,2)
            x = bn(x)
            x = x.transpose(1,2)

            # Apply dropout
            x = dropout(x)

            # Check for NaN or Inf after LSTM
            if torch.isnan(x).any() or torch.isinf(x).any():
                logger.warning(f"NaN/Inf detected after LSTM layer {i}, clipping values")
                x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
                x = torch.clamp(x, -1e6, 1e6)

        # Apply Multi-Head Attention
        # x shape: (batch_size, seq_len, hidden_dim*2)
        attended, attention_weights = self.attention(x)

        # Add residual connection and layer normalization
        x = self.layer_norm(x + attended)

        # Global average pooling over time dimension
        x = torch.mean(x, dim=1)  # (batch_size, hidden_dim*2)

        # Check for NaN after attention
        if torch.isnan(x).any():
            logger.warning("NaN detected after attention, replacing with zeros")
            x = torch.nan_to_num(x, nan=0.0)

        # Dense layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout_fc(x)

        # Check for NaN after FC1
        if torch.isnan(x).any():
            logger.warning("NaN detected after FC1, replacing with zeros")
            x = torch.nan_to_num(x, nan=0.0)

        x = self.fc2(x)

        # Final NaN check
        if torch.isnan(x).any():
            logger.warning("NaN detected in output, replacing with zeros")
            x = torch.nan_to_num(x, nan=0.0)

        if return_attention:
            return x, attention_weights

        return x


def create_classical_model(config: dict) -> BiLSTMWithAttention:
    """
    Create classical model from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        BiLSTMWithAttention model instance
    """
    model_config = config['classical_model']

    # Extract hidden sizes from layer configuration
    hidden_sizes = [layer['hidden_size'] for layer in model_config['lstm_layers']]

    # Get dropout from first layer (assuming same for all)
    dropout = model_config['lstm_layers'][0]['dropout']

    model = BiLSTMWithAttention(
        input_size=model_config['input_size'],
        hidden_sizes=hidden_sizes,
        num_attention_heads=model_config['attention']['num_heads'],
        dropout=dropout,
        output_dim=model_config['dense_output']
    )

    return model


if __name__ == "__main__":
    # Test the classical model
    import yaml

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    model = create_classical_model(config)
    print(model)

    # Test forward pass
    batch_size = 4
    num_channels = config['data']['num_channels']
    seq_len = int(config['data']['window_size'] * config['data']['sampling_rate'])

    x = torch.randn(batch_size, num_channels, seq_len)
    output = model(x)

    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
