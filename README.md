# Hybrid Quantum-Classical Neural Network for Alzheimer's Disease Classification

A hybrid quantum-classical deep learning framework for Alzheimer's disease classification using EEG data. This project combines Bidirectional LSTM with Multi-Head Attention (classical) and Variational Quantum Circuits (quantum) for enhanced feature extraction and classification.

## 🧠 Overview

This project implements a state-of-the-art hybrid quantum-classical neural network for detecting Alzheimer's disease from EEG signals. The model leverages:

- **Classical Component**: BiLSTM with Multi-Head Attention for temporal feature extraction
- **Quantum Component**: Variational Quantum Circuit (VQC) using PennyLane for quantum feature transformation
- **Robust Training**: Comprehensive error handling including NaN detection, gradient explosion prevention, and early stopping

## 🏗️ Architecture

```
EEG Signal → Preprocessing → BiLSTM → Attention → Quantum Layer → Classification
                            (Classical)            (Quantum)
```

### Classical Component (BiLSTM + Attention)
- **Input**: Multi-channel EEG time-series data
- **2 Bidirectional LSTM layers**: 
  - Layer 1: 128 hidden units
  - Layer 2: 64 hidden units
- **Multi-Head Attention**: 4 attention heads
- **Dropout**: 0.3-0.5 for regularization
- **Dense layers**: Dimensionality reduction for quantum input

### Quantum Component
- **Qubits**: 10-12 qubits
- **Encoding**: Amplitude or angle encoding
- **Circuit**: Parameterized quantum circuit with:
  - Rotation gates (RX, RY, RZ)
  - Entangling gates (CNOT)
  - 3-4 variational layers
- **Measurement**: Expectation values (Pauli-Z)

### Output
- Binary classification: Alzheimer's vs Healthy
- Supports multi-class extension
- Softmax activation for probability distribution

## 📋 Requirements

- Python 3.10+
- PyTorch 2.7.1
- PennyLane 0.39.0
- CUDA (optional, for GPU acceleration)

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/divyashhree/Qnn-for-AD-Diagnosis.git
cd Qnn-for-AD-Diagnosis
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 📊 Dataset

This project uses the [Largest Alzheimer EEG Dataset](https://www.kaggle.com/datasets/codingyodha/largest-alzheimer-eeg-dataset) from Kaggle.

### Dataset Setup

1. Download the dataset from Kaggle
2. Extract to `./data` directory
3. Organize as follows:

```
data/
├── train/
│   ├── alzheimer/
│   └── healthy/
├── val/
└── test/
```

**Note**: If data is not available, the code will automatically generate synthetic EEG-like data for testing and demonstration purposes.

## ⚙️ Configuration

All hyperparameters are defined in `config.yaml`:

```yaml
data:
  sampling_rate: 128  # Hz
  window_size: 2.0    # seconds
  num_channels: 19    # EEG channels
  bandpass_filter:
    low: 0.5
    high: 50.0

classical_model:
  lstm_layers:
    - hidden_size: 128
      dropout: 0.3
    - hidden_size: 64
      dropout: 0.3
  attention:
    num_heads: 4

quantum_layer:
  n_qubits: 10
  n_layers: 3
  encoding: "amplitude"

training:
  epochs: 100
  learning_rate: 0.001
  batch_size: 32
  gradient_clipping:
    enabled: true
    max_norm: 1.0
  early_stopping:
    enabled: true
    patience: 15
```

Customize these parameters according to your needs.

## 🏃 Usage

### Training

Train the model with default configuration:

```bash
python train.py
```

The training script will:
- Load and preprocess EEG data
- Create train/val/test splits
- Initialize the hybrid model
- Train with progress bars and logging
- Save checkpoints to `./checkpoints`
- Generate training plots in `./plots`
- Log metrics to TensorBoard in `./logs`

**Resume training** from last checkpoint:
```bash
python train.py  # Automatically detects and resumes from last checkpoint
```

### Evaluation

Evaluate the trained model on test set:

```bash
python evaluate.py
```

This will generate:
- Accuracy, Precision, Recall, F1-Score, ROC AUC
- Confusion matrix
- ROC curves
- Classification report
- Class distribution plots

### TensorBoard Monitoring

Monitor training in real-time:

```bash
tensorboard --logdir=./logs
```

Open browser at `http://localhost:6006`

## 📁 Project Structure

```
Qnn-for-AD-Diagnosis/
├── config.yaml                 # Configuration file
├── requirements.txt            # Dependencies
├── data_preprocessing.py       # Data loading and preprocessing
├── classical_model.py          # BiLSTM + Attention model
├── quantum_layer.py            # Quantum circuit implementation
├── hybrid_model.py             # Combined hybrid model
├── train.py                    # Training script
├── evaluate.py                 # Evaluation script
├── README.md                   # This file
├── checkpoints/                # Model checkpoints
├── logs/                       # TensorBoard logs
├── plots/                      # Training plots
└── predictions/                # Saved predictions
```

## 🔬 Model Details

### Data Preprocessing

1. **Loading**: Supports CSV, NPY, NPZ formats
2. **Bandpass Filtering**: 0.5-50 Hz using Butterworth filter
3. **Artifact Removal**: Threshold-based detection and interpolation
4. **Normalization**: Z-score normalization per channel
5. **Windowing**: 2-second windows with 50% overlap

### Training Features

- **Loss Function**: CrossEntropyLoss with label smoothing
- **Optimizer**: Adam with weight decay
- **Learning Rate Scheduling**: ReduceLROnPlateau
- **Gradient Clipping**: Prevents gradient explosion
- **Early Stopping**: Stops when validation loss plateaus
- **Checkpointing**: Saves best and last models
- **NaN Handling**: Automatic detection and replacement
- **GPU Support**: Automatic CUDA detection

### Error Handling

The training pipeline includes robust error handling:

1. **NaN Detection**: Monitors loss and activations
2. **Gradient Explosion**: Clips gradients and skips batches
3. **Memory Management**: Handles GPU OOM errors
4. **Dying ReLU**: Uses proper initialization
5. **Progress Tracking**: TQDM progress bars

## 📈 Results

After training, you'll find:

### In `./plots/`:
- `training_history.png`: Loss and accuracy curves
- `test_confusion_matrix.png`: Confusion matrix
- `test_roc_curve.png`: ROC curves
- `test_class_distribution.png`: Class distributions

### In `./logs/`:
- TensorBoard event files
- Classification reports

### In `./checkpoints/`:
- `best_checkpoint.pth`: Best model by validation loss
- `last_checkpoint.pth`: Most recent model

## 🔧 Troubleshooting

### Common Issues

**Issue**: CUDA out of memory
```
Solution: Reduce batch_size in config.yaml
```

**Issue**: NaN loss during training
```
Solution: 
- Reduce learning_rate
- Enable gradient clipping
- Check data for NaN values
```

**Issue**: ImportError: No module named 'pennylane'
```
Solution: pip install pennylane pennylane-lightning
```

**Issue**: Slow training
```
Solution:
- Use GPU (CUDA)
- Reduce n_qubits or n_layers
- Increase batch_size
- Reduce num_workers if CPU bottleneck
```

### Data Issues

If you don't have the Kaggle dataset:
- The code will automatically generate synthetic EEG data
- This is useful for testing the pipeline
- For real results, download the actual dataset

## 🎯 Performance Tips

1. **GPU Acceleration**: Use CUDA for 5-10x speedup
2. **Batch Size**: Larger batches stabilize training but require more memory
3. **Quantum Circuit**: Start with fewer qubits (4-6) for faster prototyping
4. **Learning Rate**: Use learning rate finder or start with 1e-3
5. **Early Stopping**: Prevents overfitting, saves time

## 🔬 Extending the Model

### Add More Classes
```yaml
# In config.yaml
hybrid_model:
  num_classes: 3  # e.g., Healthy, MCI, Alzheimer's
```

### Use Different Quantum Encoding
```yaml
quantum_layer:
  encoding: "angle"  # Options: "amplitude", "angle"
```

### Enable Ensemble
```python
# In train.py or evaluate.py
model = create_hybrid_model(config, use_ensemble=True)
```

## 📝 Citation

If you use this code, please cite:

```bibtex
@software{hybrid_qnn_alzheimer,
  title = {Hybrid Quantum-Classical Neural Network for Alzheimer's Disease Classification},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/divyashhree/Qnn-for-AD-Diagnosis}
}
```

## 📄 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📧 Contact

For questions or issues:
- Create a GitHub issue
- Contact: [Your Email]

## 🙏 Acknowledgments

- Dataset: [Largest Alzheimer EEG Dataset](https://www.kaggle.com/datasets/codingyodha/largest-alzheimer-eeg-dataset)
- Quantum Framework: [PennyLane](https://pennylane.ai/)
- Deep Learning: [PyTorch](https://pytorch.org/)

## 🔮 Future Work

- [ ] Add more quantum circuit architectures
- [ ] Implement quantum attention mechanisms
- [ ] Support for multi-modal data (EEG + MRI)
- [ ] Hyperparameter optimization with Optuna
- [ ] Model interpretability and visualization
- [ ] Real-time EEG classification
- [ ] Mobile deployment

---

**Note**: This is a research project. The model should not be used for actual medical diagnosis without proper validation and clinical trials.
