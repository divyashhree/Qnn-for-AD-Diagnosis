# Quick Start Guide

This guide will help you get started with the Hybrid Quantum-Classical Neural Network in just a few minutes.

## Prerequisites

- Python 3.10 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster training

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/divyashhree/Qnn-for-AD-Diagnosis.git
cd Qnn-for-AD-Diagnosis
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- PyTorch 2.7.1
- PennyLane 0.39.0 (quantum computing framework)
- NumPy, SciPy, Pandas
- scikit-learn, matplotlib, seaborn
- and other required packages

## Quick Demo

Run a quick demo to verify everything works:

```bash
python demo.py
```

This will:
1. Load configuration
2. Create dummy EEG data (if real data not available)
3. Build the hybrid model
4. Train for 3 epochs
5. Evaluate and show results

## Training Your Model

### With Default Settings

```bash
python train.py
```

This will train for 100 epochs with the default configuration.

### Custom Configuration

Edit `config.yaml` to customize:

```yaml
training:
  epochs: 50              # Number of training epochs
  learning_rate: 0.001    # Initial learning rate
  batch_size: 32          # Batch size

quantum_layer:
  n_qubits: 10            # Number of qubits
  n_layers: 3             # Quantum circuit depth
  encoding: "amplitude"   # Encoding method
```

Then run:

```bash
python train.py
```

## Monitoring Training

### TensorBoard

Start TensorBoard to monitor training in real-time:

```bash
tensorboard --logdir=./logs
```

Open your browser at `http://localhost:6006`

### Training Progress

The training script will show:
- Progress bars for each epoch
- Loss and accuracy metrics
- Learning rate updates
- Checkpoint saves

Example output:
```
Epoch [1/100] Train Loss: 0.6931, Train Acc: 50.00% | Val Loss: 0.6925, Val Acc: 51.23% | LR: 0.001000
```

## Evaluating Your Model

After training, evaluate on the test set:

```bash
python evaluate.py
```

This will generate:
- Classification metrics (accuracy, precision, recall, F1-score)
- Confusion matrix
- ROC curves
- Classification report

## Using Your Own Data

### Data Format

Your EEG data should be in one of these formats:
- NPZ file with 'data' and 'labels' arrays
- NPY file
- CSV file (last column = labels)

### Data Shape

- **Data**: `(n_samples, n_channels, n_timepoints)`
  - `n_samples`: Number of recordings
  - `n_channels`: Number of EEG channels (default: 19)
  - `n_timepoints`: Number of time points

- **Labels**: `(n_samples,)`
  - 0 = Healthy
  - 1 = Alzheimer's

### Update Configuration

```yaml
data:
  dataset_path: "./data/your_dataset.npz"
  num_channels: 19
  sampling_rate: 128
```

### Kaggle Dataset

For the recommended dataset:

1. Download from [Kaggle](https://www.kaggle.com/datasets/codingyodha/largest-alzheimer-eeg-dataset)
2. Extract to `./data/` directory
3. Update `config.yaml` with the correct path

## Common Commands

### Train from scratch
```bash
python train.py
```

### Resume training
```bash
# Automatically resumes from last checkpoint if it exists
python train.py
```

### Evaluate best model
```bash
python evaluate.py
```

### Run demo
```bash
python demo.py
```

### View logs
```bash
tensorboard --logdir=./logs
```

## Troubleshooting

### Out of Memory (GPU)

Reduce batch size in `config.yaml`:
```yaml
data:
  batch_size: 16  # or 8, 4
```

### Slow Training

- Use GPU (CUDA) if available
- Reduce number of qubits: `n_qubits: 6`
- Reduce quantum layers: `n_layers: 2`
- Increase batch size if memory allows

### NaN Loss

- Reduce learning rate: `learning_rate: 0.0001`
- Enable gradient clipping (already enabled by default)
- Check data for NaN values

### Import Errors

```bash
pip install -r requirements.txt --upgrade
```

## Next Steps

1. **Read the full README.md** for detailed architecture explanation
2. **Customize hyperparameters** in `config.yaml`
3. **Experiment with quantum circuits** (qubits, layers, encoding)
4. **Try ensemble mode** in `hybrid_model.py`
5. **Add your own features** to the preprocessing pipeline

## Getting Help

- Check the [README.md](README.md) for detailed documentation
- Review error messages and logs
- Open an issue on GitHub

## Performance Tips

1. **Start small**: Use 4-6 qubits for quick prototyping
2. **Use GPU**: 5-10x faster training
3. **Batch size**: Larger batches = more stable gradients
4. **Learning rate**: Start with 1e-3, reduce if unstable
5. **Early stopping**: Prevents overfitting, saves time

## Example Workflow

```bash
# 1. Setup
git clone https://github.com/divyashhree/Qnn-for-AD-Diagnosis.git
cd Qnn-for-AD-Diagnosis
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Quick test
python demo.py

# 3. Customize config
nano config.yaml  # or your preferred editor

# 4. Train
python train.py

# 5. Monitor (in another terminal)
tensorboard --logdir=./logs

# 6. Evaluate
python evaluate.py
```

That's it! You're ready to use the hybrid quantum-classical neural network. 🚀
