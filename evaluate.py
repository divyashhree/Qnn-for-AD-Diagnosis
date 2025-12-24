"""
Evaluation script for Hybrid Quantum-Classical Neural Network.

This script evaluates the trained model and generates comprehensive metrics,
including accuracy, precision, recall, F1-score, and confusion matrix.
"""

import os
import yaml
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    roc_auc_score
)
from tqdm import tqdm

from data_preprocessing import create_dataloaders
from hybrid_model import create_hybrid_model

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluator class for hybrid quantum-classical model."""

    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        device: torch.device
    ):
        """
        Initialize evaluator.

        Args:
            model: Trained model
            config: Configuration dictionary
            device: Device to run evaluation on
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.model.eval()

    def predict(
        self,
        data_loader: torch.utils.data.DataLoader
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predictions for data loader.

        Args:
            data_loader: Data loader

        Returns:
            Tuple of (predictions, true_labels, probabilities)
        """
        all_predictions = []
        all_labels = []
        all_probabilities = []

        with torch.no_grad():
            for data, target in tqdm(data_loader, desc="Predicting"):
                data = data.to(self.device)

                try:
                    output = self.model(data)

                    # Get probabilities
                    if self.config['hybrid_model']['num_classes'] == 2:
                        probabilities = torch.softmax(output, dim=1)
                    else:
                        probabilities = torch.softmax(output, dim=1)

                    # Get predictions
                    _, predicted = torch.max(output.data, 1)

                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(target.numpy())
                    all_probabilities.extend(probabilities.cpu().numpy())

                except Exception as e:
                    logger.error(f"Error during prediction: {e}")
                    continue

        return (
            np.array(all_predictions),
            np.array(all_labels),
            np.array(all_probabilities)
        )

    def calculate_metrics(
        self,
        predictions: np.ndarray,
        true_labels: np.ndarray,
        probabilities: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate evaluation metrics.

        Args:
            predictions: Predicted labels
            true_labels: True labels
            probabilities: Prediction probabilities

        Returns:
            Dictionary of metrics
        """
        num_classes = self.config['hybrid_model']['num_classes']

        metrics = {}

        # Accuracy
        metrics['accuracy'] = accuracy_score(true_labels, predictions)

        # Precision, Recall, F1-score
        if num_classes == 2:
            metrics['precision'] = precision_score(true_labels, predictions, average='binary')
            metrics['recall'] = recall_score(true_labels, predictions, average='binary')
            metrics['f1_score'] = f1_score(true_labels, predictions, average='binary')

            # ROC AUC
            try:
                metrics['roc_auc'] = roc_auc_score(true_labels, probabilities[:, 1])
            except Exception as e:
                logger.warning(f"Could not calculate ROC AUC: {e}")
                metrics['roc_auc'] = 0.0
        else:
            metrics['precision'] = precision_score(true_labels, predictions, average='macro')
            metrics['recall'] = recall_score(true_labels, predictions, average='macro')
            metrics['f1_score'] = f1_score(true_labels, predictions, average='macro')

            # ROC AUC for multi-class
            try:
                metrics['roc_auc'] = roc_auc_score(
                    true_labels,
                    probabilities,
                    multi_class='ovr',
                    average='macro'
                )
            except Exception as e:
                logger.warning(f"Could not calculate ROC AUC: {e}")
                metrics['roc_auc'] = 0.0

        return metrics

    def plot_confusion_matrix(
        self,
        predictions: np.ndarray,
        true_labels: np.ndarray,
        save_path: str,
        class_names: List[str] = None
    ):
        """
        Plot and save confusion matrix.

        Args:
            predictions: Predicted labels
            true_labels: True labels
            save_path: Path to save plot
            class_names: Names of classes
        """
        cm = confusion_matrix(true_labels, predictions)

        if class_names is None:
            num_classes = self.config['hybrid_model']['num_classes']
            if num_classes == 2:
                class_names = ['Healthy', "Alzheimer's"]
            else:
                class_names = [f'Class {i}' for i in range(num_classes)]

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix', fontsize=16, pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved confusion matrix to {save_path}")

    def plot_roc_curve(
        self,
        true_labels: np.ndarray,
        probabilities: np.ndarray,
        save_path: str
    ):
        """
        Plot and save ROC curve.

        Args:
            true_labels: True labels
            probabilities: Prediction probabilities
            save_path: Path to save plot
        """
        num_classes = self.config['hybrid_model']['num_classes']

        plt.figure(figsize=(10, 8))

        if num_classes == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(true_labels, probabilities[:, 1])
            roc_auc = auc(fpr, tpr)

            plt.plot(
                fpr, tpr,
                color='darkorange',
                lw=2,
                label=f'ROC curve (AUC = {roc_auc:.2f})'
            )
        else:
            # Multi-class classification
            for i in range(num_classes):
                binary_labels = (true_labels == i).astype(int)
                fpr, tpr, _ = roc_curve(binary_labels, probabilities[:, i])
                roc_auc = auc(fpr, tpr)

                plt.plot(
                    fpr, tpr,
                    lw=2,
                    label=f'Class {i} (AUC = {roc_auc:.2f})'
                )

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16, pad=20)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved ROC curve to {save_path}")

    def plot_class_distribution(
        self,
        predictions: np.ndarray,
        true_labels: np.ndarray,
        save_path: str
    ):
        """
        Plot class distribution comparison.

        Args:
            predictions: Predicted labels
            true_labels: True labels
            save_path: Path to save plot
        """
        num_classes = self.config['hybrid_model']['num_classes']

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # True labels distribution
        unique_true, counts_true = np.unique(true_labels, return_counts=True)
        axes[0].bar(unique_true, counts_true, color='steelblue', alpha=0.7)
        axes[0].set_xlabel('Class', fontsize=12)
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].set_title('True Label Distribution', fontsize=14)
        axes[0].grid(True, alpha=0.3)

        # Predicted labels distribution
        unique_pred, counts_pred = np.unique(predictions, return_counts=True)
        axes[1].bar(unique_pred, counts_pred, color='coral', alpha=0.7)
        axes[1].set_xlabel('Class', fontsize=12)
        axes[1].set_ylabel('Count', fontsize=12)
        axes[1].set_title('Predicted Label Distribution', fontsize=14)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved class distribution to {save_path}")

    def generate_classification_report(
        self,
        predictions: np.ndarray,
        true_labels: np.ndarray,
        save_path: str,
        class_names: List[str] = None
    ):
        """
        Generate and save classification report.

        Args:
            predictions: Predicted labels
            true_labels: True labels
            save_path: Path to save report
            class_names: Names of classes
        """
        if class_names is None:
            num_classes = self.config['hybrid_model']['num_classes']
            if num_classes == 2:
                class_names = ['Healthy', "Alzheimer's"]
            else:
                class_names = [f'Class {i}' for i in range(num_classes)]

        report = classification_report(
            true_labels,
            predictions,
            target_names=class_names,
            digits=4
        )

        with open(save_path, 'w') as f:
            f.write("Classification Report\n")
            f.write("=" * 80 + "\n\n")
            f.write(report)

        logger.info(f"Saved classification report to {save_path}")

        # Also print to console
        print("\n" + "=" * 80)
        print("Classification Report")
        print("=" * 80 + "\n")
        print(report)

    def save_predictions(
        self,
        predictions: np.ndarray,
        true_labels: np.ndarray,
        probabilities: np.ndarray,
        save_path: str
    ):
        """
        Save predictions to file.

        Args:
            predictions: Predicted labels
            true_labels: True labels
            probabilities: Prediction probabilities
            save_path: Path to save predictions
        """
        results = {
            'predictions': predictions,
            'true_labels': true_labels,
            'probabilities': probabilities
        }

        np.savez(save_path, **results)
        logger.info(f"Saved predictions to {save_path}")

    def evaluate(
        self,
        data_loader: torch.utils.data.DataLoader,
        split_name: str = 'test'
    ) -> Dict[str, float]:
        """
        Evaluate model on data loader.

        Args:
            data_loader: Data loader
            split_name: Name of data split ('train', 'val', 'test')

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating on {split_name} set...")

        # Generate predictions
        predictions, true_labels, probabilities = self.predict(data_loader)

        # Calculate metrics
        metrics = self.calculate_metrics(predictions, true_labels, probabilities)

        # Log metrics
        logger.info(f"\n{split_name.upper()} Set Metrics:")
        logger.info("=" * 80)
        for metric_name, value in metrics.items():
            logger.info(f"{metric_name}: {value:.4f}")
        logger.info("=" * 80)

        # Generate visualizations if save_plots is enabled
        if self.config['evaluation']['confusion_matrix']:
            plot_dir = self.config['logging']['plot_dir']
            os.makedirs(plot_dir, exist_ok=True)

            # Confusion matrix
            cm_path = os.path.join(plot_dir, f'{split_name}_confusion_matrix.png')
            self.plot_confusion_matrix(predictions, true_labels, cm_path)

            # ROC curve
            roc_path = os.path.join(plot_dir, f'{split_name}_roc_curve.png')
            self.plot_roc_curve(true_labels, probabilities, roc_path)

            # Class distribution
            dist_path = os.path.join(plot_dir, f'{split_name}_class_distribution.png')
            self.plot_class_distribution(predictions, true_labels, dist_path)

        # Generate classification report
        if self.config['evaluation']['classification_report']:
            report_path = os.path.join(
                self.config['logging']['log_dir'],
                f'{split_name}_classification_report.txt'
            )
            os.makedirs(self.config['logging']['log_dir'], exist_ok=True)
            self.generate_classification_report(predictions, true_labels, report_path)

        # Save predictions
        if self.config['evaluation']['save_predictions']:
            pred_dir = self.config['evaluation']['predictions_dir']
            os.makedirs(pred_dir, exist_ok=True)
            pred_path = os.path.join(pred_dir, f'{split_name}_predictions.npz')
            self.save_predictions(predictions, true_labels, probabilities, pred_path)

        return metrics


def main():
    """Main evaluation function."""
    # Load configuration
    config_path = 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup device
    device_config = config['training']['device']
    if device_config == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_config)

    logger.info(f"Using device: {device}")

    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, test_loader = create_dataloaders(config, seed=config['seed'])

    # Create model
    logger.info("Creating model...")
    model = create_hybrid_model(config, use_ensemble=False)

    # Load checkpoint
    checkpoint_dir = config['training']['checkpoint']['checkpoint_dir']
    best_checkpoint = os.path.join(checkpoint_dir, 'best_checkpoint.pth')

    if not os.path.exists(best_checkpoint):
        logger.error(f"Checkpoint not found: {best_checkpoint}")
        logger.info("Please train the model first using train.py")
        return

    logger.info(f"Loading checkpoint from {best_checkpoint}")
    checkpoint = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Create evaluator
    evaluator = ModelEvaluator(model, config, device)

    # Evaluate on all splits
    test_metrics = evaluator.evaluate(test_loader, split_name='test')
    val_metrics = evaluator.evaluate(val_loader, split_name='val')

    # Summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"\nValidation Set:")
    print(f"  Accuracy:  {val_metrics['accuracy']:.4f}")
    print(f"  Precision: {val_metrics['precision']:.4f}")
    print(f"  Recall:    {val_metrics['recall']:.4f}")
    print(f"  F1-Score:  {val_metrics['f1_score']:.4f}")
    print(f"  ROC AUC:   {val_metrics['roc_auc']:.4f}")

    print(f"\nTest Set:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1-Score:  {test_metrics['f1_score']:.4f}")
    print(f"  ROC AUC:   {test_metrics['roc_auc']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
