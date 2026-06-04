"""
Visualization helpers for ECG analysis and model evaluation
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import pandas as pd


# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_ecg_signal(signal: np.ndarray,
                    sampling_rate: int = 360,
                    title: str = "ECG Signal",
                    ax: Optional[plt.Axes] = None,
                    figsize: Tuple[int, int] = (12, 4),
                    show_peaks: Optional[np.ndarray] = None,
                    save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot an ECG signal with optional R-peak markers
    
    Args:
        signal: ECG signal array
        sampling_rate: Sampling rate in Hz
        title: Plot title
        ax: Matplotlib axes (optional)
        figsize: Figure size
        show_peaks: Array of peak indices to mark
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
        
    # Create time axis
    duration = len(signal) / sampling_rate
    time = np.linspace(0, duration, len(signal))
    
    # Plot signal
    ax.plot(time, signal, 'b-', linewidth=1.5, label='ECG')
    
    # Mark peaks if provided
    if show_peaks is not None:
        peak_times = show_peaks / sampling_rate
        peak_values = signal[show_peaks]
        ax.scatter(peak_times, peak_values, c='red', s=50, 
                  zorder=5, label='R-peaks', marker='^')
        
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Amplitude (mV)')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    return fig


def plot_beat_comparison(original_beat: np.ndarray,
                         reconstructed_beat: np.ndarray,
                         sampling_rate: int = 360,
                         title: str = "Beat Reconstruction",
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Compare original and reconstructed ECG beats
    
    Args:
        original_beat: Original ECG beat
        reconstructed_beat: Reconstructed ECG beat
        sampling_rate: Sampling rate in Hz
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
    
    duration = len(original_beat) / sampling_rate
    time = np.linspace(0, duration, len(original_beat))
    
    # Original beat
    ax1.plot(time, original_beat, 'b-', linewidth=1.5, label='Original')
    ax1.set_ylabel('Amplitude (mV)')
    ax1.set_title('Original Beat')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Reconstructed beat
    ax2.plot(time, reconstructed_beat, 'r-', linewidth=1.5, label='Reconstructed')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Amplitude (mV)')
    ax2.set_title('Reconstructed Beat')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Difference
    diff = original_beat - reconstructed_beat
    mse = np.mean(diff ** 2)
    fig.suptitle(f"{title} (MSE: {mse:.6f})", fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    return fig


def plot_confusion_matrix(confusion_matrix: np.ndarray,
                          class_names: List[str],
                          title: str = "Confusion Matrix",
                          normalize: bool = True,
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot confusion matrix as heatmap
    
    Args:
        confusion_matrix: Confusion matrix array
        class_names: List of class names
        title: Plot title
        normalize: Normalize to percentages
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if normalize:
        cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        cm_display = cm_normalized * 100
        fmt = '.1f'
        cbar_label = 'Percentage (%)'
    else:
        cm_display = confusion_matrix
        fmt = 'd'
        cbar_label = 'Count'
        
    sns.heatmap(cm_display, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar_kws={'label': cbar_label})
    
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    return fig


def plot_training_history(history: Dict[str, List[float]],
                          metrics: List[str] = ['loss', 'accuracy'],
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot training history curves
    
    Args:
        history: Dictionary with training history
        metrics: List of metrics to plot
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4))
    
    if n_metrics == 1:
        axes = [axes]
        
    for ax, metric in zip(axes, metrics):
        train_key = f'train_{metric}'
        val_key = f'val_{metric}'
        
        if train_key in history:
            ax.plot(history[train_key], 'b-', label='Train', linewidth=2)
        if val_key in history:
            ax.plot(history[val_key], 'r-', label='Validation', linewidth=2)
            
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} over Training')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    return fig


def plot_feature_importance(feature_names: List[str],
                           importance_scores: np.ndarray,
                           top_k: int = 20,
                           title: str = "Feature Importance",
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot feature importance bar chart
    
    Args:
        feature_names: List of feature names
        importance_scores: Array of importance scores
        top_k: Number of top features to show
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    # Sort by importance
    indices = np.argsort(importance_scores)[-top_k:]
    
    fig, ax = plt.subplots(figsize=(10, max(6, top_k * 0.3)))
    
    y_pos = np.arange(len(indices))
    ax.barh(y_pos, importance_scores[indices])
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Importance Score')
    ax.set_title(title)
    ax.invert_yaxis()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    return fig


def plot_reconstruction_errors(errors: np.ndarray,
                               threshold: Optional[float] = None,
                               title: str = "Reconstruction Errors",
                               save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot histogram of reconstruction errors with anomaly threshold
    
    Args:
        errors: Array of reconstruction errors
        threshold: Anomaly threshold line
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histogram
    ax1.hist(errors, bins=50, alpha=0.7, color='blue', edgecolor='black')
    if threshold:
        ax1.axvline(threshold, color='red', linestyle='--', linewidth=2, 
                   label=f'Threshold: {threshold:.4f}')
    ax1.set_xlabel('Reconstruction Error')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Error Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    bp = ax2.boxplot(errors, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    if threshold:
        ax2.axhline(threshold, color='red', linestyle='--', linewidth=2)
    ax2.set_ylabel('Reconstruction Error')
    ax2.set_title('Error Box Plot')
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    return fig


def plot_class_distribution(labels: np.ndarray,
                           class_names: List[str],
                           title: str = "Class Distribution",
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot class distribution bar chart
    
    Args:
        labels: Array of class labels
        class_names: List of class names
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    unique, counts = np.unique(labels, return_counts=True)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique)))
    
    bars = ax.bar(range(len(unique)), counts, color=colors, edgecolor='black')
    ax.set_xticks(range(len(unique)))
    ax.set_xticklabels([class_names[i] for i in unique], rotation=45, ha='right')
    ax.set_ylabel('Count')
    ax.set_title(title)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               str(count), ha='center', va='bottom')
        
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    return fig


def plot_roc_curves(fpr_dict: Dict[str, np.ndarray],
                   tpr_dict: Dict[str, np.ndarray],
                   auc_dict: Dict[str, float],
                   title: str = "ROC Curves",
                   save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot ROC curves for multiple classes
    
    Args:
        fpr_dict: Dictionary of FPR arrays per class
        tpr_dict: Dictionary of TPR arrays per class
        auc_dict: Dictionary of AUC scores per class
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = sns.color_palette("husl", len(fpr_dict))
    
    for (class_name, fpr), (_, tpr), (_, auc), color in zip(
        fpr_dict.items(), tpr_dict.items(), auc_dict.items(), colors
    ):
        ax.plot(fpr, tpr, linewidth=2, color=color,
               label=f'{class_name} (AUC = {auc:.3f})')
        
    # Diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    return fig


if __name__ == "__main__":
    # Test visualizations
    # Generate sample data
    t = np.linspace(0, 5, 5 * 360)
    signal = np.sin(2 * np.pi * 1.2 * t) + 0.2 * np.random.randn(len(t))
    
    # Add synthetic R-peaks
    peaks = np.arange(0, len(t), 360).astype(int)
    for p in peaks:
        if p < len(t):
            signal[p:p+10] += 1.0
            
    # Plot ECG
    plot_ecg_signal(signal, show_peaks=peaks, title="Test ECG Signal")
    plt.show()
    
    print("Visualization test completed")