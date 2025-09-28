"""
Coherence-based connectivity helpers for BCI feature selection
============================================================

This module contains functions for computing functional connectivity
features using coherence, adapted from pyRiemann examples.

Author: BCI Handbook Chapter 4.3 - Feature Selection Exercise
License: BSD (3-clause)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pyriemann.estimation import Coherences


def NearestSPD(A):
    """
    Find the nearest positive definite matrix to A.

    Parameters
    ----------
    A : ndarray
        Input matrix

    Returns
    -------
    B : ndarray
        Nearest positive definite matrix
    """
    # Ensure symmetry
    B = (A + A.T) / 2

    # Eigenvalue decomposition
    eigvals, eigvecs = np.linalg.eigh(B)

    # Clip negative eigenvalues
    eigvals = np.maximum(eigvals, 1e-6)

    # Reconstruct matrix
    B = eigvecs @ np.diag(eigvals) @ eigvecs.T

    return B


class Connectivities(Coherences):
    """Getting connectivity features from epoch"""

    def transform(self, X):
        X_coh = super().transform(X)
        X_con = np.mean(X_coh, axis=-1, keepdims=False)
        return X_con


def create_connectivity_features_matrix(connectivity_matrices):
    """
    Convert connectivity matrices to feature vectors.

    Parameters
    ----------
    connectivity_matrices : ndarray, shape (n_epochs, n_channels, n_channels)
        Connectivity matrices

    Returns
    -------
    features : ndarray, shape (n_epochs, n_features)
        Feature vectors (upper triangular parts)
    feature_names : list
        Names of features (channel pairs)
    """
    n_epochs, n_channels, _ = connectivity_matrices.shape

    # Get upper triangular indices (exclude diagonal)
    triu_indices = np.triu_indices(n_channels, k=1)
    n_features = len(triu_indices[0])

    # Extract features
    features = np.zeros((n_epochs, n_features))
    for epoch in range(n_epochs):
        features[epoch] = connectivity_matrices[epoch][triu_indices]

    # Create feature names
    feature_names = []
    for i, j in zip(triu_indices[0], triu_indices[1]):
        feature_names.append(f"conn_{i:02d}_{j:02d}")

    return features, feature_names


def plot_connectivity_comparison(
    conn_matrices, labels, ch_names=None, title="Connectivity Comparison"
):
    """
    Plot average connectivity matrices for different classes.

    Parameters
    ----------
    conn_matrices : ndarray
        Connectivity matrices (n_epochs, n_channels, n_channels)
    labels : ndarray
        Class labels
    ch_names : list or None
        Channel names
    title : str
        Plot title
    """
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)

    fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 4))
    if n_classes == 1:
        axes = [axes]

    for i, label in enumerate(unique_labels):
        # Average connectivity for this class
        class_mask = labels == label
        avg_conn = np.mean(conn_matrices[class_mask], axis=0)

        # Plot
        im = axes[i].imshow(avg_conn, cmap="viridis", aspect="auto")
        axes[i].set_title(f"Class {label}")
        axes[i].set_xlabel("Channel")
        axes[i].set_ylabel("Channel")

        if ch_names is not None:
            # Set channel name ticks (subsample if too many)
            n_ticks = min(10, len(ch_names))
            tick_indices = np.linspace(0, len(ch_names) - 1, n_ticks, dtype=int)
            axes[i].set_xticks(tick_indices)
            axes[i].set_xticklabels([ch_names[idx] for idx in tick_indices], rotation=45)
            axes[i].set_yticks(tick_indices)
            axes[i].set_yticklabels([ch_names[idx] for idx in tick_indices])

        plt.colorbar(im, ax=axes[i])

    plt.suptitle(title)
    plt.tight_layout()
    return fig


def analyze_connectivity_patterns(
    connectivity_features, labels, feature_names=None, top_k=10
):
    """
    Analyze connectivity patterns that differentiate between classes.

    Parameters
    ----------
    connectivity_features : ndarray
        Connectivity feature matrix
    labels : ndarray
        Class labels
    feature_names : list or None
        Feature names
    top_k : int
        Number of top discriminative features to analyze

    Returns
    -------
    analysis_results : dict
        Analysis results including top features and statistics
    """

    unique_labels = np.unique(labels)
    n_features = connectivity_features.shape[1]

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    # Compute t-statistics for each feature
    t_stats = np.zeros(n_features)
    p_values = np.zeros(n_features)

    for i in range(n_features):
        feature_values = connectivity_features[:, i]

        if len(unique_labels) == 2:
            # Two-sample t-test
            group1 = feature_values[labels == unique_labels[0]]
            group2 = feature_values[labels == unique_labels[1]]
            t_stat, p_val = stats.ttest_ind(group1, group2)
            t_stats[i] = float(np.abs(t_stat))
            p_values[i] = float(p_val)

    # Get top discriminative features
    top_indices = np.argsort(t_stats)[-top_k:][::-1]

    analysis_results = {
        "top_features": top_indices,
        "top_feature_names": [feature_names[i] for i in top_indices],
        "top_t_stats": t_stats[top_indices],
        "top_p_values": p_values[top_indices],
        "all_t_stats": t_stats,
        "all_p_values": p_values,
    }

    return analysis_results


def plot_feature_connectivity_map(
    feature_importances, n_channels, ch_names=None, title="Feature Connectivity Map"
):
    """
    Plot feature importances as a connectivity map.

    Parameters
    ----------
    feature_importances : ndarray
        Feature importance scores
    n_channels : int
        Number of channels
    ch_names : list or None
        Channel names
    title : str
        Plot title
    """
    # Create connectivity matrix from feature importances
    conn_map = np.zeros((n_channels, n_channels))

    # Fill upper triangle with feature importances
    triu_indices = np.triu_indices(n_channels, k=1)
    conn_map[triu_indices] = feature_importances

    # Make symmetric
    conn_map = conn_map + conn_map.T

    # Plot
    plt.figure(figsize=(10, 8))
    im = plt.imshow(conn_map, cmap="viridis", aspect="auto")
    plt.title(title)
    plt.xlabel("Channel")
    plt.ylabel("Channel")

    if ch_names is not None:
        # Set channel name ticks (subsample if too many)
        n_ticks = min(15, len(ch_names))
        tick_indices = np.linspace(0, len(ch_names) - 1, n_ticks, dtype=int)
        plt.xticks(tick_indices, [ch_names[idx] for idx in tick_indices], rotation=45)
        plt.yticks(tick_indices, [ch_names[idx] for idx in tick_indices])

    plt.colorbar(im, label="Feature Importance")
    plt.tight_layout()
    return plt.gcf()
