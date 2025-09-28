"""
Feature Selection using Ensemble Learning on Functional Connectivity
Adapted from pyriemann ensemble coherence example
"""

import matplotlib.pyplot as plt
import numpy as np
from mne import Epochs, pick_types
from mne.io import concatenate_raws
from mne.io.edf import read_raw_edf
from mne.datasets import testing
from mne import create_info
from mne.epochs import EpochsArray
import pandas as pd
import seaborn as sns
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from pyriemann.classification import FgMDM
from pyriemann.estimation import Coherences, Covariances
from pyriemann.spatialfilters import CSP
from pyriemann.tangentspace import TangentSpace


# Define connectivity transformer
class ConnectivityTransformer:
    """Transformer for functional connectivity features."""

    def __init__(self, method="coh"):
        self.method = method

    def fit(self, X, y):
        return self

    def transform(self, X):
        # Compute covariances matrices
        cov = Covariances().fit(X)
        X_cov = cov.transform(X)

        # Compute coherences
        coh = Coherences().fit(X_cov, y)
        X_coh = coh.transform(X_cov)

        return X_coh


# Create synthetic data for demonstration

print("Loading data...")
# For demonstration, we'll create synthetic data
n_channels = 22
n_times = 1000
n_epochs = 100
n_classes = 2

# Create synthetic EEG-like data
np.random.seed(42)
X = np.random.randn(n_epochs, n_channels, n_times)
y = np.random.randint(0, n_classes, n_epochs)

# Create MNE info
ch_names = [f"EEG{i:03d}" for i in range(n_channels)]
ch_types = "eeg"  # Should be a string, not a list
info = create_info(ch_names=ch_names, sfreq=250, ch_types=ch_types)

# Create epochs
epochs = EpochsArray(X, info, tmin=0)

# Feature extraction pipeline
connectivity = ConnectivityTransformer()
X_conn = connectivity.transform(epochs.get_data())

print(f"Connectivity features shape: {X_conn.shape}")

# Feature selection using ensemble methods
# Flatten connectivity matrices for feature selection
X_flat = X_conn.reshape(X_conn.shape[0], -1)

# Use Random Forest for feature importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif

# Feature selection pipeline
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_flat, y)

# Get feature importances
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

print("Top 10 most important features:")
print(f"Feature indices: {indices[:10]}")
print(f"Importance scores: {importances[indices[:10]]}")

# Select top features
n_features = 50  # Select top 50 features
X_selected = X_flat[:, indices[:n_features]]

print(f"Selected features shape: {X_selected.shape}")

# Classification with selected features
clf = Pipeline(
    [("tangent_space", TangentSpace()), ("classifier", LogisticRegression(max_iter=1000))]
)

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for train_idx, test_idx in cv.split(X_selected, y):
    X_train, X_test = X_selected[train_idx], X_selected[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred))

print(f"Classification accuracy: {np.mean(scores):.3f} ± {np.std(scores):.3f}")

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.bar(range(20), importances[indices[:20]])
plt.xlabel("Feature Index")
plt.ylabel("Importance Score")
plt.title("Top 20 Feature Importances")
plt.xticks(range(20), [str(i) for i in indices[:20]], rotation=45)
plt.tight_layout()
plt.savefig("feature_importances.png", dpi=150, bbox_inches="tight")
plt.show()

print("Feature selection analysis completed!")
