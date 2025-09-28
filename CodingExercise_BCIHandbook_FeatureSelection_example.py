"""
Feature Selection using Random Forest on Functional Connectivity
================================================================

This script demonstrates comprehensive feature selection using Random Forest
classification on coherence-based connectivity patterns for BCI applications.
Combines real MNE motor imagery data with advanced feature selection techniques.

Author: BCI Handbook Chapter 4.3
License: BSD (3-clause)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import SVC
import warnings

warnings.filterwarnings("ignore")

# Import helper modules
from helpers.data_preprocessing_helpers import load_motor_imagery_data, balance_classes
from helpers.coherence_helpers import (
    Connectivities,
    create_connectivity_features_matrix,
    plot_connectivity_comparison,
    analyze_connectivity_patterns,
    plot_feature_connectivity_map,
)
from helpers.feature_selection_helpers import (
    random_forest_feature_selection,
    plot_feature_importance,
    plot_feature_selection_comparison,
    ensemble_feature_selection,
    stability_selection,
)

# Set random seed for reproducibility
np.random.seed(42)

# Data loading parameters
SUBJECT = 1
RUNS = [4, 6, 8, 10, 12, 14]  # Motor imagery runs
TMIN, TMAX = 1.0, 2.0  # Time window for motor imagery
FILTER_FREQ = (8.0, 35.0)  # Bandpass filter range

# Connectivity parameters
CONNECTIVITY_METHOD = "instantaneous"
FMIN, FMAX = 8.0, 35.0  # Frequency range (for future use)
SAMPLING_FREQ = 160.0

# Feature selection parameters
N_FEATURES_TO_SELECT = 15
N_ESTIMATORS = 200
RANDOM_STATE = 42

# Cross-validation parameters
CV_FOLDS = 5


# Load motor imagery data
X, y, epochs = load_motor_imagery_data(
    subject=SUBJECT,
    runs=RUNS,
    tmin=TMIN,
    tmax=TMAX,
    filter_freq=FILTER_FREQ,
    verbose=True,
)

# Balance classes if needed
unique_labels, counts = np.unique(y, return_counts=True)
if np.abs(counts[0] - counts[1]) > 10:  # If imbalanced
    print(f"\nBalancing classes (original: {counts})...")
    X, y = balance_classes(X, y, method="undersample", random_state=RANDOM_STATE)
    print(f"Balanced classes: {np.bincount(y)}")
ch_names = epochs.ch_names
fs_value = SAMPLING_FREQ

connectivity_transformer = Connectivities(
    coh=CONNECTIVITY_METHOD, fmin=FMIN, fmax=FMAX, fs=fs_value
)
connectivity_transformer.fit(X, y)
connectivity_matrices = connectivity_transformer.transform(X)

fig_conn = plot_connectivity_comparison(
    connectivity_matrices,
    y,
    ch_names=ch_names,
    title="Average Connectivity Patterns by Class",
)
plt.savefig("connectivity_patterns_by_class.png", dpi=150, bbox_inches="tight")
plt.show()

# =============================================================================
# Step 3: Feature Extraction and Vectorization
# =============================================================================

connectivity_features, feature_names = create_connectivity_features_matrix(
    connectivity_matrices
)
connectivity_analysis = analyze_connectivity_patterns(
    connectivity_features, y, feature_names, top_k=20
)

# =============================================================================
# Step 4: Random Forest Feature Selection
# =============================================================================

rf_model, feature_importances, feature_ranking = random_forest_feature_selection(
    connectivity_features, y, n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE
)
selected_features = feature_ranking[:N_FEATURES_TO_SELECT]
X_selected_rf = connectivity_features[:, selected_features]

# Plot feature importance
fig_imp = plot_feature_importance(
    feature_importances,
    feature_names,
    top_k=25,
    title=f"Top 25 Random Forest Feature Importances",
)
plt.savefig("random_forest_feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()

# Plot connectivity map of feature importances
fig_map = plot_feature_connectivity_map(
    feature_importances,
    len(ch_names),
    ch_names,
    title="Random Forest Feature Importance Connectivity Map",
)
plt.savefig("feature_importance_connectivity_map.png", dpi=150, bbox_inches="tight")
plt.show()

# =============================================================================
# Step 5: Ensemble Feature Selection
# =============================================================================

ensemble_features, method_results = ensemble_feature_selection(
    connectivity_features,
    y,
    methods=["random_forest", "univariate", "rfe"],
    n_features=N_FEATURES_TO_SELECT,
    random_state=RANDOM_STATE,
)

X_selected_ensemble = connectivity_features[:, ensemble_features]

# =============================================================================
# Step 6: Stability Selection
# =============================================================================

stable_features, selection_probs = stability_selection(
    connectivity_features, y, n_bootstrap=50, threshold=0.6, random_state=RANDOM_STATE
)

if len(stable_features) > 0:
    # Take top N_FEATURES_TO_SELECT if we have more
    if len(stable_features) > N_FEATURES_TO_SELECT:
        stable_probs_sorted = np.argsort(selection_probs[stable_features])[::-1]
        stable_features = stable_features[stable_probs_sorted[:N_FEATURES_TO_SELECT]]

    X_selected_stable = connectivity_features[:, stable_features]

    print(f"Stable features (selection probability > 0.6):")
    for i, feat_idx in enumerate(stable_features):
        feature_name = feature_names[feat_idx]
        prob = selection_probs[feat_idx]
        print(f"  {i+1:2d}. {feature_name}: {prob:.3f}")
else:
    print("No stable features found with threshold 0.6")
    X_selected_stable = X_selected_rf  # Fallback to RF selection

# =============================================================================
# Step 7: Classification Performance Evaluation
# =============================================================================

feature_sets = {
    "Original": connectivity_features,
    "Random Forest": X_selected_rf,
    "Ensemble": X_selected_ensemble,
    "Stability": X_selected_stable,
}

classifiers = {
    "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    "SVM (RBF)": SVC(random_state=RANDOM_STATE, kernel="rbf", gamma="scale"),
}

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
results = []

for feature_set_name, X_features in feature_sets.items():
    print(
        f"\nEvaluating feature set: {feature_set_name} ({X_features.shape[1]} features)"
    )

    for clf_name, clf in classifiers.items():
        try:
            scores = cross_val_score(clf, X_features, y, cv=cv, scoring="accuracy")

            result = {
                "Feature Set": feature_set_name,
                "Classifier": clf_name,
                "Mean Accuracy": np.mean(scores),
                "Std Accuracy": np.std(scores),
                "Min Accuracy": np.min(scores),
                "Max Accuracy": np.max(scores),
                "N Features": X_features.shape[1],
            }
            results.append(result)

            print(f"  {clf_name}: {np.mean(scores):.3f} ± {np.std(scores):.3f}")

        except Exception as e:
            print(f"  Error evaluating {clf_name}: {e}")

# Create results DataFrame
results_df = pd.DataFrame(results)

# =============================================================================
# Step 8: Results Visualization and Analysis
# =============================================================================

# Performance comparison plot
fig_comp = plot_feature_selection_comparison(results_df)
plt.tight_layout()
plt.savefig("feature_selection_performance_comparison.png", dpi=150, bbox_inches="tight")
plt.show()


# =============================================================================
# Step 9: Summary and Conclusions
# =============================================================================

# Feature selection summary
print("Feature Selection Summary:")
print(f"  Original features: {connectivity_features.shape[1]}")
print(f"  Random Forest selected: {X_selected_rf.shape[1]}")
print(f"  Ensemble selected: {X_selected_ensemble.shape[1]}")
print(f"  Stability selected: {X_selected_stable.shape[1]}")
print(f"  Reduction factor: {connectivity_features.shape[1] / N_FEATURES_TO_SELECT:.1f}x")

if not results_df.empty:
    # Best performance analysis
    print("\nBest Performance by Feature Selection Method:")

    for feature_set in feature_sets.keys():
        subset = results_df[results_df["Feature Set"] == feature_set]
        if not subset.empty:
            best_row = subset.loc[subset["Mean Accuracy"].idxmax()]
            print(
                f"  {feature_set:15s}: {best_row['Mean Accuracy']:.3f} ± {best_row['Std Accuracy']:.3f} "
                f"({best_row['Classifier']})"
            )

    # Overall best performance
    best_overall = results_df.loc[results_df["Mean Accuracy"].idxmax()]
    print(f"\nBest Overall Performance:")
    print(f"  {best_overall['Feature Set']} + {best_overall['Classifier']}")
    print(
        f"  Accuracy: {best_overall['Mean Accuracy']:.3f} ± {best_overall['Std Accuracy']:.3f}"
    )
    print(f"  Features: {best_overall['N Features']}")

    # Feature selection impact
    original_best = results_df[results_df["Feature Set"] == "Original"][
        "Mean Accuracy"
    ].max()
    selected_best = results_df[results_df["Feature Set"] != "Original"][
        "Mean Accuracy"
    ].max()

    if not np.isnan(original_best) and not np.isnan(selected_best):
        improvement = selected_best - original_best
        print(f"\nFeature Selection Impact:")
        print(f"  Best with original features: {original_best:.3f}")
        print(f"  Best with selected features: {selected_best:.3f}")
        print(f"  Performance change: {improvement:+.3f}")

        if improvement > 0:
            print(f"  → Feature selection improved performance")
        elif improvement < 0:
            print(f"  → Feature selection reduced performance")
        else:
            print(f"  → Feature selection maintained performance")

# Most discriminative features
print(f"\nMost Discriminative Connectivity Patterns:")
top_features = connectivity_analysis["top_feature_names"][:5]
for i, feature_name in enumerate(top_features):
    print(f"  {i+1}. {feature_name}")
