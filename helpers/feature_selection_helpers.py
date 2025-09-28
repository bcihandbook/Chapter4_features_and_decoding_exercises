"""
Feature selection helpers for BCI applications
=============================================

This module contains functions for feature selection using Random Forest
and other machine learning techniques for BCI applications.

Author: BCI Handbook Chapter 4.3 - Feature Selection Exercise
License: BSD (3-clause)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone


def random_forest_feature_selection(
    X,
    y,
    n_estimators=200,
    max_depth=None,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
):
    """
    Perform feature selection using Random Forest feature importance.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Feature matrix
    y : ndarray, shape (n_samples,)
        Target labels
    n_estimators : int
        Number of trees in the forest
    max_depth : int or None
        Maximum depth of trees
    min_samples_split : int
        Minimum samples to split a node
    min_samples_leaf : int
        Minimum samples in a leaf
    random_state : int
        Random seed
    n_jobs : int
        Number of parallel jobs

    Returns
    -------
    rf_model : RandomForestClassifier
        Trained Random Forest model
    feature_importances : ndarray
        Feature importance scores
    feature_ranking : ndarray
        Indices of features sorted by importance (descending)
    """
    # Initialize and train Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    rf_model.fit(X, y)

    # Get feature importances
    feature_importances = rf_model.feature_importances_
    feature_ranking = np.argsort(feature_importances)[::-1]

    return rf_model, feature_importances, feature_ranking


def univariate_feature_selection(X, y, score_func=f_classif, k=10):
    """
    Perform univariate feature selection using statistical tests.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Feature matrix
    y : ndarray, shape (n_samples,)
        Target labels
    score_func : callable
        Scoring function (f_classif, mutual_info_classif, etc.)
    k : int
        Number of top features to select

    Returns
    -------
    selector : SelectKBest
        Fitted feature selector
    X_selected : ndarray
        Selected features
    selected_features : ndarray
        Indices of selected features
    scores : ndarray
        Feature scores
    """
    selector = SelectKBest(score_func=score_func, k=k)
    X_selected = selector.fit_transform(X, y)

    selected_features = selector.get_support(indices=True)
    scores = selector.scores_

    return selector, X_selected, selected_features, scores


def recursive_feature_elimination(X, y, estimator=None, n_features=10, step=1, cv=5):
    """
    Perform Recursive Feature Elimination (RFE).

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Feature matrix
    y : ndarray, shape (n_samples,)
        Target labels
    estimator : sklearn estimator or None
        Base estimator. If None, uses LogisticRegression
    n_features : int
        Number of features to select
    step : int
        Number of features to remove at each iteration
    cv : int
        Number of cross-validation folds

    Returns
    -------
    rfe : RFE
        Fitted RFE object
    X_selected : ndarray
        Selected features
    selected_features : ndarray
        Indices of selected features
    ranking : ndarray
        Feature ranking
    """
    if estimator is None:
        estimator = LogisticRegression(random_state=42, max_iter=1000)

    rfe = RFE(estimator=estimator, n_features_to_select=n_features, step=step)
    X_selected = rfe.fit_transform(X, y)

    selected_features = rfe.get_support(indices=True)
    ranking = rfe.ranking_

    return rfe, X_selected, selected_features, ranking


def ensemble_feature_selection(X, y, methods=None, n_features=10, random_state=42):
    """
    Combine multiple feature selection methods using ensemble approach.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Feature matrix
    y : ndarray, shape (n_samples,)
        Target labels
    methods : list of str or None
        Feature selection methods to combine
    n_features : int
        Number of features to select
    random_state : int
        Random seed

    Returns
    -------
    ensemble_features : ndarray
        Indices of features selected by ensemble
    method_results : dict
        Results from individual methods
    """
    if methods is None:
        methods = ["random_forest", "univariate", "rfe"]

    method_results = {}
    feature_scores = np.zeros(X.shape[1])

    for method in methods:
        if method == "random_forest":
            rf_model, importances, ranking = random_forest_feature_selection(
                X, y, random_state=random_state
            )
            # Normalize importances to 0-1 range
            normalized_scores = importances / np.max(importances)
            method_results["random_forest"] = {
                "model": rf_model,
                "scores": normalized_scores,
                "ranking": ranking,
            }
            feature_scores += normalized_scores

        elif method == "univariate":
            selector, X_sel, sel_features, scores = univariate_feature_selection(
                X, y, k=min(n_features * 2, X.shape[1])
            )
            # Normalize scores to 0-1 range
            normalized_scores = scores / np.max(scores)
            method_results["univariate"] = {
                "selector": selector,
                "scores": normalized_scores,
                "selected": sel_features,
            }
            feature_scores += normalized_scores

        elif method == "rfe":
            rfe, X_sel, sel_features, ranking = recursive_feature_elimination(
                X, y, n_features=n_features, cv=3
            )
            # Convert ranking to scores (lower rank = higher score)
            normalized_scores = 1.0 / ranking
            normalized_scores = normalized_scores / np.max(normalized_scores)
            method_results["rfe"] = {
                "rfe": rfe,
                "scores": normalized_scores,
                "ranking": ranking,
            }
            feature_scores += normalized_scores

    # Select top features based on ensemble scores
    ensemble_ranking = np.argsort(feature_scores)[::-1]
    ensemble_features = ensemble_ranking[:n_features]

    return ensemble_features, method_results


def plot_feature_importance(
    feature_importances, feature_names=None, top_k=20, title="Feature Importances"
):
    """
    Plot feature importance scores.

    Parameters
    ----------
    feature_importances : ndarray
        Feature importance scores
    feature_names : list of str or None
        Feature names
    top_k : int
        Number of top features to display
    title : str
        Plot title
    """
    # Sort features by importance
    sorted_idx = np.argsort(feature_importances)[::-1]
    top_idx = sorted_idx[:top_k]
    top_importances = feature_importances[top_idx]

    if feature_names is not None:
        top_names = [feature_names[i] for i in top_idx]
    else:
        top_names = [f"Feature {i}" for i in top_idx]

    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(top_k), top_importances[::-1])
    plt.yticks(range(top_k), top_names[::-1])
    plt.xlabel("Importance Score")
    plt.title(title)
    plt.grid(axis="x", alpha=0.3)

    # Color bars by importance
    colors = plt.cm.viridis(np.linspace(0, 1, top_k))
    for bar, color in zip(bars, colors[::-1]):
        bar.set_color(color)

    plt.tight_layout()
    return plt.gcf()


def plot_feature_selection_comparison(results_df):
    """
    Plot comparison of classification results with and without feature selection.

    Parameters
    ----------
    results_df : pandas.DataFrame
        Results from evaluate_feature_selection
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Accuracy comparison
    ax1 = axes[0, 0]
    sns.barplot(
        data=results_df, x="Classifier", y="Mean Accuracy", hue="Feature Set", ax=ax1
    )
    ax1.set_title("Classification Accuracy Comparison")
    ax1.set_ylabel("Mean Accuracy")
    ax1.tick_params(axis="x", rotation=45)

    # Error bars
    ax2 = axes[0, 1]
    for i, clf in enumerate(results_df["Classifier"].unique()):
        clf_data = results_df[results_df["Classifier"] == clf]
        orig_data = clf_data[clf_data["Feature Set"] == "Original"]
        sel_data = clf_data[clf_data["Feature Set"] == "Selected"]

        if not orig_data.empty:
            ax2.errorbar(
                i - 0.2,
                orig_data["Mean Accuracy"].iloc[0],
                yerr=orig_data["Std Accuracy"].iloc[0],
                fmt="o",
                label="Original" if i == 0 else "",
                color="blue",
                capsize=5,
            )

        if not sel_data.empty:
            ax2.errorbar(
                i + 0.2,
                sel_data["Mean Accuracy"].iloc[0],
                yerr=sel_data["Std Accuracy"].iloc[0],
                fmt="s",
                label="Selected" if i == 0 else "",
                color="orange",
                capsize=5,
            )

    ax2.set_xticks(range(len(results_df["Classifier"].unique())))
    ax2.set_xticklabels(results_df["Classifier"].unique(), rotation=45)
    ax2.set_ylabel("Mean Accuracy")
    ax2.set_title("Accuracy with Error Bars")
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Performance improvement
    ax3 = axes[1, 0]
    improvements = []
    classifiers = []

    for clf in results_df["Classifier"].unique():
        clf_data = results_df[results_df["Classifier"] == clf]
        orig_acc = clf_data[clf_data["Feature Set"] == "Original"]["Mean Accuracy"]
        sel_acc = clf_data[clf_data["Feature Set"] == "Selected"]["Mean Accuracy"]

        if not orig_acc.empty and not sel_acc.empty:
            improvement = sel_acc.iloc[0] - orig_acc.iloc[0]
            improvements.append(improvement)
            classifiers.append(clf)

    colors = ["green" if imp >= 0 else "red" for imp in improvements]
    bars = ax3.bar(classifiers, improvements, color=colors, alpha=0.7)
    ax3.set_ylabel("Accuracy Improvement")
    ax3.set_title("Feature Selection Impact")
    ax3.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    ax3.tick_params(axis="x", rotation=45)

    # Feature reduction
    ax4 = axes[1, 1]

    # Get original and selected feature counts
    orig_data = results_df[results_df["Feature Set"] == "Original"]
    selected_data = results_df[results_df["Feature Set"] != "Original"]

    if not orig_data.empty and not selected_data.empty:
        n_orig = orig_data["N Features"].iloc[0]
        n_sel = selected_data["N Features"].iloc[0]  # Take first selected method
        reduction_factor = n_orig / n_sel

        ax4.bar(
            ["Original", "Selected"],
            [n_orig, n_sel],
            color=["lightblue", "lightgreen"],
            alpha=0.7,
        )
        ax4.set_ylabel("Number of Features")
        ax4.set_title(f"Feature Reduction\n({reduction_factor:.1f}x reduction)")

        # Add text annotations
        for i, (label, value) in enumerate(
            zip(["Original", "Selected"], [n_orig, n_sel])
        ):
            ax4.text(
                i,
                value + max(n_orig, n_sel) * 0.02,
                str(value),
                ha="center",
                va="bottom",
                fontweight="bold",
            )
    else:
        ax4.text(
            0.5,
            0.5,
            "Feature reduction\ndata unavailable",
            ha="center",
            va="center",
            transform=ax4.transAxes,
        )
        ax4.set_title("Feature Reduction")

    return fig


def stability_selection(
    X, y, estimator=None, n_bootstrap=100, threshold=0.6, random_state=42
):
    """
    Perform stability selection to identify robust features.

    Parameters
    ----------
    X : ndarray
        Feature matrix
    y : ndarray
        Target labels
    estimator : sklearn estimator or None
        Base estimator for feature selection
    n_bootstrap : int
        Number of bootstrap samples
    threshold : float
        Selection threshold (fraction of bootstrap samples)
    random_state : int
        Random seed

    Returns
    -------
    stable_features : ndarray
        Indices of stable features
    selection_probs : ndarray
        Selection probability for each feature
    """
    np.random.seed(random_state)

    if estimator is None:
        estimator = RandomForestClassifier(n_estimators=50, random_state=random_state)

    n_samples, n_features = X.shape
    selection_counts = np.zeros(n_features)

    for i in range(n_bootstrap):
        # Bootstrap sample
        bootstrap_idx = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot = X[bootstrap_idx]
        y_boot = y[bootstrap_idx]

        # Train estimator and get feature importances
        est = clone(estimator)
        est.fit(X_boot, y_boot)

        if hasattr(est, "feature_importances_"):
            importances = est.feature_importances_
        elif hasattr(est, "coef_"):
            importances = np.abs(est.coef_.flatten())
        else:
            raise ValueError(
                "Estimator must have feature_importances_ or coef_ attribute"
            )

        # Select top 50% features
        n_select = max(1, n_features // 2)
        top_features = np.argsort(importances)[-n_select:]
        selection_counts[top_features] += 1

    # Compute selection probabilities
    selection_probs = selection_counts / n_bootstrap

    # Select stable features
    stable_features = np.where(selection_probs >= threshold)[0]

    return stable_features, selection_probs
