"""
Decoding Methods — MOABB Benchmark adapted to match the exercise questions.

This script follows the tasks in
"CodingExercise_BCIHandbook_DecodingMethodsNew.md" and contains explicit
references to question numbers (Q1..Q18) in comments.

Main goals:
- Benchmark multiple pipelines (CSP+LDA, Cov+MDM, TS+LR/SVM, Xdawn+MDM, FgMDM)
- Use MOABB evaluation protocols (cross-subject by default)
- Report balanced accuracy and visualize per-subject distributions
- Perform pairwise statistical tests with multiple-comparison correction
- Provide hooks for robustness checks and learning curves

Requirements (install with pip if missing):
- moabb, mne, pyriemann, numpy, pandas, scipy, scikit-learn, matplotlib, seaborn
"""

from __future__ import annotations

import json
import os
import warnings
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import (
    StratifiedKFold,
    learning_curve as sk_learning_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# PyRiemann transforms/classifiers for EEG
from pyriemann.classification import FgMDM, MDM
from pyriemann.estimation import Covariances, XdawnCovariances
from pyriemann.spatialfilters import CSP
from pyriemann.tangentspace import TangentSpace

# MOABB imports
from moabb.datasets import BNCI2014_001, BNCI2014_008, EPFLP300, PhysionetMI
from moabb.evaluations import (
    CrossSessionEvaluation,
    CrossSubjectEvaluation,
    WithinSessionEvaluation,
)
from moabb.paradigms import MotorImagery, P300

warnings.filterwarnings("ignore", category=RuntimeWarning)


# Q18: Reproducibility configuration
np.random.seed(42)


@dataclass
class Config:
    # Q1: Dataset selection (MI + optional ERP)
    mi_datasets: Optional[List[str]] = None
    erp_datasets: Optional[List[str]] = None

    # Q2: Neuroscience-driven preprocessing parameters
    mi_fmin: float = 8.0
    mi_fmax: float = 35.0
    mi_resample: Optional[float] = 128.0

    erp_fmin: float = 0.1
    erp_fmax: float = 20.0
    erp_resample: Optional[float] = 128.0

    # Q3: Evaluation protocol
    eval_mode: str = (
        "cross_subject"  # one of: cross_subject, within_session, cross_session
    )

    # Q9: Metrics (primary)
    scoring_mi: str = "balanced_accuracy"
    scoring_erp: str = "balanced_accuracy"  # for P300, ROC-AUC also common

    # Controls
    n_jobs: int = 1
    overwrite: bool = False
    out_dir: str = "decoding_outputs"

    # Optional toggles
    include_fgmdm: bool = True
    run_erp: bool = False  # set True to also run ERP benchmark
    do_robustness_subset_channels: bool = False
    do_learning_curve: bool = True  # Q15


def make_output_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


# Q1: Dataset selection and inspection
def instantiate_mi_datasets(names: Iterable[str]):
    ds = []
    for n in names:
        if n == "BNCI2014_001":
            ds.append(BNCI2014_001())
        elif n == "PhysionetMI":
            ds.append(PhysionetMI())
        else:
            raise ValueError(f"Unknown MI dataset: {n}")
    return ds


def instantiate_erp_datasets(names: Iterable[str]):
    ds = []
    for n in names:
        if n == "EPFLP300":
            ds.append(EPFLP300())
        elif n == "BNCI2014_008":
            ds.append(BNCI2014_008())
        else:
            raise ValueError(f"Unknown ERP dataset: {n}")
    return ds


def inspect_dataset(
    paradigm, dataset, max_subjects: Optional[int] = None
) -> pd.DataFrame:
    """
    Q1: Print and return basic metadata:
    - number of subjects, channels, sampling rate (via paradigm data), trials per class.
    """
    subs = dataset.subjects
    if max_subjects is not None:
        subs = subs[:max_subjects]

    X, y, meta = paradigm.get_data(dataset=dataset, subjects=subs)
    labels, counts = np.unique(y, return_counts=True)

    print(
        f"Dataset: {dataset.code if hasattr(dataset, 'code') else dataset.__class__.__name__}"
    )
    print(
        f" Subjects: {len(subs)} | Channels: inferred from X (X.shape[1]) = {X.shape[1]}"
    )
    print(f" Samples per epoch (X.shape[2]): {X.shape[2]}")
    print(" Class distribution:", dict(zip(labels.tolist(), counts.tolist())))
    print(" Meta columns:", list(meta.columns))
    print()

    df = pd.DataFrame({"label": labels, "count": counts})
    df["dataset"] = dataset.__class__.__name__
    return df


# Q2: Neuroscience-driven preprocessing plan via paradigms
def build_mi_paradigm(cfg: Config) -> MotorImagery:
    """
    MI paradigm with typical band-pass over mu/beta rhythms and resampling.
    """
    p = MotorImagery(fmin=cfg.mi_fmin, fmax=cfg.mi_fmax, resample=cfg.mi_resample)
    # Q9: keep default scoring of the MOABB paradigm/evaluation
    # (scoring is controlled by MOABB; do not set attribute here)
    # (no-op line to preserve line count)
    return p


def build_erp_paradigm(cfg: Config) -> P300:
    """
    ERP paradigm with low-frequency emphasis and resampling.
    """
    p = P300(fmin=cfg.erp_fmin, fmax=cfg.erp_fmax, resample=cfg.erp_resample)
    # Keep default scoring (do not set attribute)
    return p


# Q4..Q8: Pipelines
def build_pipelines_mi(cfg: Config) -> Dict[str, Pipeline]:
    """
    MI pipelines:
    - Q4: CSP + LDA
    - Q5: Covariances + MDM
    - Q6: Tangent Space + (LR or SVM)
    - Q8: Optional FgMDM (filter-bank MDM)
    """
    pipelines: Dict[str, Pipeline] = {}

    pipelines["CSP + LDA"] = Pipeline(
        steps=[
            ("csp", CSP(nfilter=8)),
            ("clf", LDA()),
        ]
    )

    pipelines["Cov + MDM"] = Pipeline(
        steps=[
            ("cov", Covariances(estimator="oas")),
            ("mdm", MDM(metric="riemann")),
        ]
    )

    pipelines["TS + LR"] = Pipeline(
        steps=[
            ("cov", Covariances(estimator="oas")),
            ("ts", TangentSpace(metric="riemann")),
            ("clf", LogisticRegression(max_iter=1000, n_jobs=cfg.n_jobs)),
        ]
    )

    pipelines["TS + RBF-SVM"] = Pipeline(
        steps=[
            ("cov", Covariances(estimator="oas")),
            ("ts", TangentSpace(metric="riemann")),
            ("clf", SVC(kernel="rbf", C=1.0)),
        ]
    )

    if cfg.include_fgmdm:
        pipelines["FgMDM"] = Pipeline(
            steps=[
                # FgMDM internally handles filter-bank decomposition
                ("fgmdm", FgMDM(metric="riemann")),
            ]
        )

    return pipelines


def build_pipelines_erp(cfg: Config) -> Dict[str, Pipeline]:
    """
    ERP pipelines:
    - Q7: Xdawn + MDM
    - Also include Cov + MDM and TS + LR as comparators
    """
    pipelines: Dict[str, Pipeline] = {}

    pipelines["XdawnCov + MDM"] = Pipeline(
        steps=[
            ("xdawncov", XdawnCovariances(nfilter=4, estimator="oas")),
            ("mdm", MDM(metric="riemann")),
        ]
    )

    pipelines["Cov + MDM"] = Pipeline(
        steps=[
            ("cov", Covariances(estimator="oas")),
            ("mdm", MDM(metric="riemann")),
        ]
    )

    pipelines["TS + LR"] = Pipeline(
        steps=[
            ("cov", Covariances(estimator="oas")),
            ("ts", TangentSpace(metric="riemann")),
            ("clf", LogisticRegression(max_iter=1000, n_jobs=cfg.n_jobs)),
        ]
    )

    return pipelines


# Q3: Evaluation protocol selection
def make_evaluation(paradigm, datasets: List, cfg: Config):
    mode = cfg.eval_mode.lower()
    if mode == "cross_subject":
        return CrossSubjectEvaluation(
            paradigm=paradigm,
            datasets=datasets,
            overwrite=cfg.overwrite,
        )
    if mode == "within_session":
        return WithinSessionEvaluation(
            paradigm=paradigm,
            datasets=datasets,
            overwrite=cfg.overwrite,
        )
    if mode == "cross_session":
        return CrossSessionEvaluation(
            paradigm=paradigm,
            datasets=datasets,
            overwrite=cfg.overwrite,
        )
    raise ValueError(f"Unknown eval_mode={cfg.eval_mode}")


# Q9/Q10/Q12: Run benchmarking and return a tidy result DataFrame
def run_benchmark(
    cfg: Config,
    datasets: List,
    paradigm,
    pipelines: Dict[str, Pipeline],
    suffix: str,
) -> pd.DataFrame:
    print("=" * 60)
    print(f"Benchmark start | Paradigm={paradigm.__class__.__name__} | {suffix}")
    print("=" * 60)

    eval_obj = make_evaluation(paradigm, datasets, cfg)
    results = eval_obj.evaluate(pipelines)

    # Add suffix tag and ensure consistent col names
    results["benchmark"] = suffix

    # Print quick summary
    print("Pipelines evaluated:", results["pipeline"].unique().tolist())
    print("Datasets:", results["dataset"].unique().tolist())
    print("Subjects (unique):", results["subject"].nunique())
    print("Scoring used:", getattr(paradigm, "scoring", "unknown"))
    print()

    return results


# Q10: Visualize per-subject performance and pipeline ranking
def plot_per_subject_performance(df: pd.DataFrame, out_dir: str, tag: str) -> None:
    plt.style.use("seaborn-v0_8")
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x="pipeline", y="score", hue="dataset")
    plt.title(f"Per-subject distribution ({tag})")
    plt.ylabel("Score (primary metric)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"boxplot_{tag}.png"), dpi=200)
    plt.close()

    # Ranking by mean score across subjects (and datasets)
    means = df.groupby("pipeline")["score"].mean().sort_values(ascending=True)
    plt.figure(figsize=(8, 4))
    sns.barplot(x=means.values, y=means.index, orient="h", palette="viridis")
    plt.title(f"Pipeline ranking by mean score ({tag})")
    plt.xlabel("Mean score")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"ranking_{tag}.png"), dpi=200)
    plt.close()


# Q12: Pairwise statistical comparisons with Holm correction
def holm_bonferroni(pvals: Dict[Tuple[str, str], float]) -> Dict[Tuple[str, str], float]:
    """
    Returns Holm-Bonferroni adjusted p-values for a dict mapping pairs->p.
    """
    items = list(pvals.items())
    m = len(items)
    # Sort by p ascending
    items.sort(key=lambda kv: kv[1])
    adjusted = {}
    for i, ((a, b), p) in enumerate(items, start=1):
        # Holm step-down: compare p * (m - i + 1)
        adj_p = min((m - i + 1) * p, 1.0)
        adjusted[(a, b)] = adj_p
    # Return in original keys order
    return {k: adjusted[k] for k in pvals.keys()}


def pairwise_stats_per_dataset(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    For each dataset, compute pairwise Wilcoxon tests across pipelines on
    per-subject scores. Also compute Cohen's d on paired differences.
    Returns dict: dataset_name -> DataFrame with p, p_holm, d, significant flag.
    """
    outputs = {}
    for ds, g in df.groupby("dataset"):
        # pivot: index = subject, columns = pipeline, values = score
        pivot = g.pivot_table(
            index="subject", columns="pipeline", values="score", aggfunc="mean"
        ).dropna(axis=0, how="any")

        pipelines = list(pivot.columns)
        raw_p = {}
        dvals = {}

        for a, b in combinations(pipelines, 2):
            diff = pivot[a].values - pivot[b].values
            # Wilcoxon signed-rank test (non-param) on paired differences
            try:
                stat, p = stats.wilcoxon(diff)
            except ValueError:
                # If diff all zeros or too few samples, fall back to t-test
                stat, p = stats.ttest_rel(pivot[a].values, pivot[b].values)
            raw_p[(a, b)] = p

            # Cohen's d for paired differences
            d = diff.mean() / (diff.std(ddof=1) + 1e-12)
            dvals[(a, b)] = d

        p_holm = holm_bonferroni(raw_p)

        rows = []
        for a, b in raw_p.keys():
            rows.append(
                {
                    "pipeline_a": a,
                    "pipeline_b": b,
                    "p": raw_p[(a, b)],
                    "p_holm": p_holm[(a, b)],
                    "cohens_d": dvals[(a, b)],
                    "significant_0.05": p_holm[(a, b)] < 0.05,
                }
            )

        outputs[ds] = pd.DataFrame(rows).sort_values("p_holm")
    return outputs


def plot_significance_matrix(
    df: pd.DataFrame, out_dir: str, tag: str, alpha: float = 0.05
) -> None:
    """
    Heatmap of significance (Holm-corrected) per dataset averaged if multiple.
    Cells show -log10(p_holm) with mask for non-significant pairs.
    """
    # Average p-values across datasets if multiple provided
    # Build a pipeline list union
    all_pipes = sorted(df["pipeline"].unique().tolist())

    # Compute per-dataset pairwise p-values
    per_ds = pairwise_stats_per_dataset(df)

    # Aggregate into a matrix of mean -log10(p_holm)
    mats = []
    for ds, table in per_ds.items():
        mat = pd.DataFrame(index=all_pipes, columns=all_pipes, data=np.nan)
        for _, r in table.iterrows():
            a, b = r["pipeline_a"], r["pipeline_b"]
            val = -np.log10(max(r["p_holm"], 1e-12))
            mat.loc[a, b] = val
            mat.loc[b, a] = val
        np.fill_diagonal(mat.values, 0.0)
        mats.append(mat)

    if not mats:
        return

    mean_mat = sum(mats) / len(mats)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        mean_mat,
        cmap="magma",
        annot=False,
        cbar_kws={"label": "-log10(p_holm)"},
        square=True,
    )
    plt.title(f"Significance matrix (Holm-corrected) — {tag}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"significance_matrix_{tag}.png"), dpi=200)
    plt.close()


# Q13: Rank-based analysis (average ranks across datasets/subjects)
def compute_average_ranks(df: pd.DataFrame) -> pd.DataFrame:
    # rank per subject per dataset, then average ranks per pipeline
    def _rank_group(g):
        # lower rank = better (rank 1 = best); we want descending score
        g = g.copy()
        g["rank"] = g["score"].rank(ascending=False, method="average")
        return g

    ranked = df.groupby(["dataset", "subject"]).apply(_rank_group, include_groups=False)
    avg_ranks = ranked.groupby("pipeline")["rank"].mean().sort_values()
    return avg_ranks.reset_index().rename(columns={"rank": "average_rank"})


# Q14: Robustness — restrict to a channel subset (e.g., motor strip)
def restrict_channels_in_paradigm(paradigm, channels: List[str]):
    """
    MOABB paradigms accept a 'channels' attribute in many versions.
    If unsupported, this fallback will ignore the request.
    """
    if hasattr(paradigm, "channels"):
        paradigm.channels = channels
        print("Restricted channels:", channels)
    else:
        print("Warning: paradigm has no 'channels' attribute; skipping restriction.")


# Q15: Learning curves for one pipeline on pooled data (single dataset)
def learning_curve_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    pipeline: Pipeline,
    out_dir: str,
    tag: str,
    scoring: str = "balanced_accuracy",
) -> None:
    """
    Compute and plot a learning curve using scikit-learn utilities directly on
    trial-wise data X,y (shape: n_trials x n_channels x n_samples).
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_sizes = np.linspace(0.2, 1.0, 6)

    tr_sizes, tr_scores, te_scores = sk_learning_curve(
        estimator=pipeline,
        X=X,
        y=y,
        train_sizes=train_sizes,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
    )

    tr_mean = tr_scores.mean(axis=1)
    tr_std = tr_scores.std(axis=1)
    te_mean = te_scores.mean(axis=1)
    te_std = te_scores.std(axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(tr_sizes, tr_mean, "o-", label="Train")
    plt.fill_between(tr_sizes, tr_mean - tr_std, tr_mean + tr_std, alpha=0.2)
    plt.plot(tr_sizes, te_mean, "o-", label="Validation")
    plt.fill_between(tr_sizes, te_mean - te_std, te_mean + te_std, alpha=0.2)
    plt.xlabel("Training set size (fraction of data)")
    plt.ylabel(f"{scoring}")
    plt.title(f"Learning curve — {tag}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"learning_curve_{tag}.png"), dpi=200)
    plt.close()


def save_results(df: pd.DataFrame, cfg: Config, name: str) -> str:
    path = os.path.join(cfg.out_dir, f"results_{name}.csv")
    df.to_csv(path, index=False)
    print(f"Saved results -> {path}")
    return path


def print_reproducibility_info(cfg: Config) -> None:
    # Q18: Reproducibility checklist
    info = {
        "random_seed": 42,
        "packages": {
            "moabb": "import at runtime",
            "mne": "import at runtime",
            "pyriemann": "import at runtime",
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": stats.__version__ if hasattr(stats, "__version__") else "NA",
        },
        "config": cfg.__dict__,
    }
    print("Reproducibility info:")
    print(json.dumps(info, indent=2, default=str))


def main():
    # Configuration aligned to the exercise
    cfg = Config(
        mi_datasets=["BNCI2014_001"],  # Q1: MI dataset choice
        erp_datasets=["EPFLP300"],  # Q1: ERP alternative (enable run_erp=True)
        run_erp=False,  # set True to also run ERP section
        include_fgmdm=True,
        eval_mode="cross_subject",  # Q3: evaluation choice
        n_jobs=1,
        overwrite=False,
        out_dir="decoding_outputs",
        do_robustness_subset_channels=False,  # Q14
        do_learning_curve=True,  # Q15
    )

    make_output_dir(cfg.out_dir)

    # Q2: Build paradigms with neuroscience-motivated parameters
    mi_paradigm = build_mi_paradigm(cfg)
    erp_paradigm = build_erp_paradigm(cfg)

    # Q1: Instantiate datasets and inspect
    mi_datasets = instantiate_mi_datasets(cfg.mi_datasets)
    for ds in mi_datasets:
        inspect_dataset(mi_paradigm, ds, max_subjects=2)

    if cfg.run_erp:
        erp_datasets = instantiate_erp_datasets(cfg.erp_datasets)
        for ds in erp_datasets:
            inspect_dataset(erp_paradigm, ds, max_subjects=2)
    else:
        erp_datasets = []

    # Q4..Q8: Define pipelines
    mi_pipelines = build_pipelines_mi(cfg)
    erp_pipelines = build_pipelines_erp(cfg) if cfg.run_erp else {}

    # Q3: Define evaluation and run
    mi_results = run_benchmark(
        cfg=cfg,
        datasets=mi_datasets,
        paradigm=mi_paradigm,
        pipelines=mi_pipelines,
        suffix="MI",
    )
    save_results(mi_results, cfg, "MI")

    if cfg.run_erp:
        erp_results = run_benchmark(
            cfg=cfg,
            datasets=erp_datasets,
            paradigm=erp_paradigm,
            pipelines=erp_pipelines,
            suffix="ERP",
        )
        save_results(erp_results, cfg, "ERP")
    else:
        erp_results = pd.DataFrame(columns=mi_results.columns)

    # Q10: Visualizations per-subject and ranking
    plot_per_subject_performance(mi_results, cfg.out_dir, tag="MI")
    if cfg.run_erp and not erp_results.empty:
        plot_per_subject_performance(erp_results, cfg.out_dir, tag="ERP")

    # Q12: Statistical comparisons per dataset, Holm-corrected
    stats_mi = pairwise_stats_per_dataset(mi_results)
    for ds_name, table in stats_mi.items():
        path = os.path.join(cfg.out_dir, f"pairwise_stats_{ds_name}_MI.csv")
        table.to_csv(path, index=False)
        print(f"Saved pairwise stats -> {path}")

    plot_significance_matrix(mi_results, cfg.out_dir, tag="MI")
    if cfg.run_erp and not erp_results.empty:
        stats_erp = pairwise_stats_per_dataset(erp_results)
        for ds_name, table in stats_erp.items():
            path = os.path.join(cfg.out_dir, f"pairwise_stats_{ds_name}_ERP.csv")
            table.to_csv(path, index=False)
            print(f"Saved pairwise stats -> {path}")
        plot_significance_matrix(erp_results, cfg.out_dir, tag="ERP")

    # Q11/Q13: Multi-dataset summary + average ranks
    all_results = pd.concat([mi_results, erp_results], ignore_index=True)
    avg_ranks = compute_average_ranks(all_results)
    avg_ranks.to_csv(os.path.join(cfg.out_dir, "average_ranks.csv"), index=False)
    print("Average ranks:\n", avg_ranks)

    # Q14: Robustness checks (optional): channel subset (example motor strip)
    if cfg.do_robustness_subset_channels:
        motor_channels = ["C3", "Cz", "C4", "CP3", "CPz", "CP4"]
        restrict_channels_in_paradigm(mi_paradigm, motor_channels)
        # Re-run quick MI benchmark with restricted channels
        mi_results_subset = run_benchmark(
            cfg=cfg,
            datasets=mi_datasets,
            paradigm=mi_paradigm,
            pipelines=mi_pipelines,
            suffix="MI_motor_strip",
        )
        save_results(mi_results_subset, cfg, "MI_motor_strip")
        plot_per_subject_performance(mi_results_subset, cfg.out_dir, tag="MI_motor_strip")

    # Q15: Learning curves and data efficiency (one pipeline, one dataset)
    if cfg.do_learning_curve:
        # Use first MI dataset and first pipeline by default
        ds = mi_datasets[0]
        X, y, meta = mi_paradigm.get_data(dataset=ds, subjects=[ds.subjects[0]])
        pipe_name, pipe_obj = list(mi_pipelines.items())[0]
        tag = f"{ds.__class__.__name__}_{pipe_name}_MI"
        learning_curve_pipeline(X=X, y=y, pipeline=pipe_obj, out_dir=cfg.out_dir, tag=tag)

    # Q16: Bias/confound audit — checks to log in report
    print(
        "Q16 notes:\n"
        "- Cross-subject protocol ensures no subject leakage across folds.\n"
        "- Balanced accuracy mitigates class imbalance effects.\n"
        "- MOABB handles splits across sessions/runs consistently.\n"
        "- Avoid overlapping windows in your custom preprocessing (not used here).\n"
    )

    # Q17: Report summary — minimal console summary
    print("Q17 summary (console):")
    print(
        "Which pipelines performed best (by mean score, MI)?\n",
        mi_results.groupby("pipeline")["score"].mean().sort_values(ascending=False),
    )
    if cfg.run_erp and not erp_results.empty:
        print(
            "Which pipelines performed best (by mean score, ERP)?\n",
            erp_results.groupby("pipeline")["score"].mean().sort_values(ascending=False),
        )

    # Q18: Reproducibility checklist
    print_reproducibility_info(cfg)

    print("\nCompleted benchmarking. Outputs saved to:", cfg.out_dir)
    print(
        "Figures: boxplot_*.png, ranking_*.png, significance_matrix_*.png; "
        "CSV: results_*.csv, pairwise_stats_*.csv, average_ranks.csv"
    )

    # Q9: Hint for negative control (label shuffling)
    print(
        "Q9 negative control suggestion:\n"
        "Repeat one evaluation with shuffled labels before fitting.\n"
        "Use a custom wrapper or pre-shuffle y within an ad-hoc CV loop.\n"
        "(Not executed by default to keep runtime reasonable.)"
    )


if __name__ == "__main__":
    main()
