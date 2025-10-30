# Coding Exercise: Decoding Methods — MOABB Benchmarking and Statistical Analysis

## Goals
- Compare multiple EEG decoding pipelines using a standardized protocol (MOABB)
- Design neuroscience-motivated preprocessing and evaluation choices
- Apply subject-wise cross-validation correctly, avoiding data leakage
- Quantify uncertainty with robust metrics and visualize per-subject distributions
- Perform statistical testing across pipelines and report corrected significance
- Analyze biases and confounds that can inflate decoding performance

## Description
This exercise focuses on benchmarking decoding methods for EEG-based BCIs using the MOABB
framework (Mother Of All BCI Benchmarks). You will select one or more public datasets,
build several pipelines (e.g., CSP+LDA, Riemannian MDM, Tangent Space + SVM), evaluate
them with rigorous cross-validation, and perform statistical comparisons. The intent is to
ground methodological choices in neurophysiology and experimental design rather than
only coding convenience.

EEG paradigms and datasets
- Motor Imagery (MI): two-class hand MI is a common benchmark. Typical preprocessing
  emphasizes mu/beta rhythms (~8–30 Hz) over sensorimotor areas (C3/C4).
- Event-Related Potentials (ERP/P300): oddball or speller paradigms. Typical preprocessing
  includes low-frequency components (e.g., 0.1–20/30 Hz), epoching time-locked to stimuli,
  and spatial filtering (e.g., Xdawn).
- Dataset options in MOABB: for MI, consider BNCI2014-001 or Physionet Motor Imagery;
  for ERP/P300, consider EPFL P300 or BNCI2014-008. Use MOABB to programmatically
  retrieve exact channel lists, sampling rates, and trial counts for transparency.

Experimental recording overview
- Multi-channel EEG (typically 22–64 electrodes) recorded during repeated trials per class.
- Events/cues mark trial onsets. For MI, epochs are usually extracted a few seconds after
  cue onset; for ERP, epochs are short and time-locked to stimuli with baseline periods.
- Inter-subject variability and session effects are substantial; subject-wise evaluation and
  careful preprocessing are essential.


Deliverables
- A short report describing dataset(s), preprocessing choices, pipelines, evaluation design,
  metrics, and statistical conclusions.
- Figures: per-subject performance distribution, pipeline ranking, and significance matrix.

## Tasks
1. Dataset selection and inspection
   - With MOABB, select at least one MI dataset and optionally one ERP dataset. Programmatically
     print dataset metadata (subjects, channels, sampling rate, number of trials per class,
     sessions/runs). Justify your choice in terms of neuroscientific relevance and feasibility.
   - Define a binary mapping for classification (e.g., left vs right hand for MI; target vs non-target
     for P300). Explain why this mapping is sensible.

2. Neuroscience-driven preprocessing plan
   - MI: propose referencing (e.g., CAR), notch if needed, band-pass in the mu/beta range, and
     epoching window relative to cue (justify start and length in seconds).
   - ERP: propose high-pass/low-pass cutoffs, baseline correction, and epoch window around
     the stimulus (justify in milliseconds).
   - Explain how each choice supports the underlying neural signals.

3. Define evaluation protocol with MOABB
   - Choose an evaluation setting (e.g., cross-subject, within-session, cross-session). Argue
     how this reflects the BCI goal (generalization vs personalization).
   - Ensure that all learned components (filters, scalers, features, hyperparameters) are fit
     strictly on training folds only. Describe how MOABB enforces this.

4. Baseline pipeline: CSP + LDA (MI)
   - Implement CSP with an appropriate number of spatial filters per class and a linear
     classifier. Justify the number of filters from a bias–variance perspective.
   - Plan a nested cross-validation to tune the number of filters if needed.

5. Riemannian pipeline: Covariances + MDM (MI or ERP)
   - Compute covariance matrices on epochs and classify with MDM in the Riemannian manifold.
   - Discuss why covariance geometry captures task-relevant spatial structure.

6. Tangent Space + SVM or Logistic Regression
   - Map covariances to the tangent space and train a linear or RBF classifier. Explain when
     tangent space features may outperform raw MDM.

7. ERP-focused pipeline: Xdawn + MDM (ERP)
   - If you included an ERP dataset, add Xdawn spatial filtering before covariance estimation.
   - Explain how Xdawn enhances SNR for time-locked responses.

8. Additional option: Filter-Bank MDM (FgMDM)
   - For MI, consider a filter-bank approach to capture sub-band-specific information. State
     expected benefits and computational trade-offs.

9. Metrics and chance levels
   - Use balanced accuracy as the primary metric; report per-subject values. Complement with
     class-specific precision/recall or AUC where meaningful. State the theoretical chance level
     and include a negative control by shuffling labels to verify that pipelines collapse to chance.

10. Visualize per-subject performance
    - Create boxplots or violin plots stratified by pipeline, and bar/line plots across subjects.
    - Interpret systematic subject effects and variance across folds.

11. Multi-dataset summary (if applicable)
    - Aggregate results across datasets. Compare distributions and discuss domain shift
      (montage differences, sampling rate, artifact prevalence).

12. Statistical comparisons across pipelines
    - For each dataset, perform paired tests across pipelines on the per-subject scores
      (e.g., Wilcoxon signed-rank or paired t-test after normality checks). Compute effect sizes
      (e.g., Cohen’s d or rank-biserial). Correct for multiple comparisons (e.g., Holm or FDR).
    - Produce a significance matrix (pipelines × pipelines) and annotate which differences remain
      significant after correction.

13. Rank-based analysis (optional, multi-dataset)
    - Compute average ranks per pipeline across datasets/subjects and discuss stability.

14. Robustness checks
    - Reduce the number of channels to a motor-strip subset; re-run and compare effects.
    - Vary MI epoch length or ERP latency windows; re-run to test sensitivity.
    - Add controlled noise to epochs to test SNR sensitivity.

15. Learning curves and data efficiency
    - Plot performance vs the number of training trials per subject for at least one pipeline.
      Discuss diminishing returns and practical acquisition budgets.

16. Bias and confounds audit
    - Verify there is no temporal leakage (e.g., overlapping windows across folds). Check that
      subject IDs never cross train/test splits in cross-subject evaluation. Discuss class imbalance
      and how balanced accuracy mitigates it.

17. Report and interpretation
    - Summarize which pipelines perform best under your chosen evaluation and why. Link
      outcomes to neurophysiology (e.g., sensorimotor rhythms for MI; P3 amplitude/topography
      for ERP). Include limitations and next steps.

18. Reproducibility checklist
    - Fix random seeds, record package versions, and save configuration. Provide commands or
      scripts to reproduce all figures and statistics.

Hints
- Start from MOABB’s benchmarking tutorial (multiple pipelines) and its statistical analysis
  example; adapt parameter grids modestly to keep runtime reasonable.
- Use per-subject performance vectors as the atomic inputs to statistical tests. Avoid averaging
  across subjects before testing to preserve within-subject pairing.
- Consider reporting both significance and practical relevance (effect sizes).
