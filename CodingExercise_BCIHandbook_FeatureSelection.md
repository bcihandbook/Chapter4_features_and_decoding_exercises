# Coding Exercise: Feature Selection and Ranking


## Goals
This exercise is designed to
- Understand the importance of feature selection in high-dimensional EEG data analysis
- Learn how to identify the most discriminative features for BCI classification
- Implement ensemble methods for robust feature importance estimation
- Evaluate the impact of feature selection on classification performance
- Apply proper cross-validation strategies to avoid overfitting

## Description
This exercise focuses on feature selection techniques applied to functional connectivity features extracted from EEG data. You will work with connectivity matrices derived from multi-channel EEG recordings and learn how to identify the most informative connections for BCI classification.

EEG Dataset Details:
- Paradigm: Multi-channel EEG with synthetic connectivity patterns
- Channels: 22 EEG channels
- Sampling Rate: 250 Hz
- Epoch Duration: 4 seconds
- Features: Functional connectivity matrices (coherence)
- Classification: Binary classification task

Experimental Setup:
The exercise uses synthetic EEG data with known connectivity patterns to demonstrate feature selection principles. The connectivity features represent functional relationships between different brain regions, and the goal is to identify which connections are most informative for the classification task.

## Tasks
1. Implement a connectivity-based feature extraction pipeline using coherence measures. Understand how to transform raw EEG data into connectivity matrices suitable for feature selection.

2. Use ensemble methods (Random Forest) to compute feature importance scores for connectivity features. Interpret the resulting importance rankings in terms of which brain connections are most discriminative.

3. Implement feature selection by selecting the top-k most important features. Compare classification performance with different numbers of selected features.

4. Implement proper cross-validation ensuring that feature selection is performed within each fold to avoid data leakage. Compare results with improper (outer) feature selection.

5. Evaluate the impact of feature selection on classification accuracy, precision, and recall. Create plots showing how performance metrics change with different numbers of selected features.

6. Perform statistical tests to determine whether the selected features provide significantly better performance than random feature subsets. Use permutation testing to assess feature importance reliability.

7. Investigate potential biases in feature selection, including:
   - The effect of different cross-validation strategies
   - The impact of feature selection on training vs. testing set independence
   - How feature selection might introduce overfitting in small datasets

8. Create visualizations of the selected connectivity features. Interpret which brain regions or connections are most important for the classification task and relate this to the known experimental paradigm.

9. Compare different feature selection approaches (filter methods, wrapper methods, embedded methods) and evaluate their performance on the connectivity features.

10. Test how well the selected features generalize to new data by using different train/test splits and evaluating performance consistency.
