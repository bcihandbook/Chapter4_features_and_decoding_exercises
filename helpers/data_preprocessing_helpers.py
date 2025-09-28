"""
Data preprocessing helpers for BCI feature selection
===================================================

This module contains functions for loading and preprocessing EEG data
from the MNE eegbci dataset for motor imagery tasks.

Author: BCI Handbook Chapter 4.3 - Feature Selection Exercise
License: BSD (3-clause)
"""

import numpy as np
from mne import Epochs, pick_types
from mne.io import concatenate_raws
from mne.io.edf import read_raw_edf
from mne.datasets import eegbci
from mne import events_from_annotations
from mne.channels import make_standard_montage


def load_motor_imagery_data(
    subject=1, runs=None, tmin=1.0, tmax=2.0, filter_freq=(8.0, 35.0), verbose=False
):
    """
    Load motor imagery EEG data from the eegbci dataset.

    Parameters
    ----------
    subject : int
        Subject number (1-109)
    runs : list of int or None
        Run numbers to load. If None, loads runs [4, 6, 8, 10, 12, 14]
        (motor imagery: left vs right hand)
    tmin : float
        Start time relative to event onset (seconds)
    tmax : float
        End time relative to event onset (seconds)
    filter_freq : tuple
        Low and high frequency for bandpass filter (Hz)
    verbose : bool
        Whether to print verbose output

    Returns
    -------
    X : ndarray, shape (n_epochs, n_channels, n_times)
        Epoch data
    y : ndarray, shape (n_epochs,)
        Class labels (0 for left hand, 1 for right hand)
    epochs : Epochs
        MNE Epochs object
    """
    if runs is None:
        runs = [4, 6, 8, 10, 12, 14]  # Motor imagery runs (left vs right hand)

    if verbose:
        print(f"Loading EEG data for subject {subject}, runs: {runs}")
        print(f"Time window: {tmin} to {tmax} seconds")
        print(f"Filter: {filter_freq[0]}-{filter_freq[1]} Hz")

    # Load raw files
    raw_files = eegbci.load_data(subject, runs, update_path=False)
    raw_list = [read_raw_edf(f, preload=True, verbose="ERROR") for f in raw_files]
    raw = concatenate_raws(raw_list)

    # Standardize channel names and set montage
    eegbci.standardize(raw)
    montage = make_standard_montage("standard_1005")
    raw.set_montage(montage, verbose="ERROR")

    # Apply bandpass filter
    raw.filter(
        filter_freq[0],
        filter_freq[1],
        method="iir",
        picks="eeg",
        iir_params=dict(order=6, ftype="butter"),
        verbose="ERROR",
    )

    # Extract events and create event dictionary
    events, event_id = events_from_annotations(raw, verbose="ERROR")

    # Map to binary classification: T1 (left hand) vs T2 (right hand)
    if "T1" in event_id and "T2" in event_id:
        event_dict = {"left_hand": event_id["T1"], "right_hand": event_id["T2"]}
    else:
        raise ValueError("T1 and T2 events not found in the data")

    # Pick EEG channels only
    picks = pick_types(
        raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads"
    )

    # Create epochs
    epochs = Epochs(
        raw,
        events,
        event_dict,
        tmin,
        tmax,
        proj=False,
        picks=picks,
        baseline=None,
        preload=True,
        verbose="ERROR",
    )

    # Get data and labels
    X = epochs.get_data()
    y = epochs.events[:, -1]

    # Convert to binary labels: 0 for left hand, 1 for right hand
    left_label = event_dict["left_hand"]
    y = (y != left_label).astype(int)

    if verbose:
        print(f"Data shape: {X.shape}")
        print(f"Number of channels: {X.shape[1]}")
        print(f"Number of time samples: {X.shape[2]}")
        print(f"Sample rate: {epochs.info['sfreq']} Hz")
        print(f"Class distribution: {np.bincount(y)}")

    return X, y, epochs


def balance_classes(X, y, method="undersample", random_state=42):
    """
    Balance class distribution in the dataset.

    Parameters
    ----------
    X : ndarray
        Feature data
    y : ndarray
        Labels
    method : str
        Balancing method ('undersample', 'oversample')
    random_state : int
        Random seed

    Returns
    -------
    X_balanced : ndarray
        Balanced feature data
    y_balanced : ndarray
        Balanced labels
    """
    np.random.seed(random_state)

    unique_classes, class_counts = np.unique(y, return_counts=True)

    if method == "undersample":
        # Undersample to minority class size
        min_count = np.min(class_counts)
        balanced_indices = []

        for cls in unique_classes:
            cls_indices = np.where(y == cls)[0]
            selected_indices = np.random.choice(cls_indices, min_count, replace=False)
            balanced_indices.extend(selected_indices)

        balanced_indices = np.array(balanced_indices)
        np.random.shuffle(balanced_indices)

        return X[balanced_indices], y[balanced_indices]

    elif method == "oversample":
        # Oversample to majority class size
        max_count = np.max(class_counts)
        balanced_indices = []

        for cls in unique_classes:
            cls_indices = np.where(y == cls)[0]
            if len(cls_indices) < max_count:
                # Oversample this class
                selected_indices = np.random.choice(cls_indices, max_count, replace=True)
            else:
                selected_indices = cls_indices
            balanced_indices.extend(selected_indices)

        balanced_indices = np.array(balanced_indices)
        np.random.shuffle(balanced_indices)

        return X[balanced_indices], y[balanced_indices]

    else:
        raise ValueError("Method must be 'undersample' or 'oversample'")
