from matplotlib import pyplot as plt
import mne
from mne.datasets import sample
import numpy as np
from pyriemann.classification import MDM
from pyriemann.estimation import XdawnCovariances
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline


#####################################
# Set parameters and read data
data_path = sample.data_path() / "MEG" / "sample"
raw_fname = data_path / "sample_audvis_filt-0-40_raw.fif"
event_fname = data_path / "sample_audvis_filt-0-40_raw-eve.fif"
tmin, tmax = -0.0, 1
event_id = {
    "auditory/left": 1,
    "auditory/right": 2,
    "visual/left": 3,
    "visual/right": 4,
}

# Setup for reading the raw data & high-pass
raw = mne.io.Raw(raw_fname, preload=True)
raw.filter(2, None, method="iir")
raw.info["bads"] = ["EEG 053"]  # set bad channels
events = mne.read_events(event_fname)
picks = mne.pick_types(
    raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads"
)

# Read epochs
epochs = mne.Epochs(
    raw,
    events,
    event_id,
    tmin,
    tmax,
    proj=False,
    picks=picks,
    baseline=None,
    preload=True,
    verbose=False,
)
labels = epochs.events[:, -1]
evoked = epochs.average()
epochs_data = epochs.get_data()

# Decoding with Xdawn + MDM
n_components = 3  # pick some components
cv = KFold(n_splits=10, shuffle=True, random_state=42)
pr = np.zeros(len(labels))
clf = make_pipeline(XdawnCovariances(n_components), MDM())
for train_idx, test_idx in cv.split(epochs_data):
    y_train, y_test = labels[train_idx], labels[test_idx]
    clf.fit(epochs_data[train_idx], y_train)
    pr[test_idx] = clf.predict(epochs_data[test_idx])
print(classification_report(labels, pr))

#####################################
# plot the spatial patterns, 1Hz for plotting
xd = XdawnCovariances(n_components)
xd.fit(epochs_data, labels)
info = evoked.copy().resample(1).info  # type: ignore
patterns = mne.EvokedArray(data=xd.Xd_.patterns_.T, info=info)
patterns.plot_topomap(
    times=[0, n_components, 2 * n_components, 3 * n_components],  # type: ignore
    ch_type="eeg",
    colorbar=False,
    size=1,
    time_format="Pattern %d",
)
plt.savefig("subCh5-3_Components_Xdawn_ERPs.png", dpi=600)
