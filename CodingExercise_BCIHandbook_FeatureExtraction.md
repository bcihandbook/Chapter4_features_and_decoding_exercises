# Coding Exercise: Feature Extraction

## Goals
This exercise is designed to:
- Understand the importance of spatial filtering techniques in EEG signal processing
- Learn how to extract relevant features using Riemannian geometry approaches
- Implement and evaluate Xdawn spatial filtering for event-related potentials
- Visualize and interpret spatial patterns extracted from EEG data
- Apply Minimum Distance to Mean (MDM) classification in the Riemannian space

## Description
This exercise focuses on spatial filtering techniques for EEG-based Brain-Computer Interfaces (BCIs), specifically using the Xdawn algorithm to enhance event-related potentials (ERPs). You will work with multi-channel EEG data from a classic auditory and visual stimulation paradigm.

EEG Dataset Details:
- Paradigm: Auditory and visual stimulation
- Channels: 59 EEG channels + 1 EOG channel
- Sampling Rate: 600 Hz
- Conditions:
  - Auditory/Left: 1
  - Auditory/Right: 2
  - Visual/Left: 3
  - Visual/Right: 4
- Epoch Duration: 1 second (0 to 1s post-stimulus)
- Filtering: 2 Hz high-pass filter applied

Experimental Setup:
The experiment involves presenting auditory tones and visual stimuli to the left and right sides while recording EEG activity. This creates distinct spatial patterns that can be exploited for BCI control. The Xdawn algorithm is particularly effective for enhancing signal-to-noise ratio in ERP-based BCIs.

## Tasks

1. Load the sample EEG dataset and apply appropriate preprocessing steps including high-pass filtering and bad channel rejection. Examine the data structure and understand the experimental conditions.

2. Create epochs around the stimulus events for each condition. Visualize the average ERPs for auditory and visual conditions to understand the spatial distribution of brain responses.

3. Implement the Xdawn algorithm to extract spatial filters that maximize the signal-to-noise ratio. Experiment with different numbers of components and observe how this affects classification performance.

4. Plot the spatial patterns obtained from the Xdawn filters. Interpret these patterns in terms of the underlying neural processes for auditory vs. visual stimulation.

5. Analyze the spatial patterns extracted by Xdawn. Can you identify which patterns are more relevant for auditory vs. visual stimulation? How do these patterns relate to the known neurophysiology of sensory processing?
