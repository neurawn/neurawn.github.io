---
title: "ripple-detect"
excerpt: "A Python library for detecting hippocampal sharp-wave ripples in LFP recordings."
collection: portfolio
---

**[GitHub](https://github.com/neurawn/ripple-detect)** &mdash; Python &middot; NumPy &middot; SciPy &middot; MNE

Sharp-wave ripples (SWRs) are high-frequency oscillations (~80–140 Hz) in the hippocampus that are thought to play a key role in memory consolidation during sleep and quiet wakefulness. Detecting them reliably from local field potential (LFP) recordings is a surprisingly annoying preprocessing problem.

`ripple-detect` is a lightweight Python library that wraps common detection approaches (envelope thresholding, wavelet filtering, multi-unit activity co-occurrence) into a clean, composable API. It supports data in MNE-Python and NumPy formats.

**Features**
- Multiple detection algorithms with a unified interface
- Configurable thresholds with built-in sanity-check plots
- Works with both rodent electrophysiology and human iEEG data
- Fully tested, with examples on public NWB datasets

This started as a helper script in my lab's analysis pipeline and grew into something more general when I kept rewriting it for every new dataset.
