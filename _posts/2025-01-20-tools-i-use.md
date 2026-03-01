---
title: 'The tools I actually use for neuroscience research'
date: 2025-01-20
permalink: /posts/2025/01/tools-i-use/
tags:
  - tools
  - research
  - python
---

People ask me about my analysis setup fairly often, usually other grad students trying to figure out what software stack is worth learning. Here's an honest accounting of what I actually use day-to-day, not what sounds impressive.

## Data & analysis

**Python** is the lingua franca of my lab and most of computational neuroscience. The core libraries I use constantly:

- **MNE-Python** for EEG/LFP preprocessing, filtering, and time-frequency analysis. Steep learning curve, excellent documentation once you understand the data model.
- **Nilearn** for fMRI. Makes neuroimaging in Python actually pleasant.
- **NumPy/SciPy** for everything else. These are the bedrock.
- **scikit-learn** for any ML modeling that doesn't need a GPU.
- **PyTorch** when it does.

## Experiment management

**[Weights & Biases](https://wandb.ai)** for experiment tracking. Free for academics. I resisted using it for a long time and now I can't imagine going back to managing runs with hand-labeled folders.

For data versioning, I use **[DataLad](https://www.datalad.org)**, which is a git-based system for large scientific datasets. It integrates well with the [OpenNeuro](https://openneuro.org) archive and keeps your analysis reproducible without manually copying data around.

## Writing & notes

**Obsidian** for personal notes. I keep a lab notebook there with daily entries and literature notes.

**Overleaf** for papers and anything that needs to be a PDF. I know some people prefer local LaTeX setups but the real-time collaboration in Overleaf has saved me a lot of pain when writing with co-authors.

## Visualization

**Matplotlib** for publication figures (yes, still). **Plotly** and **Seaborn** for exploratory plots where aesthetics matter less than speed.

For brain visualizations specifically: **Nilearn** for volumetric/surface plots, **MNE** for EEG topographies. Both produce reasonable figures with minimal fuss.

## Version control & reproducibility

**Git** for everything. **GitHub** for collaboration and backup.

I try to structure every project as a self-contained repository with a `Makefile` or `snakemake` workflow that can reproduce the full analysis from raw data. I don't always succeed, but the aspiration is there.

---

That's the honest list. Nothing exotic — the neuroscience software ecosystem has gotten genuinely good in the last five years, and standing on the shoulders of open-source giants is a reasonable strategy.

If you're just starting out: learn Python well, get comfortable with NumPy, and pick up MNE or Nilearn depending on your modality. The rest follows.
