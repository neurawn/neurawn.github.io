---
title: "memmap"
excerpt: "An interactive tool for visualizing fMRI memory encoding patterns across the brain."
collection: portfolio
---

**[GitHub](https://github.com/neurawn/memmap)** &mdash; Python &middot; Nilearn &middot; Plotly &middot; Streamlit

Reading neuroimaging papers is a lot easier when you can actually *see* the brain. `memmap` is a Streamlit app that lets you explore fMRI activation patterns from memory studies — either your own data or publicly available datasets from Neurovault.

You upload a statistical map (NIfTI), pick a reference atlas, and the tool gives you an interactive glass brain, a table of peak activations by region, and a simple comparison view for up to three contrasts side by side.

**Why I built it**
I got tired of writing the same Nilearn plotting code in every analysis notebook. This wraps it into something a labmate or collaborator can actually run without knowing Python.

**Features**
- Drag-and-drop NIfTI upload
- Automatic parcellation with common atlases (Schaefer, AAL, HCP MMP)
- Side-by-side contrast comparison
- Exportable summary tables and figures
