# Predictive Maintenance

This repository contains a predictive maintenance project focused on predicting equipment failures using the AI4I 2020 Predictive Maintenance Dataset. The workflow covers data ingestion, preprocessing, feature fusion, and initial model experimentation.

## Overview

Predictive maintenance helps identify potential machine failures before they happen. This project uses historical sensor and maintenance data to build a machine learning workflow for failure prediction and analysis.

## Repository structure

- data/raw/ - original dataset files
- data/processed/ - cleaned and prepared datasets
- notebooks/01_ingestion.ipynb - data loading and initial preprocessing
- notebooks/02_fusion.ipynb - feature fusion and exploratory analysis
- models/ - trained models and related artifacts

## Environment setup

1. Clone the repository.
2. Create and activate a Python environment.
3. Install the required packages:

   pip install pandas numpy matplotlib seaborn scikit-learn jupyter

## Usage

Open the notebooks in Jupyter to explore the pipeline:

```bash
jupyter notebook
```

Then run the notebooks in order:

1. notebooks/01_ingestion.ipynb
2. notebooks/02_fusion.ipynb

## Dataset

The project uses the AI4I 2020 Predictive Maintenance Dataset, which includes information about machine operating conditions and failure events.

## Current goals

- Understand the data and feature relationships
- Prepare the dataset for modeling
- Explore modeling approaches for failure prediction
- Document results and improve reproducibility

## Notes

This README will be expanded as the project grows with model performance results, experiment summaries, and deployment guidance.
