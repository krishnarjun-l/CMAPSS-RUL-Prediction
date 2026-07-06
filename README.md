# CMAPSS Remaining Useful Life (RUL) Prediction — MS Capstone

Predicting the **Remaining Useful Life** of aircraft turbofan engines from multivariate sensor time-series, using NASA's **C-MAPSS** run-to-failure dataset. The project combines deep-learning regression with uncertainty quantification and model explainability, plus a retrieval-augmented component for interpretability.

This was my MS in AI capstone project (CSC 675).

## Overview

Predictive maintenance aims to estimate how many operating cycles a machine has left before failure, so maintenance can be scheduled before a breakdown. Using NASA's C-MAPSS turbofan degradation simulation, this project predicts RUL from 21 sensor channels and 3 operating-condition settings recorded across each engine's life.

The pipeline covers:

- **EDA & preprocessing** — sensor selection, normalization, sliding-window sequence construction, and RUL target clipping.
- **Deep-learning RUL model** — a PyTorch sequence model for regression on the sensor windows.
- **Uncertainty quantification** — Bayesian modeling with NumPyro / JAX to produce calibrated predictive uncertainty rather than point estimates alone.
- **Explainability** — SHAP values to attribute predictions to specific sensors.
- **Retrieval-augmented component** — FAISS + sentence-transformers + LangChain to surface relevant context/insights.

## Data

The dataset is **not included** in this repository. It is NASA's public **C-MAPSS Turbofan Engine Degradation Simulation** dataset:

- https://data.nasa.gov/dataset/C-MAPSS-Aircraft-Engine-Simulator-Data
- Also mirrored on Kaggle: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps

Place the `CMAPSSData/` folder (containing `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`, etc.) alongside the notebook, or update the path at the top of the notebook.

## Repository structure

```
.
├── notebooks/
│   └── cmapss_rul_prediction.ipynb   # EDA, preprocessing, DL model, Bayesian UQ, SHAP, RAG
├── reports/
│   ├── Final_Project_Report.docx
│   ├── Final_Presentation.pptx
│   └── Proposal.docx
├── requirements.txt
└── README.md
```

## Getting started

1. Clone the repository and set up a virtual environment:

   ```bash
   git clone https://github.com/<your-username>/CMAPSS-RUL-Prediction.git
   cd CMAPSS-RUL-Prediction
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Download the C-MAPSS dataset (see [Data](#data)) and place `CMAPSSData/` where the notebook expects it.

3. Run the notebook:

   ```bash
   jupyter notebook notebooks/cmapss_rul_prediction.ipynb
   ```

## Requirements

- Python 3.10+
- PyTorch
- NumPyro, JAX (Bayesian uncertainty quantification)
- SHAP (explainability)
- FAISS, sentence-transformers, LangChain (retrieval component)
- NumPy, pandas, scikit-learn, SciPy, matplotlib, seaborn
- A GPU is recommended for training.

## Author

Krishnarjun Lakshminarayanan (Group 14) — CSC 675 Capstone Project in AI.
