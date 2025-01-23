# Prometheus

## Setting Up Environment
Use Python 3.11
```bash
pip install -r requirements.txt
```

---

## Project Overview

Prometheus focuses on analyzing relationships between securities and building predictive models for time-series data. This project leverages both statistical methods and machine learning models to uncover insights and make accurate forecasts.

---

## Methods and Models

### **1. Statistical Analysis**
- **Multivariate Linear Regression**: Evaluates relationships between multiple variables.
- **Granger Causality Tests**: Explores causal relationships between time-series.
- **Cointegration Analysis**: Tests long-term equilibrium relationships.
- **Dynamic Time Warping (DTW)**: Measures similarity between time-series.

### **Initial (Deprecated) Predictive Models**
- **AutoARIMA**: Statistical time-series forecasting with automated tuning.
- **Deep Learning Models**: Implements advanced models such as NHITS and NBEATS for forecasting.
- **Baseline Methods**: Includes benchmarks like linear regression and horizontal line predictions for comparison.

---

## Machine Learning Architectures Attempted
### - **Encoder-Decoder Transformer**
- Implements an encoder-decoder architecture with Transformer layers for sequence-to-sequence forecasting.
- Features:
    - Triple Positional Encoding: Combines embeddings for feature types, tickers, and time steps.
    - Supports multi-head attention mechanisms and layer normalizations.
    - Uses auxiliary losses to improve generalization and stability.

### - **StockBERT**
- A specialized variant of BERT adapted for financial data.
- Features:
    - Multi-dimensional token embeddings for tickers, time indices, and continuous features.
    - Leverages transformer encoder layers for masked language modeling (MLM) tasks.
    - Integrates positional and temporal embeddings (e.g., yearly, monthly, and hourly).

### - **Somoformer**
- A novel architecture that uses Transformer encoders tailored to handle sequential financial data.
- Features:
    - Incorporates triple positional encodings.
    - Adjusts output values using post-processing techniques, including mean-offset fixes.
    - Designed for sequence-to-sequence learning with flexible input-output sizes.

### - **Variational Autoencoder (VAE)**
- Captures latent representations of time-series data for reconstruction and forecasting.
- Features:
    - Encoder-decoder design utilizing convolutional layers for spatial embeddings.
    - Supports both smaller and larger model versions, with optional Discrete Cosine Transform (DCT) encoding.
    - Implements KL divergence losses for latent regularization.

### **3. Statistical and Spatial-Temporal Deep Learning**
Statistical methods like **Dynamic Time Warping (DTW)** and causal inference (e.g., Granger causality) were coupled with deep learning architectures for additional layers of interpretability and adaptability to financial time series.

## Future Directions
- Statistical Analysis of dataset to identify explainable key relationships and patterns.
- Polynomial regression analysis of Stock Relationships to identify broad relationships between securities for further analysis.
- Further experimentation with hybrid statistical and machine learning pipelines (using Trees and Deep learning).
- Implementation of ensemble approaches combining multiple models.
- Updated Evaluation against additional datasets and real-world use cases.