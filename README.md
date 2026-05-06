<div align="center">
  
# 📈 End-to-End Time Series Forecasting System

**A production-ready forecasting engine with automated model selection and a FastAPI REST interface.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-00a393.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

---

## 🚀 Overview

This repository contains a complete, end-to-end machine learning pipeline for time series forecasting. It was designed to predict weekly sales across 43 US states. 

The system automatically trains four different types of algorithms (Statistical, Machine Learning, and Deep Learning) per state, evaluates them on a chronological holdout validation set, selects the best-performing model based on lowest RMSE, and serves the predictions through a production-grade REST API.

### ✨ Key Features
- **Automated Model Selection:** Competes SARIMA, Prophet, XGBoost, and LSTM against each other per-state.
- **Robust Feature Engineering:** Generates sliding lags ($t-1, t-7, t-30$), rolling statistics, and calendar/holiday features.
- **Leakage-Free Validation:** Strict chronological train/validation splitting ensures true predictive performance.
- **RESTful Serving:** High-performance API built with FastAPI and Pydantic v2.
- **Stateless Inference:** Models and scalers are serialized to disk, allowing the API to run inference instantly on startup.

---

## 🏗️ Architecture & Stack

### Technology Stack
- **Core:** Pandas, NumPy, Scikit-Learn
- **Models:** Statsmodels (SARIMA), Prophet, XGBoost, TensorFlow/Keras (LSTM)
- **API:** FastAPI, Uvicorn, Pydantic
- **EDA:** Jupyter, Matplotlib, Seaborn

### Directory Structure
```text
forecasting_system/
├── api/
│   ├── main.py               # FastAPI application & lifespan manager
│   ├── routes.py             # REST endpoints (/forecast, /models, etc.)
│   └── schemas.py            # Pydantic validation models
├── artifacts/
│   ├── best_models.json      # Auto-generated model selection report
│   └── models/               # Serialized models and scalers per state
├── notebooks/
│   └── analysis.ipynb        # Exploratory Data Analysis & visual walkthrough
├── src/
│   ├── data_loader.py        # Dependency-free Excel parser
│   ├── preprocessor.py       # Feature engineering & chronological splitting
│   ├── trainer.py            # Training orchestrator
│   ├── forecaster.py         # CLI inference engine
│   └── models/               # Individual model implementations
├── setup.py                  # CLI entry-point for training pipeline
└── requirements.txt          # Project dependencies
```

---

## 📊 Model Competition & Results

The system evaluates four distinct algorithmic approaches:
1. **SARIMA** – Baseline statistical model with AIC-based grid search.
2. **Prophet** – Additive model handling yearly/weekly seasonality and US holidays.
3. **XGBoost** – Gradient boosted trees utilizing engineered lag and rolling window features via recursive multi-step forecasting.
4. **LSTM** – Deep learning sequence model with MinMax scaling and early stopping.

### Evaluation Criteria
Models are evaluated on a strict **16-week chronological holdout set**. The system calculates Root Mean Squared Error (RMSE) and Mean Absolute Percentage Error (MAPE), automatically persisting the model with the lowest RMSE.

*In our latest run across 43 states, **XGBoost** won 33 states, and **LSTM** won 10 states, achieving a highly accurate average MAPE of **0.71%** across the entire dataset.*

---

## 🛠️ Installation & Setup

### 1. Clone & Environment Setup
```bash
git clone https://github.com/yourusername/time-series-forecasting.git
cd time-series-forecasting
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Mac users: If XGBoost fails to load, you may need to run `brew install libomp`)*

### 3. Run the Training Pipeline
Ensure your dataset is placed in the parent directory (`../Copy of Forecasting Case- Study.xlsx`).

```bash
# Train all models for all states (~45-60 mins)
python setup.py

# Quick smoke test (1 state, skips LSTM)
python setup.py --state California --skip-lstm --fast
```

---

## 🌐 REST API Usage

Once training is complete, spin up the API server:

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

The API will automatically load the best models from the `artifacts/` directory into memory during startup.

### Interactive Documentation
Navigate to **[http://localhost:8000/docs](http://localhost:8000/docs)** to use the Swagger UI.

### Example API Calls

**Get 8-week forecast for a state (GET)**
```bash
curl http://localhost:8000/api/v1/forecast/Texas
```

**Get custom forecast length (POST)**
```bash
curl -X POST http://localhost:8000/api/v1/forecast \
  -H "Content-Type: application/json" \
  -d '{"state": "California", "weeks": 4}'
```

**Check which model won in each state (GET)**
```bash
curl http://localhost:8000/api/v1/models
```

---

## 📓 Exploratory Data Analysis

A comprehensive Jupyter Notebook is provided containing:
- Target distribution and trend analysis
- Seasonal decomposition
- Feature matrix visualizations
- Forecast vs Actual plot comparisons

Launch the notebook to view the charts:
```bash
jupyter notebook notebooks/analysis.ipynb
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
