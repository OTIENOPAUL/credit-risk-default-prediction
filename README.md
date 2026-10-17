# Home Credit Default Risk Prediction

A machine learning project for estimating the risk of payment difficulties on consumer loan applications. The work uses the Home Credit Default Risk competition data and includes a notebook, a Streamlit app, and saved model artifacts.

## Dataset

Download the competition files from [Kaggle: Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data). A Kaggle account and acceptance of the competition rules may be required. Place the downloaded CSV files in a `data/` directory before running the notebook or any workflow that reads the source data.

The main training target is `TARGET` in `application_train.csv` (`1` indicates payment difficulties; `0` indicates no recorded difficulties). The project may use additional competition tables for applicant history and features.

## Run the app

Install dependencies and start Streamlit:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app loads `credit_risk_lgb_model.pkl` and `model_features.pkl` from the project root.

## Project files

- `Credit_Default_Risk.ipynb` — analysis and modeling notebook
- `app.py` — Streamlit scoring interface
- `credit_risk_lgb_model.pkl`, `model_features.pkl` — saved model and feature schema
- `data/` — downloaded Kaggle data (not included in this repository)

## Live demo

<a href="https://your-streamlit-app-url.streamlit.app" target="_blank" class="btn btn-primary">
  🚀 Launch Live Underwriting Simulator
</a>
