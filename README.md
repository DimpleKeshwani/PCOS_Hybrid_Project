# PCOS Check: Explainable PCOS Risk Screening

A web app that estimates how closely a person's symptoms match PCOS patterns,
explains each result with SHAP, and helps find nearby doctors.

> **Screening aid only. This is not a medical diagnosis.**

**Live demo:** https://pcos-hybrid-project.onrender.com/

## Features
- Multi-page site: Home, About PCOS, Screening, Result, Find Doctors
- Random Forest model trained on 541 patient records
- Optional ultrasound (follicle count) model for higher accuracy
- Per-person explanation of the result using SHAP
- Nearby hospital and clinic finder using OpenStreetMap
- Deployed on Render

## Model comparison (5-fold cross-validation, 9 symptom features)

| Model | Accuracy | PCOS recall | AUC |
|---|---|---|---|
| Logistic Regression | 0.836 | 0.820 | 0.879 |
| Random Forest | 0.813 | 0.757 | 0.881 |
| XGBoost | 0.819 | 0.740 | 0.862 |
| Gradient Boosting | 0.815 | 0.661 | 0.861 |

Adding ultrasound follicle counts raised accuracy to about 0.90.
Recall matters most for screening, because a missed case costs more than a false alarm.

## Tech stack
Python, Flask, scikit-learn, pandas, SHAP, HTML/CSS/JavaScript, Leaflet, Gunicorn, Render

## Run it locally
```
git clone https://github.com/DimpleKeshwani/PCOS-Check.git
cd PCOS-Check
pip install -r requirements.txt
cd python_backend
python app.py
```
Open http://127.0.0.1:5000

## Retrain the model
Download the dataset from
https://www.kaggle.com/datasets/prasoonkottarathil/polycystic-ovary-syndrome-pcos,
export the `Full_new` sheet as `python_backend/pcos_data.csv`, then run
`python train_model.py`.

## Limitations
- Small dataset: 541 women from 10 hospitals in Kerala, India
- Results may not generalise to other populations
- The risk percentage is not a calibrated medical probability
- No validation on an outside dataset yet
- The Low, Moderate and High cut-offs (30% and 60%) are my own choices, not clinical thresholds

## Project structure
```
python_backend/
├── app.py            Flask routes and prediction API
├── train_model.py    trains and saves the model
├── compare_models.py compares four algorithms
├── static/           style.css
└── templates/        HTML pages
```
