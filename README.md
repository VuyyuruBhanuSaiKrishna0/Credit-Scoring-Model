# Credit Scoring Model
## Objective
Demonstrate a classification workflow that predicts a synthetic creditworthiness label from income, debt, payment history, age, credit-history length, and employment type.

## Run
```bash
pip install -r requirements.txt
python main.py
```

## Methods and evaluation
- Logistic Regression and Random Forest
- Train/test split with stratification
- Accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrix
- Preprocessing pipeline: imputation, scaling, and one-hot encoding

## Important limitation
The data and labels are synthetic and intended only to demonstrate the workflow. Do not use this model for real lending, eligibility, or financial decisions. For a real project, use a legally obtained, documented dataset and assess fairness, privacy, calibration, and applicable laws.
