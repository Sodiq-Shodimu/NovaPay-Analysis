# NovaPay Fraud Detection

I built an end-to-end machine learning fraud detection project that covered data preparation, feature engineering, model development, evaluation, and explainability.

## Project Summary

- I analyzed 10,780 transactions with a 9.10% fraud rate.
- I used a chronological 80/20 train-test split to reduce leakage risk.
- I trained and compared Logistic Regression, Random Forest, XGBoost, and LightGBM models.
- I used SHAP to explain model behavior and top fraud drivers.

## Best Model Performance

The Random Forest model performed best on the test set:

| Metric | Score |
|--------|-------|
| Precision | 100% |
| Recall | 92% |
| F1-Score | 96% |
| ROC-AUC | 0.975 |

## Business Impact

- I achieved 100% precision, which avoided false-positive customer blocks in the test set.
- I detected 92% of fraud cases, which improved risk coverage.
- I delivered a strong ROC-AUC of 0.975, showing reliable fraud discrimination.

## End-to-End Scope Delivered

1. Data loading and quality checks
2. Data cleaning and sanity validation
3. EDA and feature engineering
4. Model training and comparison
5. Hyperparameter tuning
6. SHAP explainability
7. Model export for reuse

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, XGBoost, LightGBM, SHAP, Matplotlib, Seaborn

## Repository Files

- `NovaPay.ipynb` - Full end-to-end notebook workflow
- `nova_pay_combined.csv` - Source dataset
- `rf_model.joblib` - Trained Random Forest model
- `shap_explainer_rf.joblib` - SHAP explainer artifact
