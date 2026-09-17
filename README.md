# 💳 Fraud Detection - Machine Learning Models

An end-to-end Machine Learning project to detect fraudulent financial transactions from mobile payment data with high precision and recall, addressing extreme class imbalance (0.129% fraud).

---

## 📌 Models Evaluated

The dataset consists of **636,262 transactions** with extreme class imbalance (**821 fraud** vs **635,441 legitimate**). 

The following algorithms were trained and benchmarked using **Stratified 5-Fold Cross-Validation**:

1. **Baseline / Dummy Classifier**: Stratified random predictor for baseline calibration.
2. **Logistic Regression**: Linear baseline with class weighting.
3. **Support Vector Machine (SVM)**: Linear support vector classifier.
4. **K-Nearest Neighbors (KNN)**: Distance-based local neighborhood classifier.
5. **LightGBM**: Fast gradient boosting framework.
6. **Random Forest Classifier**: Ensemble of bagged decision trees.
7. **XGBoost Classifier**: Regularized gradient boosted decision trees (**Selected Champion**).

---

## 📊 Model Comparison & Cross-Validation Results

All models were evaluated across **5 stratified folds** on imbalanced data using **Balanced Accuracy**, **Precision**, **Recall**, **F1-Score**, and **Cohen's Kappa**:

### Stratified 5-Fold Cross-Validation Performance

| Model | Balanced Accuracy | Precision | Recall | F1-Score | Cohen's Kappa |
|---|:---:|:---:|:---:|:---:|:---:|
| **Dummy** | 0.499 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | -0.001 ± 0.000 |
| **Logistic Regression** | 0.565 ± 0.009 | **1.000 ± 0.000** | 0.129 ± 0.017 | 0.229 ± 0.027 | 0.228 ± 0.027 |
| **LightGBM** | 0.701 ± 0.089 | 0.180 ± 0.100 | 0.407 ± 0.175 | 0.241 ± 0.128 | 0.239 ± 0.129 |
| **SVM** | 0.595 ± 0.013 | **1.000 ± 0.000** | 0.190 ± 0.026 | 0.319 ± 0.037 | 0.319 ± 0.037 |
| **K-Nearest Neighbors** | 0.705 ± 0.037 | 0.942 ± 0.022 | 0.409 ± 0.074 | 0.568 ± 0.073 | 0.567 ± 0.073 |
| **Random Forest** | 0.865 ± 0.017 | 0.972 ± 0.014 | 0.731 ± 0.033 | 0.834 ± 0.022 | 0.833 ± 0.022 |
| **XGBoost** 🏆 | **0.880 ± 0.016** | **0.963 ± 0.008** | **0.761 ± 0.033** | **0.850 ± 0.023** | **0.850 ± 0.023** |

---

## ⚙️ Hyperparameter Fine-Tuning

**XGBoost** demonstrated the best balance of precision, recall, and computational efficiency. Fine-tuning was performed using `GridSearchCV` with **Stratified 5-Fold Cross-Validation** optimized for **F1-score**:

```python
param_grid = {
    'booster': ['gbtree', 'gblinear', 'dart'],
    'eta': [0.3, 0.1, 0.01],
    'scale_pos_weight': [1, 774, 508, 99]
}
```

### Optimal Parameters Found:
- **`booster`**: `'gbtree'`
- **`eta` (learning rate)**: `0.3`
- **`scale_pos_weight`**: `1`
- **CV Best F1-Score**: `0.864`

---

## 🎯 Final Model Performance (Unseen Test Data)

The fine-tuned XGBoost model was evaluated against a completely independent holdout test set (20% of data):

| Metric | Score | Note |
|---|:---:|---|
| **Balanced Accuracy** | **0.915** | Robust discrimination between minority (fraud) and majority classes |
| **Precision** | **0.944** | 94.4% of flagged transactions are confirmed fraud |
| **Recall** | **0.829** | Successfully detects ~83% of all fraudulent transactions |
| **F1-Score** | **0.883** | Harmonic mean reflecting high reliability |
| **Cohen's Kappa** | **0.883** | High agreement beyond chance |

---

## 🧬 Model Features (Boruta Selection)

Using the **Boruta algorithm**, the feature space was filtered to the 7 most relevant predictors:

1. **`step`**: Transaction simulation hour (1 to 744).
2. **`oldbalance_org`**: Initial balance of the origin account before transaction.
3. **`newbalance_orig`**: Final balance of the origin account after transaction.
4. **`newbalance_dest`**: Final balance of the recipient account after transaction.
5. **`diff_new_old_balance`**: Difference between initial and final balance in the origin account (`newbalance_orig - oldbalance_org`).
6. **`diff_new_old_destiny`**: Difference between initial and final balance in the recipient account (`newbalance_dest - oldbalance_dest`).
7. **`type_TRANSFER`**: Binary flag indicating transfer-type transactions.

---

## 🚀 Model Inference & API Usage

The trained model and transformers are saved under:
- Model: `models/model_cycle1.joblib`
- Scaler: `functions/minmaxscaler_cycle1.joblib`
- Encoder: `functions/onehotencoder_cycle1.joblib`

### Running the API:
```bash
cd api
python handler.py
```

### Making a Prediction Request:
```bash
curl -X POST http://127.0.0.1:5000/fraud/predict \
     -H "Content-Type: application/json" \
     -d '{
           "step": 283,
           "type": "CASH_OUT",
           "amount": 215489.19,
           "nameOrig": "C123456789",
           "oldbalanceOrg": 215489.19,
           "newbalanceOrig": 0.0,
           "nameDest": "M987654321",
           "oldbalanceDest": 0.0,
           "newbalanceDest": 215489.19
         }'
```

### Response:
```json
[
  {
    "step": 283,
    "type": "CASH_OUT",
    "amount": 215489.19,
    "nameOrig": "C123456789",
    "oldbalanceOrg": 215489.19,
    "newbalanceOrig": 0.0,
    "nameDest": "M987654321",
    "oldbalanceDest": 0.0,
    "newbalanceDest": 215489.19,
    "prediction": 1
  }
]
```
