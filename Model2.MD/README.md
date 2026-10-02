MODEL2.MD
# Model 2 – Gradient Boosting

## Model Description

Gradient Boosting was selected as Model 2 because it is an ensemble machine learning technique that combines multiple decision trees to improve predictive performance.

## Hyperparameters

The model was configured using:

| Parameter | Value |
|---|---|
| Number of estimators | 200 |
| Learning rate | 0.05 |
| Maximum depth | 3 |
| Minimum samples split | 10 |
| Minimum samples leaf | 5 |
| Subsample | 0.8 |
| Random state | 42 |

## Class Imbalance

GradientBoostingClassifier does not directly provide a `class_weight` parameter.

Therefore, balanced sample weights were calculated using:

```python
compute_sample_weight(
    class_weight="balanced",
    y=y_train
)


---

# 12. Step 10 — Create `model2_gradient_boosting.py`


```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight

gb_model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    min_samples_split=10,
    min_samples_leaf=5,
    subsample=0.8,
    random_state=42
)

sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)

gb_model.fit(
    X_train,
    y_train,
    classifier__sample_weight=sample_weights
)

y_pred_gb = gb_model.predict(X_test)

y_prob_gb = gb_model.predict_proba(X_test)[:, 1]

print("Gradient Boosting training completed.")