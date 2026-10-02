MODEL1.md
# Model 1 – Logistic Regression

## Model Description

Logistic Regression was selected as Model 1 because it is a widely used classification algorithm for predicting binary outcomes.

The model predicts the probability that a customer will subscribe to a term deposit.

## Hyperparameters

The following parameters were used:

| Parameter | Value |
|---|---|
| C | 1.0 |
| Solver | liblinear |
| Maximum iterations | 1000 |
| Class weight | balanced |
| Random state | 42 |

## Class Imbalance

The dataset contains substantially more `no` observations than `yes` observations.

Therefore:

```python
class_weight="balanced"

