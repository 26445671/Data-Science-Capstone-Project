============================================================
LOGISTIC REGRESSION RESULTS
============================================================
Accuracy:  0.7564
Precision: 0.2682
Recall:    0.6257
F1 Score:  0.3754
ROC-AUC:   0.7701

Confusion Matrix:
[[9267 2710]
 [ 594  993]]

Classification Report:
                 precision    recall  f1-score   support

No Subscription       0.94      0.77      0.85     11977
   Subscription       0.27      0.63      0.38      1587

       accuracy                           0.76     13564
      macro avg       0.60      0.70      0.61     13564
   weighted avg       0.86      0.76      0.79     13564
   
# Model 1 Performance – Logistic Regression

## Objective

The objective of Model 1 was to predict whether a customer would subscribe to a term deposit using Logistic Regression.

The target variable was encoded as:

* `0` = No subscription
* `1` = Subscription

The model was evaluated on a stratified 20% test dataset.

## Evaluation Metrics

The following metrics were used to evaluate the model:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion matrix
* Classification report

These metrics were selected because the Bank Marketing dataset contains an imbalanced target variable. Therefore, accuracy alone does not provide a complete assessment of the model's ability to identify customers who subscribed.

## Model Performance

| Metric    | Result |
| --------- | -----: |
| Accuracy  | 0.7673 |
| Precision | 0.2792 |
| Recall    | 0.6257 |
| F1-score  | 0.3862 |
| ROC-AUC   | 0.7737 |

### Interpretation

The Logistic Regression model achieved an accuracy of **76.73%**.

The precision for the subscription class was **27.92%**, meaning that approximately 28% of customers predicted as subscribers actually subscribed.

The recall was **62.57%**, indicating that the model identified approximately 63% of the customers who actually subscribed.

The F1-score was **0.3862**, representing the balance between precision and recall for the subscription class.

The ROC-AUC was **0.7737**, indicating that the model provides useful discrimination between customers who subscribed and those who did not.

## Confusion Matrix

The confusion matrix was:

```text
[[9414, 2563],
 [ 594,  993]]
```

The results can be interpreted as:

|            | Predicted No | Predicted Yes |
| ---------- | -----------: | ------------: |
| Actual No  |        9,414 |         2,563 |
| Actual Yes |          594 |           993 |

Therefore:

* True Negatives = 9,414
* False Positives = 2,563
* False Negatives = 594
* True Positives = 993

The model correctly identified **993 customers who subscribed** and missed **594 customers who subscribed**.

## Classification Report

The classification report provides precision, recall and F1-score for both classes.

The subscription class achieved:

* Precision = 0.2792
* Recall = 0.6257
* F1-score = 0.3862

The relatively lower precision compared with recall indicates that the model identifies a reasonable proportion of actual subscribers, but also classifies a number of non-subscribers as potential subscribers.

## Statistical Measures

ROC-AUC was used as an additional performance measure because it evaluates the model's ability to distinguish between the two classes across different classification thresholds.

The ROC-AUC value of **0.7737** indicates that the model has useful discriminatory ability.

The confusion matrix was also used to examine the types of classification errors made by the model, particularly false positives and false negatives.

## Related Code

The performance calculations are included in:

[model1_logistic_regression.ipynb](../scripts/model1_logistic_regression.py)

## How to Run

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Model 1 script from the repository root:

```bash
python scripts/model1_logistic_regression.py
```

The script trains the Logistic Regression model, generates predictions and calculates the evaluation metrics presented above.
