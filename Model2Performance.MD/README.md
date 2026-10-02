============================================================
GRADIENT BOOSTING RESULTS
============================================================
Accuracy:  0.7965
Precision: 0.3189
Recall:    0.6509
F1 Score:  0.4281
ROC-AUC:   0.7960

Confusion Matrix:
[[9771 2206]
 [ 554 1033]]

Classification Report:
                 precision    recall  f1-score   support

No Subscription       0.95      0.82      0.88     11977
   Subscription       0.32      0.65      0.43      1587

       accuracy                           0.80     13564
      macro avg       0.63      0.73      0.65     13564
   weighted avg       0.87      0.80      0.82     13564

   # Model 2 Performance – Gradient Boosting

## Objective

The objective of Model 2 was to predict whether a customer would subscribe to a term deposit using Gradient Boosting.

The target variable was encoded as:

* `0` = No subscription
* `1` = Subscription

The model was evaluated using the same stratified 20% test dataset approach used for Model 1.

## Evaluation Metrics

The following metrics were used:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion matrix
* Classification report

These measures were selected because the target variable is imbalanced and therefore accuracy alone is not sufficient to evaluate the model's ability to identify subscribers.

## Model Performance

| Metric    | Result |
| --------- | -----: |
| Accuracy  | 0.7959 |
| Precision | 0.3183 |
| Recall    | 0.6515 |
| F1-score  | 0.4276 |
| ROC-AUC   | 0.7951 |

### Interpretation

The Gradient Boosting model achieved an accuracy of **79.59%**.

The precision for the subscription class was **31.83%**, meaning that approximately 32% of customers predicted as subscribers actually subscribed.

The recall was **65.15%**, indicating that the model identified approximately 65% of customers who actually subscribed.

The F1-score was **0.4276**, showing an improvement in the balance between precision and recall compared with Logistic Regression.

The ROC-AUC was **0.7951**, indicating useful discriminatory performance between the two target classes.

## Confusion Matrix

The confusion matrix was:

```text
[[9762, 2215],
 [ 553, 1034]]
```

The results can be interpreted as:

|            | Predicted No | Predicted Yes |
| ---------- | -----------: | ------------: |
| Actual No  |        9,762 |         2,215 |
| Actual Yes |          553 |         1,034 |

Therefore:

* True Negatives = 9,762
* False Positives = 2,215
* False Negatives = 553
* True Positives = 1,034

The model correctly identified **1,034 customers who subscribed** and missed **553 customers who subscribed**.

## Classification Report

For the subscription class, the model achieved:

* Precision = 0.3183
* Recall = 0.6515
* F1-score = 0.4276

The recall indicates that the model identified a substantial proportion of actual subscribers, while the precision shows that some customers predicted as subscribers did not actually subscribe.

## Statistical Measures

ROC-AUC was used to evaluate the discriminatory ability of the model across different classification thresholds.

The ROC-AUC value of **0.7951** indicates that the model provides useful separation between customers who subscribed and those who did not.

The confusion matrix was also examined to understand false positives and false negatives.

## How to Run

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Model 2 script from the repository root:

```bash
python scripts/model2_gradient_boosting.py
```

The script trains the Gradient Boosting model, generates predictions and calculates the evaluation metrics presented above.

