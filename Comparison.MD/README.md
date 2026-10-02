# Model Comparison – Logistic Regression vs Gradient Boosting

## Objective

The purpose of this analysis was to compare the performance of Model 1, Logistic Regression, and Model 2, Gradient Boosting, when predicting whether a customer would subscribe to a term deposit.

Both models were evaluated using the same Bank Marketing dataset and a stratified 80:20 train-test split.

## Performance Comparison

| Evaluation Metric | Logistic Regression | Gradient Boosting |
| ----------------- | ------------------: | ----------------: |
| Accuracy          |              0.7673 |        **0.7959** |
| Precision         |              0.2792 |        **0.3183** |
| Recall            |              0.6257 |        **0.6515** |
| F1-score          |              0.3862 |        **0.4276** |
| ROC-AUC           |              0.7737 |        **0.7951** |
| True Negatives    |               9,414 |         **9,762** |
| False Positives   |               2,563 |         **2,215** |
| False Negatives   |                 594 |           **553** |
| True Positives    |                 993 |         **1,034** |

## Accuracy Comparison

Logistic Regression achieved an accuracy of **76.73%**, while Gradient Boosting achieved **79.59%**.

The difference is:

**79.59% − 76.73% = 2.86 percentage points**

Therefore, Gradient Boosting produced a higher overall accuracy on the test dataset.

## Precision Comparison

Logistic Regression achieved a precision of **27.92%**, compared with **31.83%** for Gradient Boosting.

Gradient Boosting therefore produced fewer false positive classifications relative to its positive predictions.

## Recall Comparison

Logistic Regression achieved a recall of **62.57%**, while Gradient Boosting achieved **65.15%**.

This means that Gradient Boosting identified a larger proportion of customers who actually subscribed.

Gradient Boosting identified **1,034 true positive cases**, compared with **993** for Logistic Regression.

## F1-score Comparison

The F1-score for Logistic Regression was **0.3862**, while Gradient Boosting achieved **0.4276**.

The higher F1-score indicates that Gradient Boosting provided a better balance between precision and recall for the subscription class on this test set.

## ROC-AUC Comparison

The Logistic Regression model achieved a ROC-AUC of **0.7737**, compared with **0.7951** for Gradient Boosting.

The difference is:

**0.7951 − 0.7737 = 0.0214**

Gradient Boosting therefore demonstrated higher discriminatory performance on the test data according to ROC-AUC.

## Confusion Matrix Comparison

Logistic Regression produced:

```text
[[9414, 2563],
 [ 594,  993]]
```

Gradient Boosting produced:

```text
[[9762, 2215],
 [ 553, 1034]]
```

Gradient Boosting produced:

* 348 more true negatives
* 348 fewer false positives
* 41 more true positives
* 41 fewer false negatives

This indicates that Gradient Boosting made fewer errors in both the negative and positive classes for this test set.

## Overall Comparison

Across the evaluation metrics used in this project, Gradient Boosting produced higher values than Logistic Regression.

The differences were:

| Metric    | Difference |
| --------- | ---------: |
| Accuracy  |    +0.0286 |
| Precision |    +0.0391 |
| Recall    |    +0.0258 |
| F1-score  |    +0.0414 |
| ROC-AUC   |    +0.0214 |

The results indicate that Gradient Boosting achieved stronger predictive performance on this test dataset across the selected evaluation measures.

However, model selection should consider the intended business objective. For example, if identifying as many actual subscribers as possible is the priority, recall is particularly relevant. If reducing unnecessary customer targeting is more important, precision and false positives become more important.

## Statistical and Evaluation Methods

The comparison was based on:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion matrices
* True positives
* True negatives
* False positives
* False negatives

Because the dataset contains an imbalanced target, accuracy was not considered in isolation. Precision, recall, F1-score and ROC-AUC were also considered when comparing the models.

No formal statistical significance test was performed between the two classifiers. The comparison therefore describes differences observed on the held-out test set rather than claiming that the performance differences are statistically significant.

## Related Code

The model comparison calculations are contained in:

[model_comparison.py](../scripts/model_comparison.py)

## How to Run

Install the required packages:

```bash
pip install -r requirements.txt
```

Run:

```bash
python scripts/model_comparison.py
```

The script loads the model predictions and calculates the comparative performance metrics.
