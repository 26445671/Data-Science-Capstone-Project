### visualisations

Contains graphs and visualisations created during the project.

Visualisations are used to compare training, validation and unseen
test performance. The metric comparison graph helps identify
possible overfitting. The confusion matrix illustrates correct
and incorrect classifications, while the ROC and precision-recall
curves show how model performance changes across classification
thresholds.

# 5.1 Training versus validation and test performance
metrics_to_plot = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-score",
    "ROC-AUC",
    "Average Precision"
]

plot_data = results_df.set_index("Dataset")[metrics_to_plot]

ax = plot_data.T.plot(
    kind="bar",
    figsize=(12, 6)
)

ax.set_title("Gradient Boosting Performance Comparison")
ax.set_ylabel("Score")
ax.set_xlabel("Evaluation Metric")
ax.set_ylim(0, 1)
ax.legend(title="Dataset")
ax.grid(axis="y", alpha=0.3)

plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.show()
# 5.2 ROC curve
from sklearn.metrics import RocCurveDisplay

RocCurveDisplay.from_predictions(
    y_test,
    y_prob
)

plt.title("ROC Curve — Unseen Test Data")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
# 5.3 Precision-recall curve
from sklearn.metrics import PrecisionRecallDisplay

PrecisionRecallDisplay.from_predictions(
    y_test,
    y_prob
)

plt.title("Precision-Recall Curve — Unseen Test Data")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
# 6.4 Feature importance
# Retrieve the fitted preprocessing and classification steps
fitted_preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

# Recover feature names after encoding
feature_names = fitted_preprocessor.get_feature_names_out()

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": classifier.feature_importances_
})

# Display the 15 most important features
top_features = importance_df.sort_values(
    by="Importance",
    ascending=False
).head(15)

display(top_features)

# Plot feature importance
top_features = top_features.sort_values("Importance")

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.title("Top 15 Feature Importances")
plt.xlabel("Importance Score")
plt.ylabel("Feature")

plt.tight_layout()
plt.show()

# Interpretation of Classification Results for STADIOEquities

## 1. Introduction

The classification model was developed to predict whether a registered STADIOEquities account is likely to make its first deposit. This supports the business objective of identifying customers who are likely to activate their accounts and distinguishing them from those who have not yet funded their accounts.

STADIOEquities has a large registered customer base, but a significant proportion of accounts remain unfunded. The model can help the business identify potential customers for targeted onboarding communication and activation campaigns. The classification report and confusion matrix below show how well the model performed on unseen test data.

## 2. Interpretation of the classification report

| Metric             | No First Deposit | First Deposit |
| ------------------ | ---------------- | ------------- |
| Precision          | 83%              | 49%           |
| Recall             | 59%              | 77%           |
| F1-score           | 69%              | 60%           |
| Support (accounts) | 1,506            | 778           |

No First Deposit: The model achieved a precision of 83%, meaning that when it predicted an account would not make a first deposit, it was correct approximately 83% of the time. However, its recall was 59%, indicating that it correctly identified 887 of the 1,506 accounts that did not make a first deposit. Consequently, the model missed 619 accounts in this category by incorrectly predicting that they would make a deposit.

From a business perspective, these missed accounts could represent customers who need additional onboarding support or targeted activation campaigns.

First Deposit: The model achieved a precision of 49%, meaning that 49% of accounts predicted to make a first deposit actually belonged to that class in the test data. Its recall was 77%, indicating that it successfully identified 598 of the 778 accounts that made a first deposit. This is useful because the model identifies a relatively large proportion of the accounts in the positive class.

However, the lower precision means that many accounts flagged as likely to make a first deposit were incorrectly classified. If STADIOEquities uses these predictions to target customers with marketing campaigns, some resources may be spent on accounts that do not subsequently activate.

F1-score: The model achieved an F1-score of 69% for accounts with no first deposit and 60% for accounts that made a first deposit. These scores indicate that the model has a better balance between precision and recall for the no-first-deposit class than for the first-deposit class. The first-deposit predictions require further improvement, particularly in reducing false positives.

Accuracy: The overall accuracy was 65%, meaning that the model correctly classified 1,485 out of 2,284 test accounts. Although this provides a general indication of performance, accuracy alone is insufficient to evaluate the model because STADIOEquities needs to understand the different business consequences of missed activations and incorrect activation predictions.

## 3. Interpretation of the confusion matrix

The confusion matrix is:

Confusion Matrix — Unseen Test Data

Rows = actual class · Columns = predicted class

Predicted: No Deposit

Predicted: First Deposit

Actual: No Deposit

# 887

Correct predictions

# 619

Missed no-deposit accounts

Actual: First Deposit

# 180

Missed first deposits

# 598

Correct first-deposit predictions

The confusion matrix provides four important findings:

1. 887 true negatives: The model correctly predicted that 887 accounts would not make a first deposit.
2. 619 false positives: The model predicted that 619 accounts would make a first deposit when they belonged to the no-first-deposit class. This may lead to inefficient targeting if these predictions are used to allocate activation resources.
3. 180 false negatives: The model failed to identify 180 accounts that made a first deposit. These accounts may be overlooked if the business uses the model to prioritise customers for activation-related interventions.
4. 598 true positives: The model correctly identified 598 accounts in the first-deposit class, demonstrating its ability to recognise a substantial proportion of this group.

The confusion matrix therefore shows that the model identifies first-deposit accounts reasonably well in terms of recall, but it also produces a considerable number of false-positive predictions.

## 4. Business implications for STADIOEquities

The results suggest that the model could provide useful initial support for STADIOEquities' account activation strategy. Its 77% recall for the first-deposit class means it identifies a large proportion of accounts belonging to that class in the test data.

However, the model should not yet be relied upon as a standalone decision-making tool. Its 49% precision for the first-deposit class indicates that more than half of the accounts predicted to belong to that class were incorrectly classified.

STADIOEquities could use the predictions to support customer segmentation and prioritisation, subject to further validation. For example, the business could investigate which customer characteristics and onboarding behaviours are associated with first-deposit activity, then test whether tailored communication improves activation rates.

Before operational use, the model should also be evaluated for probability calibration, performance across relevant customer segments, and the business costs of false-positive and false-negative predictions. Importantly, a classification prediction is not evidence that a marketing intervention will cause a customer to deposit.

## 5. Results visualisation

For the Results Visualisation section of the notebook, include the following graphs.

[Evaluation Metrics for Classification 
Figure 1: Confusion matrix heatmap

Visualises the 887, 619, 180 and 598 predictions. This makes it easy to compare correct classifications with the two types of error.

Figure 2: Precision, recall and F1-score comparison

Compares the performance of the two classes and highlights the higher recall but lower precision for first-deposit accounts.


Figure 3: ROC curve

Evaluates how well the model distinguishes the two classes across different classification thresholds. Generate this graph using the model's predicted probabilities.

## 6. Overall conclusion

The model achieved 65% accuracy on unseen test data, with a recall of 77% for the first-deposit class. This indicates that the model can identify a substantial proportion of accounts belonging to that class, which could be useful for STADIOEquities' account activation analysis. Nevertheless, the precision of 49% and the 619 false-positive predictions indicate that the model requires further refinement.

Future improvements could include tuning the classification threshold, testing class-weighting or resampling methods where appropriate, comparing alternative models, and evaluating precision-recall trade-offs. The final model should be selected based on STADIOEquities' business objective and the relative costs of different prediction errors, rather than accuracy alone.

Important project distinction: These results evaluate the model's ability to classify first-deposit outcomes. They do not, by themselves, demonstrate that the model will increase deposits or reduce customer dormancy. Those outcomes would require separate business evaluation.
