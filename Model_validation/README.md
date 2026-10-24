The trained model is evaluated using the validation set during development and the held-out test set for final evaluation.

Accuracy measures overall correctness. Precision measures the proportion of predicted activations that are actual activations. Recall measures the proportion of actual activations correctly identified. The F1-score balances precision and recall.

ROC-AUC measures the model's ability to rank positive cases above negative cases, while average precision summarises precision-recall performance across thresholds. These metrics provide a more complete assessment than accuracy alone.

The test set is not used to fit the model. It is reserved for estimating performance on unseen accounts.
The trained model is evaluated using the validation set during development and the held-out test set for final evaluation.

Accuracy measures overall correctness. Precision measures the proportion of predicted activations that are actual activations. Recall measures the proportion of actual activations correctly identified. The F1-score balances precision and recall.

ROC-AUC measures the model's ability to rank positive cases above negative cases, while average precision summarises precision-recall performance across thresholds. These metrics provide a more complete assessment than accuracy alone.

The test set is not used to fit the model. It is reserved for estimating performance on unseen accounts.

from sklearn.metrics import average_precision_score
def evaluate_model(model, X_data, y_data, dataset_name):

    predictions = model.predict(X_data)
    probabilities = model.predict_proba(X_data)[:, 1]

    results = {
        "Dataset": dataset_name,
        "Accuracy": accuracy_score(y_data, predictions),
        "Precision": precision_score(
            y_data, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_data, predictions, zero_division=0
        ),
        "F1-score": f1_score(
            y_data, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_data, probabilities),
        "Average Precision": average_precision_score(
            y_data, probabilities
        )
    }

    return results
    # Evaluate training and validation performance
train_results = evaluate_model(
    model, X_train, y_train, "Training"
)

val_results = evaluate_model(
    model, X_val, y_val, "Validation"
)

# Evaluate the final unseen test set
test_results = evaluate_model(
    model, X_test, y_test, "Unseen Test"
)

results_df = pd.DataFrame([
    train_results,
    val_results,
    test_results
])

display(results_df.round(4))

classification report

from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Predictions for unseen test data
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Classification Report — Unseen Test Data")

print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=[
        "No First Deposit",
        "First Deposit"
    ],
    zero_division=0
))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

print("Confusion Matrix:")
print(cm)
ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Deposit", "Deposit"]
).plot(cmap="Blues", values_format="d")

plt.title("Gradient Boosting Confusion Matrix")
plt.tight_layout()
plt.show()