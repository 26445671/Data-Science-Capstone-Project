### visualisations

Contains graphs and visualisations created during the project.

Examples include:

- Training versus validation and test performance
- Feature distributions
- Correlation plots
- Confusion matrices
- ROC curves
- Feature importance plots

Training versus validation and test performance
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

 ROC curve
 from sklearn.metrics import RocCurveDisplay

RocCurveDisplay.from_predictions(
    y_test,
    y_prob
)

plt.title("ROC Curve — Unseen Test Data")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

Precision-recall curve
from sklearn.metrics import PrecisionRecallDisplay

PrecisionRecallDisplay.from_predictions(
    y_test,
    y_prob
)

plt.title("Precision-Recall Curve — Unseen Test Data")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

 Feature importance
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
