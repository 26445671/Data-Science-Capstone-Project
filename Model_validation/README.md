The trained model is evaluated using the validation set during development and the held-out test set for final evaluation.

Accuracy measures overall correctness. Precision measures the proportion of predicted activations that are actual activations. Recall measures the proportion of actual activations correctly identified. The F1-score balances precision and recall.

ROC-AUC measures the model's ability to rank positive cases above negative cases, while average precision summarises precision-recall performance across thresholds. These metrics provide a more complete assessment than accuracy alone.

The test set is not used to fit the model. It is reserved for estimating performance on unseen accounts.
