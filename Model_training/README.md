Gradient Boosting is selected as the final classification method
from SS2. It builds an ensemble of decision trees sequentially,
with each new tree attempting to correct errors made by previous
trees. It can identify nonlinear relationships and interactions
between customer characteristics and engagement variables.

The model uses 200 estimators to provide sufficient boosting
iterations, a learning rate of 0.05 to make gradual updates,
a maximum tree depth of 2 to limit model complexity, and a minimum
leaf size of 10 to reduce overfitting. A subsample of 0.8 adds
regularisation by training each boosting stage on a random subset
of the training data.

The random state is fixed at 42 to support reproducibility.
Preprocessing is included in the model pipeline to ensure that
imputation, scaling and encoding are fitted using training data
only.

Train Gradient boosting Model
# Calculate class weights for the training data
sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)

# Create the complete model pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=2,
        min_samples_leaf=10,
        subsample=0.8,
        random_state=RANDOM_STATE
    ))
])

# Fit the model with sample weights
model.fit(
    X_train,
    y_train,
    classifier__sample_weight=sample_weights
)

print("Gradient Boosting model trained successfully with class weights.")


