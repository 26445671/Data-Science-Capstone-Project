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


