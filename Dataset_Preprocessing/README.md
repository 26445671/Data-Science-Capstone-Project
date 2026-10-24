The STADIOEquities dataset was prepared to ensure that the data was clean, consistent, and suitable for training a machine learning classification model to predict whether a registered account would make its first deposit within 30 days.

First, records with missing target values were removed because supervised learning requires known outcomes, and the target variable was converted to an integer representing the two classes: 0 (No First Deposit) and 1 (First Deposit). Duplicate account records were removed to prevent repeated observations from distorting the model's learning process. Registration and data extraction dates were converted into numerical calendar features, including year, month, and day of the week, allowing the model to identify potential temporal patterns.

Next, identifiers and variables that could reveal the outcome, such as account IDs, first deposit amounts, deposit counts, and days to first deposit, were excluded to reduce target leakage and ensure that predictions relied on relevant information available before activation. Columns containing only missing values were also removed because they provide no useful information. The remaining features were separated into numerical and categorical variables to support appropriate preprocessing, such as imputation and encoding.

The dataset was then divided into three subsets using Scikit-learn's train_test_split function and a fixed random state of 42 to ensure reproducibility. The first split allocated 60% of the data to training and 40% to a temporary dataset, which was subsequently divided equally into 20% validation and 20% testing data. Stratified sampling was applied to preserve approximately the same proportions of activated and non-activated accounts across all three subsets.

Finally, the number of records and class proportions in each subset were printed to verify the split and assess potential class imbalance. The training set is used to fit the model, the validation set supports model selection and parameter tuning, and the test set provides an independent evaluation of performance on unseen data. Overall, this preparation improves data quality, reduces leakage risks, supports reproducible experiments, and establishes a reliable foundation for predicting first-deposit activation at STADIOEquities.
Implement data cleaning and feature preparation
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

# Remove records with a missing target
df = df.dropna(subset=[target]).copy()
df[target] = df[target].astype(int)

# Remove duplicate account records
if "account_id" in df.columns:
    df = df.drop_duplicates(
        subset=["account_id"], keep="first"
    ).copy()
else:
    df = df.drop_duplicates().copy()

# Convert dates into numerical calendar features
for col in ["registration_date", "data_extraction_date"]:
    if col in df.columns:
        dates = pd.to_datetime(df[col], errors="coerce")

        df[f"{col}_year"] = dates.dt.year
        df[f"{col}_month"] = dates.dt.month
        df[f"{col}_dayofweek"] = dates.dt.dayofweek

        df.drop(columns=[col], inplace=True)

# Exclude identifiers and variables that could reveal the outcome
excluded_columns = [
    "account_id",
    target,
    "days_to_first_deposit",
    "first_deposit_amount",
    "total_deposits_30_days",
    "number_of_deposits_30_days",
    "withdrawal_made",
    "trading_activity",
    "premium_subscription",
    "account_status"
]

X = df.drop(
    columns=[c for c in excluded_columns if c in df.columns]
)

y = df[target]

# Remove columns that contain no usable values
empty_columns = X.columns[X.isna().all()].tolist()
X = X.drop(columns=empty_columns)

print("Features available:", X.shape[1])
print("Target records:", len(y))

# Identify numerical and categorical variables
numeric_features = X.select_dtypes(
    include=["number", "bool"]
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=["number", "bool"]
).columns.tolist()

print("Numerical features:", numeric_features)
print("Categorical features:", categorical_features)

# First split: 60% training, 40% temporary
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.40,
    random_state=RANDOM_STATE,
    stratify=y
)

# Second split: 20% validation, 20% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=y_temp
)

print("\nTraining records:", len(X_train))
print("Validation records:", len(X_val))
print("Testing records:", len(X_test))

print("\nTraining class proportions:")
print(y_train.value_counts(normalize=True).sort_index().round(3))

print("\nValidation class proportions:")
print(y_val.value_counts(normalize=True).sort_index().round(3))

print("\nTesting class proportions:")
print(y_test.value_counts(normalize=True).sort_index().round(3))
