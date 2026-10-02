# Preprocessing

# Data Preprocessing

## Dataset

The project uses the UCI Bank Marketing `bank-full.csv` dataset. The dataset contains 45,211 observations and 17 variables. The target variable `y` indicates whether the customer subscribed to a term deposit.

## Preprocessing Script

The preprocessing implementation is contained in:

`/scripts/preprocessing.py`

## Data Quality

The uploaded dataset contains no missing values and no duplicate records. Nevertheless, data-quality checks are included in the preprocessing script to ensure that these conditions are verified programmatically.

## Target Encoding

The target variable is converted from:

* `no` → `0`
* `yes` → `1`

## Data Leakage

The `duration` variable is removed because it represents the duration of the telephone call and is only known after the customer interaction. Including this variable could result in data leakage.

## Train-Test Split

The dataset is divided into:

* 70% training data
* 30% testing data

A stratified split with `random_state=42` is used to preserve the proportion of the target classes in both datasets.

## Categorical Variables

Categorical variables are converted into numerical representations using one-hot encoding.

## Numerical Variables

Numerical variables are standardised for Logistic Regression. Gradient Boosting does not require standardisation because it is a tree-based algorithm.

## Class Imbalance

The target variable is imbalanced, with approximately 88.3% of observations belonging to the `no` class and 11.7% belonging to the `yes` class.

Logistic Regression uses `class_weight='balanced'`. Gradient Boosting uses balanced sample weights during training.

## How to Run

Install the required packages:

`pip install -r requirements.txt`

Run:

`python scripts/preprocessing.py`

## Final preprocessing pipeline

A typical pipeline is:

1. Load raw dataset
2. Remove duplicates and clean invalid values
3. Encode categorical variables
4. Impute missing values
5. Scale continuous variables when required
6. Split into train/test sets
7. Train model on transformed data

## Deliverable

The preprocessing stage produces a clean, model-ready dataset that supports reliable evaluation of the classification models and allows fair comparison between them.

