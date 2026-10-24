Load the STADIOEquities dataset.
Inspect columns, data types, missing values and target distribution
The dataset provided by the client contains 11,447 instances (rows) and 39 features (columns).
Each row represents a unique customer account activation record, with attributes such as account_id, registration_date, region, investment_goal, risk_appetite, kyc_completed, first_deposit_made, and others.

Classes (Target Variable):
The relevant classification column/target variable is first_deposit_made, where 0 represents an account that did not make a first deposit within the defined activation window and 1 represents an account that did

Class Distribution:
The dataset is imbalanced because the first deposit_made is 3901, while no first deposit_made is 7546

Imbalance Proportion:
The imbalance proportion is Majority class 65.92 % and Minority class 34.08 % , this is a moderate imbalance . Hence we will apply class weights during model training
