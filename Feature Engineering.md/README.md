# Feature Engineering

## Objective

Feature engineering aims to improve the predictive power of the bank marketing model by creating variables that reflect customer behavior, campaign characteristics, and engagement patterns. These derived features can help the model separate likely subscribers from non-subscribers more effectively.

Feature Engineering
Implementation

Feature engineering is implemented in:

/scripts/feature_engineering.py

Engineered Features
## Previously Contacted

A binary variable called previously_contacted is created from pdays. A value of 0 indicates that the customer was not previously contacted, while a value of 1 indicates that the customer had been contacted previously.

## Total Contacts

total_contacts combines the current campaign contacts and previous campaign contacts.

## Has Any Loan

has_any_loan indicates whether the customer has either a housing loan or personal loan.

## Previous Campaign Success

previous_success indicates whether the customer's previous marketing campaign was successful.

## Age Group

Customers are grouped into age categories to allow the models to capture potential differences in subscription behaviour across age groups.
Age is often nonlinear in relation to subscription likelihood. A simple numeric age value may not capture this pattern well, so it is useful to create age bins such as:
- 18–29
- 30–44
- 45–59
- 60+

This helps capture behavior differences between younger and older customers.

Purpose

The feature engineering process creates additional variables that represent customer contact history, financial commitments and previous campaign behaviour. These features may provide additional predictive information beyond the original variables.

The raw bank marketing dataset contains many useful but underused signals. Some variables are only meaningful in combination, while others are highly skewed or not directly interpretable. Feature engineering turns these raw indicators into more actionable inputs for machine learning algorithms.

