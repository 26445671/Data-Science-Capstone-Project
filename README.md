
# Data-Science-Capstone-Project
## SS1
## STADIOEquities
## PART A
## Motivation
STADIOEquities is a retail investing and fintech firm, a digital investment platform on a mission to let anyone in South Africa own a piece of the market—starting with as little as the price of a cup of coffee. While traditional stockbrokers demanded thick wallets and thick paperwork, STADIOEquities lets first-time investors open an account on their phone in minutes and buy a fraction of a share.  

The retail investing and fintech industry is increasingly using digital platforms to make investing more accessible. STADIOEquities has attracted about 2.3 million registered accounts, particularly among young, first-time investors. However, 41% of these accounts have never been funded, while customer dormancy within six months has increased to 31%.  STADIOEquities struggles to convert registered customers into active, long-term investors. The registration-to-first-deposit conversion rate has declined from 64% to 59%, while onboarding abandonment and dormancy have increased. 

STADIOEquities already collects extensive data on customer behavior, funding, financials, trading, demographics, support, and compliance. However, it does not fully use this data to identify customers likely to deposit within 30 days. Current onboarding communications and nudges are largely delivered on fixed schedules rather than targeted to individual customer behavior. Moreover, STADIOEquities spends about R180 to acquire each customer, and when new customers do not become active investors, the company loses potential value from its marketing investment. This matters because customer activation and engagement directly affect STADIOEquities' revenue and the return on its customer acquisition costs.  

Therefore, a data science approach is uniquely suited to solving this challenge. Instead of treating all clients uniformly, predictive models and clustering techniques can uncover patterns and client behavior that distinguish likely activators from likely drop-offs and segment clients by investing style. Predictive analytics can identify which new registrants are most likely to fund accounts, enabling targeted onboarding interventions that reduce wasted marketing spend. 
Segmentation models (e.g., clustering) can group clients by behavior, allowing personalized nudges, product recommendations, and premium subscription offers that resonate with each profile. Churn prediction can flag accounts at risk of dormancy, enabling proactive re-engagement strategies. Natural language processing (NLP) can mine support tickets and feedback to detect recurring pain points, reducing confusion and improving client satisfaction.  

Therefore, the study aims to apply a data science approach to develop and evaluate machine learning classification models for identifying registered accounts that are most likely to remain unfunded. The study will also examine the customer characteristics and behavioral factors associated with an account remaining unfunded after registration. The resulting customer segments and predictive insights may help STADIOEquities better understand the activation gap and identify groups of customers that may require different engagement approaches. 

## Problem Statement
STADIOEquities has successfully lowered the barriers to entry for retail investing in South Africa, attracting over 2.3 million registered accounts. However, the platform faces a critical challenge: a large proportion of accounts remain unfunded, and accounts quickly lapse into dormancy. According to the data provided, currently, 41% of accounts never deposit, and 31% of funded accounts become inactive within six months. This “activation gap” undermines the company’s economic model, as its revenue streams depend on funded balances, trading activity, and long-term engagement. In addition, STADIOEquities incurs an acquisition cost of approximately R180 per account, meaning that accounts that remain unfunded provide limited opportunity to recover the cost of acquisition. 

Although the company collects extensive data on customer demographics, app and web behavior, onboarding activities, deposits, withdrawals, funding, trading activity, and marketing engagement, these data sources are not yet fully integrated to generate actionable customer-level insight. Therefore, STADIOEquities has an opportunity to use its available customer, onboarding, behavioral, and marketing data to better understand the characteristics of accounts that remain unfunded.  The lack of differentiated client experiences further compounds the issue. Despite diverse investor behaviors ranging from disciplined savers to speculative traders, all clients receive the same onboarding nudges, product suggestions, and subscription offers. This one-size-fits-all approach leads to poor conversion, declining premium subscription uptake, and rising support complaints.  

Thus, the study aims to apply a data science approach to develop and evaluate machine learning classification models to predict which registered accounts are most likely to make their first deposit within 30 days. By leveraging predictive modeling, clustering, and churn analysis, the study seeks to identify groups and characteristics associated with activation vs. remaining unfunded and provide evidence that could support differentiated onboarding and engagement. This aligns with STADIOEquities’ 2030 strategy of turning registrations into lifetime investors, driving revenue growth, reducing wasted acquisition and marketing expenditure, and advancing financial inclusion in South Africa.

## SS2
# Bank Marketing ML

This project applies machine learning to the UCI Bank Marketing dataset to predict whether a client will subscribe to a term deposit after a direct marketing campaign.The findings provide a useful methodological foundation for the proposed STADIOEquities capstone project. However, the Bank Marketing problem differs from the intended STADIOEquities objective. The Bank Marketing dataset focuses on predicting whether a customer will subscribe to a term deposit following a marketing campaign, whereas the proposed SS1 project focuses on predicting which registered accounts are likely to make their first deposit within 30 days of registration, referred to as customer activation. Consequently, the Bank Marketing dataset should be regarded primarily as a public benchmark for developing and evaluating the modelling approach, rather than as a direct representation of the STADIOEquities customer activation problem.
# The project compares two classification models:
- Logistic Regression
- Gradient Boosting

## Project structure
The workflow is organized into four stages:
1. Data preprocessing
2. Feature engineering
3. Model development
4. Model evaluation and recommendations


### Part B – Modelling

- [Preprocessing](Preprocessing.md/README.md)
- [Feature Engineering](Feature%20Engineering.md/README.md)
- [Model 1 – Logistic Regression](Model1.MD)
- [Model 2 – Gradient Boosting](Model2.MD)

### Part C – Model performance

- [Model 1 Performance](Model1Performance.MD)
- [Model 2 Performance](Model2Performance.MD)
- [Model Comparison](Comparison.MD)

### Part D – Recommendations

- [Part D Recommendations Report](Part_D_Recommendations_Report.pdf)

## Dataset

The project uses the UCI Bank Marketing dataset, which contains customer demographics, previous marketing contact history, and campaign-level features.

## Objectives

- Clean and standardize raw campaign data
- Derive meaningful features to capture customer behavior
- Train predictive models that estimate subscription probability
- Compare model performance and recommend business actions

## Requirements

The Python dependencies required to run the project are listed in `requirements.txt`.


# SS3 Part B — STADIOEquities First-Deposit Activation Prediction
# Files
- SS3_PartB.ipynb — end-to-end notebook for loading, preprocessing, training, validation and visualisation.
- stadioequities_account_activation_extract.csv — client extract supplied for this task .
- requirements.txt — Python dependencies.
# Project objective
Apply the Gradient Boosting Classifier selected as the final method to predict whether a registered account makes a first deposit within the defined 30-day activation window. The target is first_deposit_made (0 = no first deposit, 1 = first deposit).
# Dataset summary
The supplied extract contains 11,447 rows and 39 columns. Target counts are:
- No first deposit (0): 7,546 (65.92%)
- First deposit (1): 3,901 (34.08%)
The target distribution is moderately imbalanced. Therefore, precision, recall, F1, ROC-AUC and average precision are reported in addition to accuracy.
# How to run
1. Put SS3_PartB.ipynb and stadioequities_account_activation_extract.csv in the same directory.
2. Create/activate a Python environment.
3. Install dependencies:pip install -r requirements.txt
4. Open Jupyter Notebook or JupyterLab and run all cells in SS3_PartB.ipynb from top to bottom.
# Method and design choices
- Model: GradientBoostingClassifier.
- Split: 60% training, 20% validation, 20% held-out test, stratified by target.
- Preprocessing: median imputation and scaling for numeric variables; most-frequent imputation and one-hot encoding for categorical variables.
- Leakage controls: identifier, target, direct deposit outcome fields and downstream account outcomes are excluded from predictors.
- Imbalanced dataset has been treated with class weights
- Evaluation: accuracy, precision, recall, F1, ROC-AUC, average precision, classification report and confusion matrix.
- Visualisations: training/validation metric comparison, unseen-test confusion matrix, ROC curve, precision–recall curve and feature importance.
# Important limitation
The extract does not document the exact observation timestamp for every predictor. Some behavioural fields may have been collected during the same 30-day period as the target. Confirm the prediction point and remove any feature not available at that point before treating the model as a prospective prediction system. Feature importance is not causal evidence.
