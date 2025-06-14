Credit Card Default Behaviour Score Prediction

This project aims to develop a classification model to predict whether a credit card customer will default in the following month. Using financial and behavioural data from over 25,000 customers, the goal is to build a model that is not only predictive but also interpretable for risk-based decision-making.

Objective
•	Perform Exploratory Data Analysis (EDA) and data cleaning
•	Engineer financial features related to repayment and default behaviour
•	Develop and compare multiple classification models
•	Evaluate models using F2 score and other credit-relevant metrics
•	Generate predictions on unseen data and interpret business implications
Methodology
1.	Data Preparation: Read dataset, drop irrelevant columns (like Customer ID), and handle missing values.
2.	EDA and Visualization:
o	Analysed default trends and feature distributions across age, education, gender, and marital status.
o	Visualized average credit limit (LIMIT_BAL) and default counts using bar plots.
o	Identified strong signals in overdue payments (PAY_0 to PAY_6) and low repayment ratios.
3.	Feature Engineering:
•	 Created `pay_amt_total` as the sum of all six-monthly repayment amounts (`pay_amt1` to `pay_amt6`), capturing total repayment behaviour.
•	 Computed `avg_bill_amt` by averaging bill amounts over six months (`bill_amt1` to `bill_amt6`) to measure monthly exposure.
•	Derived `pay_to_bill_ratio` as the ratio of total payments to total bills, indicating repayment discipline or risk of revolving credit.
•	Used payment status fields (`PAY_0` to `PAY_6`) to identify delinquency streaks and behavioural trends such as chronic late payments.
•	Age was binned into groups for EDA purposes but kept as continuous for modelling.
•	Ensured categorical fields like `sex`, `education`, and `marriage` were label-encoded numerically for modelling compatibility.
4.	Model Training:
o	Trained Logistic Regression, Decision Tree, XGBoost, LightGBM, and Random Forest.
o	Handled class imbalance using SMOTE.
5.	Evaluation:
o	Prioritized F2 score to capture defaulters (maximize recall).
o	Final selected model: Decision Tree, due to its higher recall (0.615) and strongest F2 score (0.549) among all models. This makes it most effective at identifying likely defaulters, which is critical in credit risk applications where missing a defaulter is more costly than a false positive.

6.	Results

Model	Accuracy	Precision	Recall	F1 Score	F2 Score
Decision Tree	0.738	0.383	0.615	0.472	0.549
Logistic Regression	0.764	0.413	0.563	0.476	0.525
XGBoost	0.821	0.543	0.398	0.459	0.421
LightGBM	0.834	0.592	0.410	0.484	0.437
Random Forest	0.836	0.649	0.300	0.410	0.336

7.	Classification Cutoff
After experimenting with probability thresholds between 0.3 and 0.5, a cutoff of 0.4 was selected as it provided the best trade-off between increasing recall (catching more actual defaulters) and keeping precision at a practical level. This is crucial in credit risk where false negatives—missed defaulters—pose a higher financial threat than false positives.

8.	Summary
•	From the visual analysis of customer segments, we observed that:
o	Customers aged 41–60 had the highest average credit limits, while younger users (21–30) showed a higher count of defaults.
o	Males and married individuals received higher credit limits but also exhibited marginally higher default volumes.
o	Those with graduate or university-level education were granted higher limits, yet contributed more to defaults, possibly due to higher exposure.
•	Among all tested models, the Decision Tree classifier was selected for its superior recall (0.615) and F2 score (0.549), making it the most effective model at identifying defaulters.
•	Although ensemble models had better accuracy and precision, they lagged in recall, which is crucial for credit risk tasks. The Decision Tree also offered transparency in feature importance and decision paths.
•	A classification cutoff of 0.4 was selected to optimize recall without overly compromising precision, prioritizing the minimization of false negatives (missed defaulters).
•	Overall, the project demonstrates that a balance between predictive power and business interpretability can be achieved using well-engineered features and recall-optimized modelling.
•	Socio-demographic variables help further segment risk.
•	Decision Tree was ultimately chosen for its ability to maximize recall and F2 score, aligning well with the business goal of minimizing undetected credit risk. Its interpretability also aids risk analysts in understanding key drivers behind the model's decisions.

Dependencies
•	pandas, NumPy, seaborn, matplotlib
•	scikit-learn, imbalanced-learn, XGboost, LightGBM

