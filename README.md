# Analysis of Bank Marketing Dataset

This repository contains an analysis of a dataset derived from direct marketing campaigns conducted by a Portuguese banking institution. The dataset includes client data and outcomes of marketing campaigns.

## Dataset Investigation
### Background : 
The dataset consists of 41188 examples with 20 input features. The classification goal is to predict whether a client will subscribe to a term deposit.

### Explanation of Variables
The dataset includes various input variables such as age, job, marital status, education, and output variable 'deposit' indicating if the client subscribed to a term deposit.

### Type Transformation and Missing Values
Missing values coded as "Unknown" are removed from the dataset. Categorical variables are transformed into factor levels.


## Variables Investigation
Numeric variables are explored using histograms. Categorical variables are examined for balance between groups.


## Resampling
Downsampling is employed to address class imbalance within the 'deposit' target variable.


## Models
Models such as Logistic Regression, K-Nearest Neighbors (KNN), Random Forest, Neural Network, and Gradient Boosting Machine are trained and evaluated using various metrics including Misclassification Rate (MCR), Accuracy, Precision, Recall, and F1 Score.


## Conclusion
Logistic Regression emerges as the top-performing model, displaying the highest predictive accuracy. Economic indicators like 'euribor3m' are identified as key predictors influencing term deposit subscriptions. Resampling techniques like downsampling enhance predictive performance.
