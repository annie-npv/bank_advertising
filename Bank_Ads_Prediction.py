# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample


# Read the bank data file
bank_data = pd.read_csv("bank-additional-full.csv", sep=';')

# Change the columns name
bank_data.rename(columns={'loan': 'personal', 'y': 'deposit'}, inplace=True)

# Drop the 'month' and 'day_of_week' columns
bank_data.drop(['month', 'day_of_week'], axis=1, inplace=True)

# Check for missing values in the entire data frame
print(bank_data.isna().sum())

# View data
print(bank_data.head())

# Examine missing data
unknown_percentages = (bank_data == 'unknown').mean() * 100
print(unknown_percentages)

# Drop all "unknown" data
bank_data_clean = bank_data[(bank_data != 'unknown').all(axis=1)]

# Convert categorical variables to factors
categorical_cols = ['job', 'marital', 'default', 'housing', 'personal', 'contact', 'poutcome', 'deposit']
for col in categorical_cols:
    bank_data_clean[col] = bank_data_clean[col].astype('category')

# Convert ordinal variable 'education' to ordered factor with custom levels
education_order = ["illiterate", "basic.4y", "basic.6y", "basic.9y", "high.school", "professional.course", "university.degree"]
bank_data_clean['education'] = pd.Categorical(bank_data_clean['education'], categories=education_order, ordered=True)

# Extract numeric variables from the bank_data_clean dataset
numeric_data = bank_data_clean.select_dtypes(include=['int64', 'float64'])

# Convert target variable to binary numeric
bank_data_clean['deposit_numeric'] = (bank_data_clean['deposit'] == 'yes').astype(int)

# Calculate the correlation matrix
bank_corr = numeric_data.corr()

# Plot the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(bank_corr, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Plot of Numeric Variables and Deposit')
plt.show()

# Drop columns 'emp.var.rate', 'nr.employed', and 'duration'
bank_data_clean.drop(['emp.var.rate', 'nr.employed', 'duration'], axis=1, inplace=True)

# Calculate Cramér's V for factor variables
from scipy.stats import chi2_contingency

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    return np.sqrt(chi2 / (n * (min(confusion_matrix.shape) - 1)))

factor_cols = bank_data_clean.select_dtypes(include='category').columns
cramer_matrix = pd.DataFrame(np.zeros((len(factor_cols), len(factor_cols))), index=factor_cols, columns=factor_cols)

for col1 in factor_cols:
    for col2 in factor_cols:
        if col1 != col2:
            cramer_matrix.loc[col1, col2] = cramers_v(bank_data_clean[col1], bank_data_clean[col2])

print(cramer_matrix)

# Remove columns 'poutcome' and 'default'
bank_data_clean.drop(['poutcome', 'default'], axis=1, inplace=True)

# Plot frequency of 'deposit' values
sns.countplot(x='deposit', data=bank_data_clean)
plt.title('Frequency of each Term Deposit Status')
plt.show()

# Plot percentage of 'deposit' values
bank_data_clean['deposit'].value_counts().plot.pie(autopct='%1.1f%%', colors=['skyblue', 'lightgreen'], startangle=90)
plt.title('Percentage of each Term Deposit Status')
plt.ylabel('')
plt.show()

# Downsampling
X = bank_data_clean.drop(['deposit', 'deposit_numeric'], axis=1)
y = bank_data_clean['deposit_numeric']

# Separate majority and minority classes
bank_majority = bank_data_clean[bank_data_clean['deposit'] == 'no']
bank_minority = bank_data_clean[bank_data_clean['deposit'] == 'yes']

# Downsample the majority class
bank_majority_downsampled = resample(bank_majority,
                                     replace=False,                     # sample without replacement
                                     n_samples=len(bank_minority),     # match minority class
                                     random_state=10)                  # for reproducible results

# Combine downsampled majority class with minority class
bank_downsampled = pd.concat([bank_majority_downsampled, bank_minority], axis=0)

# Display new class counts
print(bank_downsampled['deposit'].value_counts())

X_resampled = bank_downsampled.drop(['deposit', 'deposit_numeric'], axis=1)
y_resampled = bank_downsampled['deposit_numeric']

# Convert categorical variables to numeric codes
for col in X_resampled.select_dtypes(include='category').columns:
    X_resampled[col] = X_resampled[col].cat.codes

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=51)

# Logistic Regression
log_model = LogisticRegression(max_iter=10000)
log_model.fit(X_train, y_train)

# K-Nearest Neighbors (KNN)
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)

# Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Neural Network
nn_model = MLPClassifier(max_iter=1000)
nn_model.fit(X_train, y_train)

# Gradient Boosting Machine
gbm_model = GradientBoostingClassifier()
gbm_model.fit(X_train, y_train)

# Predictions
log_pred = log_model.predict(X_test)
knn_pred = knn_model.predict(X_test)
rf_pred = rf_model.predict(X_test)
nn_pred = nn_model.predict(X_test)
gbm_pred = gbm_model.predict(X_test)

# Calculate metrics
def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return accuracy, precision, recall, f1

metrics = {
    'Logistic Regression': calculate_metrics(y_test, log_pred),
    'KNN': calculate_metrics(y_test, knn_pred),
    'Random Forest': calculate_metrics(y_test, rf_pred),
    'Neural Network': calculate_metrics(y_test, nn_pred),
    'Gradient Boosting Machine': calculate_metrics(y_test, gbm_pred)
}

metrics_df = pd.DataFrame(metrics, index=['Accuracy', 'Precision', 'Recall', 'F1 Score'])
print(metrics_df)

# Plot ROC curves
def plot_roc_curve(y_true, y_pred_prob, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')

plt.figure(figsize=(10, 8))
plot_roc_curve(y_test, log_model.predict_proba(X_test)[:, 1], 'Logistic Regression')
plot_roc_curve(y_test, knn_model.predict_proba(X_test)[:, 1], 'KNN')
plot_roc_curve(y_test, rf_model.predict_proba(X_test)[:, 1], 'Random Forest')
plot_roc_curve(y_test, nn_model.predict_proba(X_test)[:, 1], 'Neural Network')
plot_roc_curve(y_test, gbm_model.predict_proba(X_test)[:, 1], 'Gradient Boosting Machine')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.show()

# Variable importance for Gradient Boosting Machine
feature_importance = pd.Series(gbm_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance, y=feature_importance.index)
plt.title('Variable Importance - Gradient Boosting Machine')
plt.show()

# Variable importance for Logistic Regression
feature_importance_log = pd.Series(np.abs(log_model.coef_[0]), index=X_train.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance_log, y=feature_importance_log.index)
plt.title('Variable Importance - Logistic Regression')
plt.show()

# Plot histogram of 'euribor3m' by deposit status
sns.histplot(data=bank_data_clean, x='euribor3m', hue='deposit_numeric', multiple='stack', binwidth=1)
plt.title('Distribution of Euribor 3-month Rate by Deposit Status')
plt.show()
