# 📚 Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier


# Load the data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sample_submission = pd.read_csv("SampleSubmission.csv")

# Fill missing values for numeric columns
numeric_cols = ['GRE Score', 'TOEFL Score', 'SOP', 'CGPA']
train[numeric_cols] = train[numeric_cols].fillna(train[numeric_cols].median())
test[numeric_cols] = test[numeric_cols].fillna(test[numeric_cols].median())

# One-hot encode the Location column
train = pd.get_dummies(train, columns=['Location'], drop_first=True)
test = pd.get_dummies(test, columns=['Location'], drop_first=True)

# Drop ID column if not needed
train = train.drop(columns=['ID'])
test = test.drop(columns=['ID'])

# Feature Engineering: Example of interaction feature
# --- Binning Age ---
bins = [train['Age'].min()-1, 25, 30, train['Age'].max()+1]
labels = ['Young', 'Mid', 'Senior']
train['Age_bin'] = pd.cut(train['Age'], bins=bins, labels=labels)
test['Age_bin'] = pd.cut(test['Age'], bins=bins, labels=labels)
train = pd.get_dummies(train, columns=['Age_bin'], drop_first=True)
test = pd.get_dummies(test, columns=['Age_bin'], drop_first=True)

# --- Quantile Binning GRE Score, TOEFL Score, CGPA ---
train['GRE_bin'] = pd.qcut(train['GRE Score'], q=4, labels=False)
test['GRE_bin'] = pd.qcut(test['GRE Score'], q=4, labels=False)
train['TOEFL_bin'] = pd.qcut(train['TOEFL Score'], q=4, labels=False)
test['TOEFL_bin'] = pd.qcut(test['TOEFL Score'], q=4, labels=False)
train['CGPA_bin'] = pd.qcut(train['CGPA'], q=4, labels=False)
test['CGPA_bin'] = pd.qcut(test['CGPA'], q=4, labels=False)

# --- Composite Academic Score ---
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
academic_features = ['GRE Score', 'TOEFL Score', 'CGPA']
train_scaled = scaler.fit_transform(train[academic_features])
test_scaled = scaler.transform(test[academic_features])
train_scaled = pd.DataFrame(train_scaled, columns=[f'{col}_scaled' for col in academic_features], index=train.index)
test_scaled = pd.DataFrame(test_scaled, columns=[f'{col}_scaled' for col in academic_features], index=test.index)
train = pd.concat([train, train_scaled], axis=1)
test = pd.concat([test, test_scaled], axis=1)
train['Academic_Score'] = train[[f'{col}_scaled' for col in academic_features]].mean(axis=1)
test['Academic_Score'] = test[[f'{col}_scaled' for col in academic_features]].mean(axis=1)

# --- Interaction Feature ---
train['GRE_TOEFL_interaction'] = train['GRE Score'] * train['TOEFL Score']
test['GRE_TOEFL_interaction'] = test['GRE Score'] * test['TOEFL Score']

# --- Outlier Flagging (Example for GRE Score) ---
train['GRE_outlier'] = ((train['GRE Score'] < train['GRE Score'].quantile(0.01)) | 
                        (train['GRE Score'] > train['GRE Score'].quantile(0.99))).astype(int)
test['GRE_outlier'] = ((test['GRE Score'] < test['GRE Score'].quantile(0.01)) | 
                       (test['GRE Score'] > test['GRE Score'].quantile(0.99))).astype(int)

# Define features and target
X = train.drop(columns=['Admitted'])
y = train['Admitted']

# Split the data
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest
from sklearn.ensemble import RandomForestClassifier
model = CatBoostClassifier(random_state=42, verbose=0)
model.fit(X_train, y_train)

# Evaluate the model
from sklearn.metrics import accuracy_score, f1_score, classification_report
y_pred = model.predict(X_val)
print("Accuracy:", accuracy_score(y_val, y_pred))
print("F1 Score:", f1_score(y_val, y_pred, average='weighted'))
print(classification_report(y_val, y_pred))


test_predictions = model.predict(test)

# Prepare submission
submission = sample_submission.copy()
submission['Admitted'] = test_predictions

# Save to CSV
submission.to_csv('submission_1.csv', index=False)
print("Submission file saved!")