import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import optuna
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# Assuming your feature set X and target y are already defined.
# For example:
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


X = train.drop(columns=['Admitted'])
y = train['Admitted']

# Split the data into training and validation sets (assuming X and y are already defined)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def objective_xgb(trial):
    params = {
        # Maximum tree depth for base learners
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        # Learning rate (eta) - use loguniform for a wide range
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        # Number of boosting rounds (n_estimators)
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        # Subsample ratio of the training instances
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        # Subsample ratio of columns when constructing each tree
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        # Minimum loss reduction required to make a further partition on a leaf node of the tree
        'gamma': trial.suggest_loguniform('gamma', 1e-8, 5.0),
        # Set a fixed random seed for reproducibility
        'random_state': 42,
        # Use binary logistic loss for classification
        'objective': 'binary:logistic'
    }
    
    model = XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return f1_score(y_val, y_pred, average='weighted')

study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(objective_xgb, n_trials=150)

print("Best XGBoost trial:")
trial_xgb = study_xgb.best_trial
print(f"  F1 Score: {trial_xgb.value}")
print("  Hyperparameters:")
for key, value in trial_xgb.params.items():
    print(f"    {key}: {value}")

# Train final XGBoost model with best parameters
best_params_xgb = trial_xgb.params
final_xgb_model = XGBClassifier(**best_params_xgb, use_label_encoder=False, eval_metric='logloss')
final_xgb_model.fit(X, y)
# Predict on test set (assuming test set is preprocessed similar to training)
test_predictions_xgb = final_xgb_model.predict(test)
# Prepare submission
submission_xgb = sample_submission.copy()
submission_xgb['Admitted'] = test_predictions_xgb
submission_xgb.to_csv('submission_xgboost.csv', index=False)
print("XGBoost submission saved as submission_xgboost.csv")