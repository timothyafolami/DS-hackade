import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
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
train[numeric_cols] = train[numeric_cols].fillna(train[numeric_cols].min())
test[numeric_cols] = test[numeric_cols].fillna(test[numeric_cols].min())

# One-hot encode the Location column
train = pd.get_dummies(train, columns=['Location'], drop_first=True)
test = pd.get_dummies(test, columns=['Location'], drop_first=True)

# Drop ID column if not needed
train = train.drop(columns=['ID'])
test = test.drop(columns=['ID'])

# Feature Engineering: Example of interaction feature

# --- Composite Academic Score ---
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


X = train.drop(columns=['Admitted'])
y = train['Admitted']

# Split the data into training and validation sets (assuming X and y are already defined)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def objective_cat(trial):
    params = {
        # 'depth' controls the tree depth (analogous to max_depth in other libraries)
        'depth': trial.suggest_int('depth', 3, 10),
        # learning_rate: try a loguniform distribution between 0.01 and 0.3
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        # Number of boosting rounds (iterations)
        'iterations': trial.suggest_int('iterations', 50, 300),
        # L2 regularization coefficient
        'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1, 10),
        # Controls the sampling of weights, similar to subsample
        'bagging_temperature': trial.suggest_uniform('bagging_temperature', 0.0, 1.0),
        # Fix the seed for reproducibility
        'random_seed': 42,
        # Set verbose to False to suppress output during tuning
        'verbose': 0
    }
    
    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return f1_score(y_val, y_pred, average='weighted')

study_cat = optuna.create_study(direction='maximize')
study_cat.optimize(objective_cat, n_trials=150)

print("Best CatBoost trial:")
trial_cat = study_cat.best_trial
print(f"  F1 Score: {trial_cat.value}")
print("  Hyperparameters:")
for key, value in trial_cat.params.items():
    print(f"    {key}: {value}")

# Train final CatBoost model with best parameters
best_params_cat = trial_cat.params
final_cat_model = CatBoostClassifier(**best_params_cat, random_seed=42, verbose=0)
final_cat_model.fit(X, y)
# Predict on test set (assuming test set is preprocessed similar to training)
test_predictions_cat = final_cat_model.predict(test)
# Prepare submission
submission_cat = sample_submission.copy()
submission_cat['Admitted'] = test_predictions_cat
submission_cat.to_csv('base_submission_catboost.csv', index=False)
print("CatBoost submission saved as submission_catboost.csv")