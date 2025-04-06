import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import precision_score, make_scorer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

# Define a precision scorer (weighted precision)
precision_scorer = make_scorer(precision_score, average='weighted')

# Assume X, y, test, and sample_submission are already defined and preprocessed
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

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# For demonstration, here are example tuned hyperparameters.
# Replace these with your best hyperparameters from Optuna tuning.
rf_model = RandomForestClassifier(
    n_estimators=150,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42
)

cat_model = CatBoostClassifier(
    iterations=200,
    depth=6,
    learning_rate=0.05,
    l2_leaf_reg=3,
    random_seed=42,
    verbose=0
)

xgb_model = XGBClassifier(
    n_estimators=150,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

lgbm_model = LGBMClassifier(
    n_estimators=150,
    max_depth=10,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=10,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# Create a StratifiedKFold for cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate each base model using cross-validation and print weighted precision scores
rf_cv_scores = cross_val_score(rf_model, X, y, cv=skf, scoring=precision_scorer)
cat_cv_scores = cross_val_score(cat_model, X, y, cv=skf, scoring=precision_scorer)
xgb_cv_scores = cross_val_score(xgb_model, X, y, cv=skf, scoring=precision_scorer)
lgbm_cv_scores = cross_val_score(lgbm_model, X, y, cv=skf, scoring=precision_scorer)

print("RandomForest CV Weighted Precision: {:.4f} ± {:.4f}".format(rf_cv_scores.mean(), rf_cv_scores.std()))
print("CatBoost CV Weighted Precision: {:.4f} ± {:.4f}".format(cat_cv_scores.mean(), cat_cv_scores.std()))
print("XGBoost CV Weighted Precision: {:.4f} ± {:.4f}".format(xgb_cv_scores.mean(), xgb_cv_scores.std()))
print("LightGBM CV Weighted Precision: {:.4f} ± {:.4f}".format(lgbm_cv_scores.mean(), lgbm_cv_scores.std()))

# Build a stacking ensemble with the four models as base estimators.
estimators = [
    ('rf', rf_model),
    ('cat', cat_model),
    ('xgb', xgb_model),
    ('lgbm', lgbm_model)
]

stacking_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv=skf,          # use the same cross-validation strategy for stacking
    passthrough=False,  # set to True to include original features for meta-model training
    n_jobs=-1
)

# Evaluate the stacking ensemble using cross-validation
stacking_cv_scores = cross_val_score(stacking_model, X, y, cv=skf, scoring=precision_scorer)
print("Stacking Ensemble CV Weighted Precision: {:.4f} ± {:.4f}".format(
    stacking_cv_scores.mean(), stacking_cv_scores.std()))

# Train the final stacking model on the full training data
stacking_model.fit(X, y)

# Predict on the test set
test_predictions = stacking_model.predict(test)

# Prepare submission
submission = sample_submission.copy()
submission['Admitted'] = test_predictions
submission.to_csv('submission_stacking.csv', index=False)
print("Stacking ensemble submission saved as submission_stacking.csv")
