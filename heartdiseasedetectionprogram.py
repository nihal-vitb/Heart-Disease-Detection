print("\n= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =")
print(" <- - - - - - -HEART DISEASE DETECTION PROGRAM- - - - - - ->")
print("= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =\n")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

print("\n   < LOADING DATA >",end="                             ")
try:
    data = pd.read_csv("heart_cleveland_dataset.csv")
    print("[✓]\n")
except FileNotFoundError:
    print("\n ERROR: 'heart_cleveland_dataset.csv' not found;\n\n Download the Heart Cleveland Dataset:\n https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data\n")
    exit()

print("\n   < MISSING DATA HANDLING >",end="                    ")
illogical_features = ["trestbps","chol","thalach","oldpeak"]
data[illogical_features] = data[illogical_features].replace(0, np.nan)
print("[✓]\n")

print("\n   < FEATURE & TARGET SEPARATION >",end="              ")
x = data.drop("condition", axis = 1)
y = data["condition"]
print("[✓]\n")

print("\n   < TRAIN & TEST DATA DISTRIBUTION >",end="           ")
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size = 0.3,
    random_state = 42,
    stratify = y
)
print("[✓]\n")

print("\n   < INITIATING PIPELINE & PARAMETER GRIDS >",end="    ")

pipeline = Pipeline([
    ('imputer', SimpleImputer(missing_values= np.nan, strategy = 'mean')),
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier( random_state = 42 ))
])

param_grid = {
    'model__n_estimators': [150, 250, 500],
    'model__max_depth': [7, 10, 12],
    'model__min_samples_leaf' : [1, 2, 3] 
}
print("[✓]\n")


print("\n   < INITIALIZING MODEL TRAINING & TUNING >",end="     ")


grid_search = GridSearchCV(
    estimator = pipeline,
    param_grid = param_grid,
    cv = 5,
    scoring = 'f1',
    n_jobs = -1
)
print("[✓]\n")

grid_search.fit(x_train, y_train)

print("\n   < BEST MODEL FOUND >",end="                         ")
best_model = grid_search.best_estimator_
print("[✓]\n")
print("= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =")

y_pred = best_model.predict(x_test)
y_proba = best_model.predict_proba(x_test)[:, 1]
print("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
print("     < MODEL EVALUATION (RANDOM FOREST CLASSIFICATION) >")
print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n")
print(f" • Best F1 Score Found (Tuning): {grid_search.best_score_:.4f}\n")
print(f" • Best Parameters: \n")
for i in grid_search.best_params_:
    print(f"     > {i} : {grid_search.best_params_[i]}")
print(f"\n • Test Set Accuracy: {accuracy_score(y_test, y_pred)*100:.2f} %\n")
print(f" • Test Set F1 Score: {f1_score(y_test, y_pred):.4f}\n")
print(f" • Test Set AUC-ROC Score: {roc_auc_score(y_test, y_proba):.4f}")
print("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
print("            < CONFUSION MATRIX ON TEST SET >")
print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n")
print(confusion_matrix(y_test, y_pred)) 

TN = confusion_matrix(y_test, y_pred)[0][0]
FP = confusion_matrix(y_test, y_pred)[0][1]
FN = confusion_matrix(y_test, y_pred)[1][0]
TP = confusion_matrix(y_test, y_pred)[1][1]

total_samples = TN + FP + FN + TP
total_actual_pos = TP + FN
total_actual_neg = TN + FP 
recall = TP / total_actual_pos 
specificity = TN / total_actual_neg
precision = TP / (TP + FP)

print("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
print("             < CONFUSION MATRIX ANALYSIS >")
print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n")
print(f" • Total Samples Tested: {total_samples}\n")
print(f" • Total Actual Positive Cases (Have Heart Disease): {total_actual_pos}\n")
print(f" • Total Actual Negative Cases (No Heart Disease): {total_actual_neg}\n")
print("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
print("                   - ERROR ANALYSIS -")
print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n")
print(f" • Recall (Sensitivity): {recall:.4f}")
print(f" * Model caught {recall*100:.2f} % of all actual heart disease patients")
print(f" * Missed Cases: {FN} [ !! CRITICAL ERROR !! ]\n")
print(f" • Precision: {precision:.4f}")
print(f" * Model predicted positive correctly {precision*100:.2f} % of the time")
print(f" * Falsely Predicted having Heart Disease: {FP}\n")
print(f" • Specificity: {specificity:.4f}")
print(f" * Model identified {specificity*100:.2f} % of all non-heart disease patients\n")
print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n")