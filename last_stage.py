# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 06:33:47 2025

@author: ratho
"""

## Regression Evaluation 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
print("R² Score:", r2_score(y_test, y_pred))

## Classification Evaluation
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

##  Clustering Evaluation
from sklearn.metrics import silhouette_score

score = silhouette_score(df[['avg_temp', 'pesticides_tonnes']], df['cluster'])
print("Silhouette Score:", score)

## Interpreting Linear Regression
# Check learned coefficients
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.4f}")
