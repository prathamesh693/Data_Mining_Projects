# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 06:32:33 2025

@author: ratho
"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Select features and target
X = df[['avg_temp', 'pesticides_tonnes']]
y = df['hg/ha_yield']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))


from sklearn.cluster import KMeans

# Cluster countries by average yield and temperature
cluster_data = df.groupby('Entity')[['hg/ha_yield', 'avg_temp']].mean()
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_data['cluster'] = kmeans.fit_predict(cluster_data)

print(cluster_data.head())
