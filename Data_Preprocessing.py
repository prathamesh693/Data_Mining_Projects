# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 06:31:35 2025

@author: ratho
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA

# Load dataset
df = pd.read_csv("R:/Project/AI-Driven Crop Yield Prediction Based on Soil Health Analysis/Data_set/updated_data.csv")

# 1. Data Cleaning
print(df.columns)

# 1.1 Handle Missing Values
df['pesticides_tonnes'].fillna(df['pesticides_tonnes'].mean(), inplace=True)
df['avg_temp'].fillna(df['avg_temp'].mean(), inplace=True)

# 1.2 Noisy Data
# (a) Binning
df['temp_binned'] = pd.cut(df['avg_temp'], bins=3, labels=['Low', 'Medium', 'High'])

# (b) Regression
X = df[['pesticides_tonnes']]
y = df['avg_temp']
model = LinearRegression()
model.fit(X, y)
df['avg_temp_predicted'] = model.predict(X)

# (c) Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(df[['avg_temp', 'pesticides_tonnes']])

# 1.3 Removing Duplicates
df = df.drop_duplicates()

# 2. Data Integration
# 2.1 Record Linkage – Placeholder (requires string-based merging logic)
# 2.2 Data Fusion – Example feature
df['temp_yield_ratio'] = df['avg_temp'] / (df['hg/ha_yield'] + 1)

# 3. Data Transformation

# 3.1 Normalization
scaler = MinMaxScaler()
df[['norm_yield', 'norm_temp']] = scaler.fit_transform(df[['hg/ha_yield', 'avg_temp']])

# 3.2 Discretization
df['yield_bins'] = pd.qcut(df['hg/ha_yield'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])

# 3.3 Data Aggregation
agg_df = df.groupby('Year').agg({
    'hg/ha_yield': 'mean',
    'pesticides_tonnes': 'sum'
}).reset_index()

# 3.4 Concept Hierarchy Generation
df['decade'] = (df['Year'] // 10) * 10

# 4. Data Reduction

# 4.1 Feature Selection
selector = VarianceThreshold(threshold=0.01)
reduced_data = selector.fit_transform(df[['hg/ha_yield', 'pesticides_tonnes', 'avg_temp']])

# 4.2 PCA
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df[['hg/ha_yield', 'pesticides_tonnes', 'avg_temp']])