import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LassoLarsCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from tpot.builtins import StackingEstimator
import torch

# Load your clean features and labels
features = pd.read_csv('data/testing_features_cleaned.csv')
target = pd.read_csv('data/testing_labels_cleaned.csv')['ENERGY STAR Score']

# Replace inf/-inf with NaN
features.replace([np.inf, -np.inf], np.nan, inplace=True)

# Split into training and testing sets
training_features, testing_features, training_target, testing_target = \
    train_test_split(features, target, random_state=42)

# Handle missing values
imputer = SimpleImputer(strategy="median")
imputer.fit(training_features)
training_features = imputer.transform(training_features)
testing_features = imputer.transform(testing_features)

# Build the pipeline
# Build the pipeline
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=LassoLarsCV()),
    GradientBoostingRegressor(
        alpha=0.95,
        learning_rate=0.1,
        loss="absolute_error",
        max_depth=7,
        max_features=0.75,
        min_samples_leaf=3,
        min_samples_split=18,
        n_estimators=100,
        subsample=0.6
    )
)


# Fit the model
exported_pipeline.fit(training_features, training_target)

# Predict on test data
results = exported_pipeline.predict(testing_features)

# Output
print("Predictions on test data:")
print(results)
