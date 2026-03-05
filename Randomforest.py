import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Sample dataset (you can replace later)
X = [[10], [20], [30], [40], [50], [60]]
y = [15, 25, 35, 45, 55, 65]

# Create model
RandomForestRegModel = RandomForestRegressor()

# Train model
RandomForestRegModel.fit(X, y)

# Prediction
X_marks = [[70]]
prediction = RandomForestRegModel.predict(X_marks)

print("Predicted value:", prediction)