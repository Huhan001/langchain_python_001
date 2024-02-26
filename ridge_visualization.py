import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('data.csv')
X = df[['depth']]
y = df['price']

# Create a Ridge regression model
ridge = Ridge(alpha=1.0)
ridge.fit(X, y)

# Visualizing the data
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue')
plt.plot(X, ridge.predict(X), color='red', linewidth=3)
plt.xlabel('Depth')
plt.ylabel('Price')
plt.title('Ridge Regression: Depth vs Price')
plt.grid(True)
plt.show()
