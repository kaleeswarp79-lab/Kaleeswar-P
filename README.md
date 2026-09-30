import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Historical stock data
days = np.array([
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 14, 15
]).reshape(-1, 1)

prices = np.array([
    100, 102, 101, 105, 107,
    106, 110, 112, 111, 115,
    117, 116, 120, 122, 124
])

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(days, prices)

# Predict prices for existing days
predicted_prices = model.predict(days)

# Predict price for the next day
next_day = np.array([[16]])
next_price = model.predict(next_day)

# Display output
print("====================================")
print("       STOCK PRICE PREDICTOR")
print("====================================")
print("Model Used : Linear Regression")
print("Number of Historical Days :", len(days))

print("\nHistorical Prices:")
print(prices)

print("\nPredicted Price for Day 16:")
print("₹", round(float(next_price[0]), 2))

# Display graph
plt.figure(figsize=(8, 5))
plt.scatter(days, prices, label="Actual Price")
plt.plot(days, predicted_prices, label="Regression Line")

plt.xlabel("Day")
plt.ylabel("Stock Price (₹)")
plt.title("Stock Price Prediction using Linear Regression")
plt.legend()
plt.grid(True)
plt.show()
