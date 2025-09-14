
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

  # Y = WX + B

LEARNING_RATE = 0.01
N_ITERATIONS = 1000 #also known as epochs, where one epoch is one full pass over the training data
weight = np.random.randn(1,1)
bias = np.random.randn(1,1)
RANDOM_BIAS = 4

# Training Data
df = pd.read_csv('Salary_dataset.csv')

#  NumPy's calculations often work best with 2D column vectors, not 1D arrays. Right after extracting the values, you should reshape X and y
X = df['YearsExperience'].values
Y = df ['Salary'].values

X = X.reshape(-1, 1)
Y = Y.reshape(-1, 1)

n_samples = len(X)

for i in range(1, N_ITERATIONS+1):
    # Make a prediction with the weight and bias on the training data
    y_predicted = np.dot(X,weight) + bias

    # calculate loss
    cost = (1/n_samples) * np.sum((y_predicted-Y)**2)

    if i % 100 == 0:
        print(f"Epoch {i}, Cost: {cost}")

    #compute gradients
    # dw (derivative with respect to weight)
    # db (derivative with respect to bias)

    dw = (2/n_samples) * np.dot(X.T, (y_predicted - Y))
    db = (2/n_samples) * np.sum(y_predicted - Y)

    weight = weight - LEARNING_RATE * dw
    bias = bias - LEARNING_RATE * db

print("----- Training Complete -----")
print(f"Final Weight: {weight.item()}")
print(f"Final Bias: {bias.item()}")

model_predictions = np.dot(X,weight) + bias

plt.figure(figsize=(10, 6))
plt.scatter(X, Y, color='blue', label='Actual Data')
# Regression line (Models Results)
plt.plot(X, model_predictions, color='red', linewidth=3, label='Regression Line')
plt.title('Salary vs. Years of Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.grid(True)
plt.show()