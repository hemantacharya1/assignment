import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('fruit.csv')
X = data[['length_cm', 'weight_g', 'yellow_score']].values
y = data['label'].values.reshape(-1, 1)

# Normalize features (optional but helps)
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Initialize weights and bias
np.random.seed(42)
weights = np.random.randn(X.shape[1], 1)
bias = 0

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Loss function: Binary Cross-Entropy
def loss(y_true, y_pred):
    m = y_true.shape[0]
    return -(1/m) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Prediction function
def predict(X, weights, bias):
    z = np.dot(X, weights) + bias
    return sigmoid(z)

# Training parameters
learning_rate = 0.1
epochs = 500
loss_history = []
accuracy_history = []

# Training loop
for epoch in range(epochs):
    # Forward pass
    y_pred = predict(X, weights, bias)
    
    # Calculate loss
    current_loss = loss(y, y_pred)
    loss_history.append(current_loss)
    
    # Calculate accuracy
    predictions = (y_pred >= 0.5).astype(int)
    accuracy = np.mean(predictions == y)
    accuracy_history.append(accuracy)
    
    # Gradient calculation
    m = y.shape[0]
    dz = y_pred - y
    dw = (1/m) * np.dot(X.T, dz)
    db = (1/m) * np.sum(dz)
    
    # Update weights and bias
    weights -= learning_rate * dw
    bias -= learning_rate * db
    
    # Early stopping if loss < 0.05
    if current_loss < 0.05:
        print(f"Stopping early at epoch {epoch}")
        break
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch}: Loss = {current_loss:.4f}, Accuracy = {accuracy:.4f}")

# Plot loss and accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(loss_history)
plt.title('Loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(accuracy_history)
plt.title('Accuracy over epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.show()
