import numpy as np  # numerical computations
import pandas as pd  # data handling
import torch  # deep learning framework
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_california_housing  # built-in dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score  # evaluation metrics
import matplotlib.pyplot as plt  # plotting


# 1. Load dataset
# Fetch the data, wrap it in a DataFrame, and separate features/target.
data = fetch_california_housing(as_frame=True)
df = data.frame  # pandas DataFrame
X = df.drop(columns=[data.target.name])  # features
y = df[data.target.name]  # median house value
# split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Normalize features
# Scale each feature to the [0, 1] range based on training set statistics
train_min = X_train.min()
train_max = X_train.max()
X_train = (X_train - train_min) / (train_max - train_min)
X_test = (X_test - train_min) / (train_max - train_min)

# 3. Create tensors for PyTorch
# Convert pandas arrays to torch tensors for model consumption
X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
y_train_t = torch.tensor(y_train.values, dtype=torch.float32)
X_test_t  = torch.tensor(X_test.values,  dtype=torch.float32)
y_test_t  = torch.tensor(y_test.values,  dtype=torch.float32)

# 3. Wrap data in DataLoader
# Using batch size 32, shuffle training data each epoch.
train_ds = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

# 4. Define an ANN model with multiple hidden layers and dropout
# Extra layers: 64 -> 32 -> 16 units plus dropout for regularization
input_dim = X_train_t.shape[1]
model = nn.Sequential(
    nn.Linear(input_dim, 64),  # input to first hidden
    nn.ReLU(),
    nn.Dropout(0.1),            # dropout to reduce overfitting
    nn.Linear(64, 32),          # second hidden layer
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(32, 16),          # third hidden layer
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(16, 1)            # hidden to output
)

# Setup optimizer and loss function
# Using Adam with weight decay (L2) and MSE loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
loss_fn = nn.MSELoss()

# 5. Train the model
# Train for 60 epochs; print the average MSE every 10 epochs
epochs = 60
model.train()
for epoch in range(epochs):
    losses = []
    for xb, yb in train_loader:
        optimizer.zero_grad()          # clear gradients
        pred = model(xb).view(-1)     # forward pass
        loss = loss_fn(pred, yb)      # compute loss
        loss.backward()                # backpropagation
        optimizer.step()               # update weights
        losses.append(loss.item())
    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{epochs} MSE: {np.mean(losses):.3f}")

# 6. Evaluate on the test set
# Switch to eval mode and compute predictions
model.eval()
with torch.no_grad():
    y_pred = model(X_test_t).view(-1).numpy()
    y_true = y_test_t.numpy()

# Calculate overall metrics
mse = mean_squared_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
print(f"Test MSE: {mse:.3f}")
print(f"Test R²: {r2:.3f}")
accuracy_pct = r2 * 100
print(f"Model accuracy (R² * 100): {accuracy_pct:.1f}%")

# 7. Feature importance (proxy via first-layer weights)
# Sum absolute weights to see which inputs have the biggest effect
first_w = model[0].weight.detach().abs().sum(dim=0).numpy()
feat_imp = pd.Series(first_w, index=X.columns).sort_values(ascending=False)

# Plot the top 8 features
# Visualize which features the model relies on most
top = feat_imp.head(8)
plt.figure(figsize=(6,4))
top[::-1].plot(kind='barh', edgecolor='k')
plt.title('Top 8 Feature Importances')
plt.xlabel('Sum of absolute weights (proxy)')
plt.tight_layout()
plt.show()

# 8. Predicted vs Actual scatter
# A red dashed diagonal shows perfect predictions.
plt.figure(figsize=(5,5))
plt.scatter(y_true, y_pred, alpha=0.6)
lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
plt.plot(lims, lims, 'r--')
plt.xlabel('Actual Median Value')
plt.ylabel('Predicted Median Value')
plt.title('Predicted vs Actual Comparison')
plt.tight_layout()
plt.show()
