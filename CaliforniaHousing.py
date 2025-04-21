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
X = df.drop(columns=[data.target.name])  # all columns except the target
y = df[data.target.name]  # median house value
# split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Create tensors for PyTorch
# Convert pandas arrays to torch tensors for model consumption
X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
y_train_t = torch.tensor(y_train.values, dtype=torch.float32)
X_test_t  = torch.tensor(X_test.values,  dtype=torch.float32)
y_test_t  = torch.tensor(y_test.values,  dtype=torch.float32)

# 3. Wrap data in DataLoader
# Using batch size 32, shuffle training data each epoch.
train_ds = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

# 4. Define a simple ANN model
# One hidden layer with 32 units + ReLU activation
dim = X_train_t.shape[1]
model = nn.Sequential(
    nn.Linear(dim, 32),  # input to hidden
    nn.ReLU(),            # non-linearity
    nn.Linear(32, 1)      # hidden to output
)

# Setup optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()  # mean squared error

# 5. Train the model
# I usually run about 30 epochs; print a quick status every 10.
model.train()
epochs = 30
for epoch in range(epochs := epochs):
    losses = []
    for xb, yb in train_loader:
        optimizer.zero_grad()           # clear gradients
        pred = model(xb).view(-1)      # forward pass
        loss = loss_fn(pred, yb)       # compute loss
        loss.backward()                 # backprop
        optimizer.step()                # update weights
        losses.append(loss.item())
    # every 10 epochs, print the average MSE
    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{epochs} MSE: {np.mean(losses):.3f}")

# 6. Evaluate on the test set
# Switch to evaluation mode and compute predictions
model.eval()
with torch.no_grad():
    y_pred = model(X_test_t).view(-1).numpy()
    y_true = y_test_t.numpy()

# Calculate overall metrics
mse = mean_squared_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
# Print metrics
print(f"Test MSE: {mse:.3f}")
print(f"Test R²: {r2:.3f}")
# Convert R² into a percentage "accuracy" metric
accuracy_pct = r2 * 100
print(f"Model accuracy (R² * 100): {accuracy_pct:.1f}%")

# 7. Feature importance (proxy via first-layer weights)
# Sum absolute weights to see which inputs have the biggest effect
first_w = model[0].weight.detach().abs().sum(dim=0).numpy()
feat_imp = pd.Series(first_w, index=X.columns).sort_values(ascending=False)

# Plot the top 8 features I care most about
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
