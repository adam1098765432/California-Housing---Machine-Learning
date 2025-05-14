# California Housing Regression ANN

A simple PyTorch feed-forward neural network that predicts median house values in California from census block-group data. It reports Test MSE and R² “accuracy,” visualizes top feature importances, and plots predicted vs. actual values.

## Dataset

We use the **California Housing Prices** dataset, originally from the 1990 U.S. Census. It contains 20 350 block-group observations with the following features:

- **MedInc**: median income in block group  
- **HouseAge**: median age of houses  
- **AveRooms**, **AveBedrms**, **Population**, **AveOccup**: average rooms, bedrooms, population, occupants  
- **Latitude**, **Longitude**: geographic coordinates  
- **Target**: median house value (in \$100,000 units)

This dataset is exposed in scikit-learn via `fetch_california_housing()`.

## Implementation

- **Model**: 3 hidden layers (64 → 32 → 16 units) with ReLU activations, dropout, and BatchNorm1d  
- **Training**: 60 epochs, Adam optimizer with L2 weight decay, MSE loss  
- **Normalization**: Min–max scales each feature to [0, 1] based on train-set statistics  

## Results


 - Epoch 0/60 MSE: 1.701
 - Epoch 10/60 MSE: 0.437  
 - Epoch 20/60 MSE: 0.407  
 - Epoch 30/60 MSE: 0.407  
 - Epoch 40/60 MSE: 0.403  
 - Epoch 50/60 MSE: 0.392  
 - Test MSE: 0.332  
 - Test R²: 0.747  
 - Model accuracy (R² * 100): 74.7%  

![Top 8 Feature Importance](https://github.com/user-attachments/assets/8e7f8c4f-61a3-4852-b383-2f51f3f64b67)

![predicted vs actual comparison](https://github.com/user-attachments/assets/129c333c-53aa-4697-8092-fe1de172565f)
