import torch.nn as nn
import torch.nn.functional as F
import torch




class CustomCNN(nn.Module):
    def __init__(self,window_size):
        super(CustomCNN, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=14, kernel_size=3,stride=1)
        self.max_pool = nn.AvgPool1d(kernel_size=2, stride=1)
        
        # Calculate the size after conv1d and maxpool layers
        conv_output_size = window_size - 2  # window_size after Conv1d with kernel_size=3
        pool_output_size = conv_output_size -1  # after MaxPool1d with kernel_size=2 and stride=1
        self.fc1 = nn.Linear(14 * pool_output_size, 10)
        self.fc3 = nn.Linear(10,10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu(self.conv1d(x))
        x = self.max_pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        
        x = self.fc2(x)  
        return x


def train_model(model, X, y, optimizer, criterion, epochs):
    """
    Trains a given model on the provided dataset.

    Parameters:
    - model: The PyTorch model to be trained.
    - X: Input data (features).
    - y: Ground truth labels.
    - optimizer: Optimizer instance to update model parameters.
    - criterion: Loss function to compute the error.
    - epochs: Number of training iterations (epochs).

    Returns:
    - y_hat: tensor
    """
    for epoch in range(1, epochs + 1):
        model.train()  


        y_hat = model(X)
        loss = criterion(y_hat, y)

        # Backward pass: compute gradients and update parameters
        optimizer.zero_grad()  
        loss.backward()      
        optimizer.step()       

        print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}")

    return y_hat




def predict(model_trained, X_test):
    """
    Makes predictions using a trained model on the given test data.

    Parameters:
    - model_trained: The trained PyTorch model.
    - X_test: The test data (input features) for which predictions are required.

    Returns:
    - y_pred: The model's predictions for the test data.
    """
    model_trained.eval()  # Set the model to evaluation mode (disables dropout, batch norm updates)
    
    with torch.no_grad():
        # Perform forward pass without tracking gradients
        y_pred = model_trained(X_test)

    return y_pred
