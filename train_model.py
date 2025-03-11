import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math

class SolarDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SolarNet(nn.Module):
    def __init__(self, input_size):
        super(SolarNet, self).__init__()
        # Feature attention mechanism
        self.attention = nn.Linear(input_size, input_size)
        
        # Main network
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.LeakyReLU(0.1)
        self.norm1 = nn.InstanceNorm1d(128, affine=False)
        self.norm2 = nn.InstanceNorm1d(64, affine=False)
        self.norm3 = nn.InstanceNorm1d(32, affine=False)
        
        # Special processing for most important features
        self.daylight_proc = nn.Sequential(
            nn.Linear(1, 16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, 16)
        )
        self.radiation_proc = nn.Sequential(
            nn.Linear(1, 16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, 16)
        )
        
        self.skip1 = nn.Linear(input_size + 32, 64)  # Increased for feature processing
        self.skip2 = nn.Linear(64, 32)
    
    def forward(self, x, eval_mode=False):
        # Apply attention to input features
        attention = torch.sigmoid(self.attention(x))
        x = x * attention
        
        # Special processing for important features
        daylight = self.daylight_proc(x[:, 7:8])  # daylight_hours
        radiation = self.radiation_proc(x[:, 6:7])  # solar_radiation
        
        # Main network path
        identity1 = self.skip1(torch.cat([x, daylight, radiation], dim=1))
        x = self.fc1(x)
        x = self.norm1(x.unsqueeze(1)).squeeze(1)
        x = self.relu(x)
        if not eval_mode:
            x = self.dropout(x)
        x = self.fc2(x)
        x = x + identity1
        
        # Rest of the network
        identity2 = self.skip2(x)
        x = self.norm2(x.unsqueeze(1)).squeeze(1)
        x = self.relu(x)
        if not eval_mode:
            x = self.dropout(x)
        x = self.fc3(x)
        x = x + identity2
        
        x = self.norm3(x.unsqueeze(1)).squeeze(1)
        x = self.relu(x)
        if not eval_mode:
            x = self.dropout(x)
        x = self.fc4(x)
        return x

def load_data(file_path='processed_data/combined_solar_weather_data.csv'):
    df = pd.read_csv(file_path)
    
    # Prepare features (X) and target (y)
    features = ['max_temp', 'min_temp', 'precipitation', 'rain', 
               'snowfall', 'cloud_cover', 'solar_radiation',
               'daylight_hours']
    X = df[features].values
    y = df['total'].values  # Predict total energy production
    
    # Remove any rows with NaN values
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X = X[mask]
    y = y[mask]
    
    return X, y, features

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, target_val_loss=0.01, max_restarts=500):
    best_overall_loss = float('inf')
    best_overall_model = None
    restarts = 0
    
    while restarts < max_restarts:
        print(f"\nTraining attempt {restarts + 1}/{max_restarts}")
        
        # Reset model if not first attempt
        if restarts > 0:
            print("Reinitializing model with new random weights...")
            model = SolarNet(model.fc1.in_features).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, verbose=True, min_lr=1e-6)
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_model = None
        patience = 1000
        patience_counter = 0
        no_improvement_epochs = 0
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y.unsqueeze(1))
                
                # Check for NaN loss
                if math.isnan(loss.item()):
                    print("NaN loss detected, skipping batch")
                    continue
                    
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            # Validation phase
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X, eval_mode=True)
                    val_loss += criterion(outputs, batch_y.unsqueeze(1)).item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            # Skip if loss is NaN
            if math.isnan(train_loss) or math.isnan(val_loss):
                print(f"NaN loss detected in epoch {epoch+1}, skipping...")
                continue
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Print progress
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'Training Loss: {train_loss:.4f}')
            print(f'Validation Loss: {val_loss:.4f}')
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Modified early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model.state_dict().copy()
                patience_counter = 0
                no_improvement_epochs = 0
            else:
                patience_counter += 1
                no_improvement_epochs += 1
            
            # Check if we've met our target
            if val_loss <= target_val_loss:
                print(f"\nTarget validation loss {target_val_loss} achieved!")
                return best_model, train_losses, val_losses
            
            # Check if we're stuck
            if no_improvement_epochs >= 100:  # No improvement in 100 epochs
                print("\nTraining stuck, trying new initialization...")
                break
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Update best overall model if this attempt was better
        if best_val_loss < best_overall_loss:
            best_overall_loss = best_val_loss
            best_overall_model = best_model
        
        restarts += 1
        print(f"Best validation loss this attempt: {best_val_loss:.6f}")
        print(f"Best overall validation loss so far: {best_overall_loss:.6f}")
    
    print("\nMax restarts reached. Using best model found.")
    return best_overall_model, train_losses, val_losses

def calculate_feature_importance(model, X_train, feature_names, device):
    model.eval()
    importance_scores = []
    
    # Create a baseline prediction
    X_baseline = torch.FloatTensor(X_train).to(device)
    with torch.no_grad():
        baseline_pred = model(X_baseline, eval_mode=True)
    
    # Calculate importance for each feature
    for i in range(X_train.shape[1]):
        X_modified = X_baseline.clone()
        X_modified[:, i] = 0  # Zero out the feature
        with torch.no_grad():
            modified_pred = model(X_modified, eval_mode=True)
            
        # Calculate importance as the mean absolute difference in predictions
        importance = torch.abs(modified_pred - baseline_pred).mean().item()
        importance_scores.append(importance)
    
    # Create feature importance DataFrame
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_scores
    }).sort_values('Importance', ascending=False)
    
    return feature_importance

def train_and_evaluate():
    # Set device (GPU if available, else CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and preprocess the data
    X, y, feature_names = load_data()
    
    # Scale the target variable as well
    y_scaler = MinMaxScaler()
    y = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Create data loaders
    train_dataset = SolarDataset(X_train_scaled, y_train)
    val_dataset = SolarDataset(X_val_scaled, y_val)
    test_dataset = SolarDataset(X_test_scaled, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Initialize model with smaller learning rate
    model = SolarNet(X_train.shape[1]).to(device)
    
    # Define loss function and optimizer with weight decay
    criterion = nn.MSELoss()
    
    # Adjusted optimization parameters
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.85, 
                                patience=35, verbose=True, min_lr=1e-6)
    
    # More demanding target
    best_model_state, train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        num_epochs=50000, device=device, target_val_loss=0.006, max_restarts=5
    )
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Evaluate on test set
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X, eval_mode=True)
            predictions.extend(outputs.cpu().numpy().flatten())
            actuals.extend(batch_y.cpu().numpy())
    
    # Inverse transform predictions and actuals
    predictions = y_scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    actuals = y_scaler.inverse_transform(np.array(actuals).reshape(-1, 1)).flatten()
    
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    print(f'\nTest Mean Absolute Error: {mae:.2f} kWh')
    print(f'Test Root Mean Square Error: {rmse:.2f} kWh')
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Model Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_history.png')
    plt.close()
    
    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(actuals, predictions, alpha=0.5)
    plt.plot([0, max(actuals)], [0, max(actuals)], 'r--')  # Perfect prediction line
    plt.xlabel('Actual Energy Production (kWh)')
    plt.ylabel('Predicted Energy Production (kWh)')
    plt.title('Actual vs Predicted Solar Energy Production')
    plt.savefig('prediction_scatter.png')
    plt.close()
    
    # Save the model and scalers
    torch.save({
        'model_state_dict': best_model_state,
        'input_size': X_train.shape[1],
        'feature_names': feature_names
    }, 'solar_prediction_model.pth')
    import joblib
    joblib.dump(scaler, 'feature_scaler.joblib')
    joblib.dump(y_scaler, 'target_scaler.joblib')
    
    # Calculate feature importance
    feature_importance = calculate_feature_importance(model, X_train_scaled, feature_names, device)
    print("\nFeature Importance:")
    print(feature_importance)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.bar(feature_importance['Feature'], feature_importance['Importance'])
    plt.xticks(rotation=45)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    return model, scaler, y_scaler

if __name__ == "__main__":
    train_and_evaluate()