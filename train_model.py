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
import os
import json
from datetime import datetime
import joblib

MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

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
        # Basic architecture
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
    
    def forward(self, x, eval_mode=False):
        x = self.relu(self.fc1(x))
        x = self.dropout(x) if not eval_mode else x
        x = self.relu(self.fc2(x))
        x = self.dropout(x) if not eval_mode else x
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def load_data(file_path='processed_data/combined_solar_weather_data.csv'):
    df = pd.read_csv(file_path)
    
    # Prepare features (X) and target (y)
    features = ['max_temp', 'min_temp', 'precipitation', 'rain', 
               'snowfall', 'cloud_cover', 'solar_radiation']
    X = df[features].values
    y = df['total'].values  # Predict total energy production
    
    # Remove any rows with NaN values
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X = X[mask]
    y = y[mask]
    
    return X, y, features

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model = None
    patience = 1000
    patience_counter = 0
    
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
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    if best_model is None:
        print("Warning: No valid model state was saved. Using current model state.")
        best_model = model.state_dict().copy()
    
    return best_model, train_losses, val_losses

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
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    # Initialize model with smaller learning rate
    model = SolarNet(X_train.shape[1]).to(device)
    
    # Define loss function and optimizer with weight decay
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-6)
    
    # Train the model
    best_model_state, train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        num_epochs=200000, device=device
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
    
    # Get next version number
    version_file = os.path.join(MODEL_DIR, 'model_versions.json')
    try:
        with open(version_file, 'r') as f:
            versions = json.load(f)
        next_version = max(v['version'] for v in versions) + 1
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        versions = []
        next_version = 1

    # Generate filenames
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    model_filename = f'model_v{next_version}.pth'
    feature_scaler_filename = f'feature_scaler_v{next_version}.joblib'
    target_scaler_filename = f'target_scaler_v{next_version}.joblib'

    # Save model and scalers in models directory
    torch.save({
        'model_state_dict': best_model_state,
        'input_size': X_train.shape[1],
        'feature_names': feature_names
    }, os.path.join(MODEL_DIR, model_filename))
    
    joblib.dump(scaler, os.path.join(MODEL_DIR, feature_scaler_filename))
    joblib.dump(y_scaler, os.path.join(MODEL_DIR, target_scaler_filename))

    # Update version tracking with native Python types
    version_info = {
        'version': next_version,
        'timestamp': timestamp,
        'mae': float(mae),  # Convert numpy.float32 to Python float
        'rmse': float(rmse),
        'val_loss': float(min(val_losses)),
        'train_loss': float(min(train_losses)),
        'model_file': model_filename,
        'feature_scaler': feature_scaler_filename,
        'target_scaler': target_scaler_filename
    }
    versions.append(version_info)

    # Save version info
    with open(version_file, 'w') as f:
        json.dump(versions, f, indent=4)

    print(f"\nModel saved as version {next_version}")
    print(f"MAE: {mae:.2f} kWh")
    print(f"RMSE: {rmse:.2f} kWh")

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