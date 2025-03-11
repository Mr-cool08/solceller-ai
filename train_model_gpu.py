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
import torch.backends.cudnn as cudnn

# Enable CUDA optimizations
torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on matmul
torch.backends.cudnn.allow_tf32 = True        # Allow TF32 on cudnn
cudnn.benchmark = True                        # Enable cudnn auto-tuner

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
        
        # Main network with larger sizes for GPU
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)
        
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.LeakyReLU(0.1)
        self.norm1 = nn.InstanceNorm1d(512, affine=False)
        self.norm2 = nn.InstanceNorm1d(256, affine=False)
        self.norm3 = nn.InstanceNorm1d(128, affine=False)
        
        # Special processing for most important features
        self.daylight_proc = nn.Sequential(
            nn.Linear(1, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 32)
        )
        self.radiation_proc = nn.Sequential(
            nn.Linear(1, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 32)
        )
        
        self.skip1 = nn.Linear(input_size + 64, 256)
        self.skip2 = nn.Linear(256, 128)

    def forward(self, x, eval_mode=False):
        # Apply attention to input features
        attention = torch.sigmoid(self.attention(x))
        x = x * attention
        
        # Special processing for important features
        daylight = self.daylight_proc(x[:, 7:8])
        radiation = self.radiation_proc(x[:, 6:7])
        
        # Main network path with skip connections
        identity1 = self.skip1(torch.cat([x, daylight, radiation], dim=1))
        x = self.fc1(x)
        x = self.norm1(x.unsqueeze(1)).squeeze(1)
        x = self.relu(x)
        if not eval_mode:
            x = self.dropout(x)
        x = self.fc2(x)
        x = x + identity1
        
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

def check_cuda_availability():
    """Check CUDA availability and print helpful messages"""
    print("\nChecking GPU availability...")
    
    if not torch.cuda.is_available():
        print("\nCUDA is not available. This could be due to:")
        print("1. NVIDIA GPU drivers not installed")
        print("2. CUDA toolkit not installed")
        print("3. PyTorch not installed with CUDA support")
        print("\nTo fix this:")
        print("1. Make sure you have an NVIDIA GPU")
        print("2. Install latest NVIDIA drivers from: https://www.nvidia.com/Download/index.aspx")
        print("3. Install CUDA toolkit from: https://developer.nvidia.com/cuda-downloads")
        print("4. Reinstall PyTorch with CUDA support:")
        print("   pip3 uninstall torch")
        print("   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("\nCurrent PyTorch setup:")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
        raise RuntimeError("CUDA is not available. See above instructions for setup.")
    
    # Print GPU info if available
    device_count = torch.cuda.device_count()
    print(f"\nFound {device_count} CUDA device(s):")
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        print(f"  {i}: {props.name}")
        print(f"     Memory: {props.total_memory / 1024**3:.1f}GB")
        print(f"     CUDA Capability: {props.major}.{props.minor}")
    
    return torch.device('cuda:0')

def train_and_evaluate():
    # Replace device setup with new check
    device = check_cuda_availability()
    torch.cuda.set_device(device)
    
    # Print GPU info
    print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**2:.0f}MB")
    
    # Load and preprocess data
    X, y, feature_names = load_data()
    
    # Use 32-bit precision for better GPU performance
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    
    # Scale and split data
    y_scaler = MinMaxScaler()
    y = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Create data loaders with larger batch sizes for GPU
    train_dataset = SolarDataset(X_train_scaled, y_train)
    val_dataset = SolarDataset(X_val_scaled, y_val)
    test_dataset = SolarDataset(X_test_scaled, y_test)
    
    # Increased batch sizes for GPU
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=128, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=128, pin_memory=True)
    
    # Initialize model
    model = SolarNet(X_train.shape[1]).to(device)
    
    # Use mixed precision training
    scaler = torch.cuda.amp.GradScaler()
    
    # Optimizer with higher learning rate for GPU
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, verbose=True, min_lr=1e-6)
    
    # Train with auto-restart and target loss
    best_model_state, train_losses, val_losses = train_model(
        model, train_loader, val_loader, nn.MSELoss(), optimizer, scheduler,
        num_epochs=50000, device=device, target_val_loss=0.005, max_restarts=500,
        scaler=scaler  # Pass gradient scaler
    )
    
    # Rest of the evaluation code...
    # ...existing evaluation code from train_model.py...

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                num_epochs, device, target_val_loss=0.005, max_restarts=500, scaler=None):
    # Modified training loop with mixed precision
    best_overall_loss = float('inf')
    best_overall_model = None
    restarts = 0
    
    while restarts < max_restarts:
        print(f"\nTraining attempt {restarts + 1}/{max_restarts}")
        
        if restarts > 0:
            model = SolarNet(model.fc1.in_features).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.001)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, verbose=True, min_lr=1e-6)
        
        # ...rest of training loop with mixed precision updates...
        # (Keep existing training loop structure but add mixed precision training)
    
    return best_overall_model, train_losses, val_losses

if __name__ == "__main__":
    train_and_evaluate()
