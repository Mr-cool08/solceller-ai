import json
import os
from tabulate import tabulate

MODEL_DIR = 'models'

def list_model_versions():
    """List all trained model versions and their performance metrics"""
    version_file = os.path.join(MODEL_DIR, 'model_versions.json')
    try:
        with open(version_file, 'r') as f:
            versions = json.load(f)
        
        if not versions:
            print("No model versions found. Train a model first.")
            return
        
        # Prepare table data
        headers = ['Version', 'Date', 'MAE (kWh)', 'RMSE (kWh)', 'Val Loss', 'Train Loss']
        table_data = []
        
        for v in versions:
            table_data.append([
                v['version'],
                v['timestamp'],
                f"{v['mae']:.2f}",
                f"{v['rmse']:.2f}",
                f"{v['val_loss']:.4f}",
                f"{v['train_loss']:.4f}"
            ])
        
        # Sort by RMSE (best performing first)
        table_data.sort(key=lambda x: float(x[3]))
        
        print("\nModel Versions (sorted by RMSE):")
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
        
        # Print best version details
        best_version = min(versions, key=lambda x: x['rmse'])
        print(f"\nBest Model (Version {best_version['version']}):")
        print(f"MAE: {best_version['mae']:.2f} kWh")
        print(f"RMSE: {best_version['rmse']:.2f} kWh")
        print(f"Validation Loss: {best_version['val_loss']:.4f}")
        print(f"Training Loss: {best_version['train_loss']:.4f}")
        print(f"Model File: {best_version['model_file']}")
        
    except FileNotFoundError:
        print("No model versions found. Train a model first.")
        print(f"Expected version file: {version_file}")

def get_best_model():
    """Return the best performing model version based on RMSE"""
    version_file = os.path.join(MODEL_DIR, 'model_versions.json')
    try:
        with open(version_file, 'r') as f:
            versions = json.load(f)
        return min(versions, key=lambda x: x['rmse']) if versions else None
    except FileNotFoundError:
        return None

if __name__ == "__main__":
    list_model_versions()
