import json
import os
from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np

MODEL_DIR = 'models'
ANALYSIS_DIR = 'analysis'

os.makedirs(ANALYSIS_DIR, exist_ok=True)

def load_model_versions():
    version_file = os.path.join(MODEL_DIR, 'model_versions.json')
    try:
        with open(version_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("No model versions found")
        return []

def compare_models():
    versions = load_model_versions()
    if not versions:
        return
    
    # Prepare data for comparison
    data = []
    maes = []
    rmses = []
    versions_nums = []
    val_losses = []
    
    for v in versions:
        data.append([
            v['version'],
            v['timestamp'],
            f"{v['mae']:.2f}",
            f"{v['rmse']:.2f}",
            f"{v['val_loss']:.4f}",
            f"{v['train_loss']:.4f}"
        ])
        maes.append(v['mae'])
        rmses.append(v['rmse'])
        versions_nums.append(v['version'])
        val_losses.append(v['val_loss'])
    
    # Print table
    headers = ['Version', 'Date', 'MAE (kWh)', 'RMSE (kWh)', 'Val Loss', 'Train Loss']
    print("\nModel Version Comparison:")
    print(tabulate(data, headers=headers, tablefmt='grid'))
    
    # Plot metrics over versions
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(versions_nums, maes, 'b-o', label='MAE')
    plt.plot(versions_nums, rmses, 'r-o', label='RMSE')
    plt.xlabel('Model Version')
    plt.ylabel('Error (kWh)')
    plt.title('Error Metrics by Version')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(versions_nums, val_losses, 'g-o', label='Validation Loss')
    plt.xlabel('Model Version')
    plt.ylabel('Loss')
    plt.title('Validation Loss by Version')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, 'model_comparison.png'))
    plt.close()
    
    # Find best model for each metric
    best_mae = min(versions, key=lambda x: x['mae'])
    best_rmse = min(versions, key=lambda x: x['rmse'])
    best_val = min(versions, key=lambda x: x['val_loss'])
    
    print("\nBest Models:")
    print(f"Lowest MAE: Version {best_mae['version']} ({best_mae['mae']:.2f} kWh)")
    print(f"Lowest RMSE: Version {best_rmse['version']} ({best_rmse['rmse']:.2f} kWh)")
    print(f"Lowest Val Loss: Version {best_val['version']} ({best_val['val_loss']:.4f})")

if __name__ == "__main__":
    compare_models()
