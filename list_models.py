import json
from tabulate import tabulate

def list_model_versions():
    try:
        with open('model_versions.json', 'r') as f:
            versions = json.load(f)
        
        # Prepare table data
        headers = ['Version', 'Date', 'MAE (kWh)', 'RMSE (kWh)', 'Val Loss', 'Train Loss']
        table_data = [
            [v['version'], 
             v['timestamp'], 
             f"{v['mae']:.2f}", 
             f"{v['rmse']:.2f}", 
             f"{v['val_loss']:.4f}",
             f"{v['train_loss']:.4f}"]
            for v in versions
        ]
        
        # Sort versions by RMSE (best performing first)
        table_data.sort(key=lambda x: float(x[3]))
        
        print("\nModel Version History (sorted by RMSE):")
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
        
        # Print best version
        best_version = min(versions, key=lambda x: x['rmse'])
        print(f"\nBest Model: Version {best_version['version']}")
        print(f"File: {best_version['model_file']}")
        print(f"MAE: {best_version['mae']:.2f} kWh")
        print(f"RMSE: {best_version['rmse']:.2f} kWh")
        
    except FileNotFoundError:
        print("No model versions found. Train a model first.")

if __name__ == "__main__":
    list_model_versions()
