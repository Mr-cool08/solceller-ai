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

def calculate_overfitting_score(train_loss, val_loss):
    """Calculate overfitting score based on loss difference"""
    return abs(train_loss - val_loss) / val_loss * 100  # As percentage

def calculate_model_score(version):
    """Calculate overall model score (0-100%)"""
    # Weightings for different metrics
    weights = {
        'mae_weight': 0.3,      # Mean Absolute Error
        'rmse_weight': 0.3,     # Root Mean Square Error
        'val_weight': 0.2,      # Validation Loss
        'balance_weight': 0.2   # Training-Validation Balance
    }
    
    # Calculate balance score (how close train and val losses are)
    balance_score = 100 * (1 - abs(version['train_loss'] - version['val_loss']) / version['val_loss'])
    
    # Get the best scores across all versions for normalization
    all_versions = load_model_versions()
    best_mae = min(v['mae'] for v in all_versions)
    best_rmse = min(v['rmse'] for v in all_versions)
    best_val = min(v['val_loss'] for v in all_versions)
    
    # Calculate normalized scores (0-100%)
    mae_score = 100 * (best_mae / version['mae'])
    rmse_score = 100 * (best_rmse / version['rmse'])
    val_score = 100 * (best_val / version['val_loss'])
    
    # Calculate weighted average
    total_score = (
        mae_score * weights['mae_weight'] +
        rmse_score * weights['rmse_weight'] +
        val_score * weights['val_weight'] +
        balance_score * weights['balance_weight']
    )
    
    return total_score

def get_grade(score):
    """Convert numerical score to letter grade"""
    if score >= 95: return 'A+'
    if score >= 90: return 'A'
    if score >= 85: return 'A-'
    if score >= 80: return 'B+'
    if score >= 75: return 'B'
    if score >= 70: return 'B-'
    if score >= 65: return 'C+'
    if score >= 60: return 'C'
    if score >= 55: return 'C-'
    if score >= 50: return 'D+'
    if score >= 45: return 'D'
    return 'F'

def compare_models():
    versions = load_model_versions()
    if not versions:
        return
    
    # Calculate scores for all versions
    data = []
    for v in versions:
        score = calculate_model_score(v)
        grade = get_grade(score)
        data.append([
            v['version'],
            v['timestamp'],
            f"{v['mae']:.2f}",
            f"{v['rmse']:.2f}",
            f"{v['val_loss']:.4f}",
            f"{v['train_loss']:.4f}",
            f"{abs(v['train_loss'] - v['val_loss']):.4f}",
            f"{score:.1f}%",
            grade
        ])
    
    # Sort by score (descending)
    data.sort(key=lambda x: float(x[7].rstrip('%')), reverse=True)
    
    # Print table
    headers = ['Version', 'Date', 'MAE', 'RMSE', 'Val Loss', 'Train Loss', 
              'Loss Diff', 'Score', 'Grade']
    print("\nModel Performance Ranking:")
    print(tabulate(data, headers=headers, tablefmt='grid'))
    
    # Plot scores
    plt.figure(figsize=(12, 6))
    versions_nums = [row[0] for row in data]
    scores = [float(row[7].rstrip('%')) for row in data]
    grades = [row[8] for row in data]
    
    plt.bar(versions_nums, scores)
    plt.axhline(y=90, color='g', linestyle='--', alpha=0.5, label='A Grade (90%)')
    plt.axhline(y=80, color='y', linestyle='--', alpha=0.5, label='B Grade (80%)')
    plt.axhline(y=70, color='orange', linestyle='--', alpha=0.5, label='C Grade (70%)')
    plt.axhline(y=60, color='r', linestyle='--', alpha=0.5, label='D Grade (60%)')
    
    # Add grade labels on top of bars
    for i, (score, grade) in enumerate(zip(scores, grades)):
        plt.text(versions_nums[i], score + 1, grade, ha='center')
    
    plt.xlabel('Model Version')
    plt.ylabel('Overall Score (%)')
    plt.title('Model Performance Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(ANALYSIS_DIR, 'model_scores.png'))
    plt.close()

if __name__ == "__main__":
    compare_models()
