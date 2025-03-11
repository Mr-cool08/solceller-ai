import os
import shutil

def organize_files():
    # Create directories
    model_dir = 'models'
    training_dir = 'training_artifacts'
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(training_dir, exist_ok=True)
    
    # Move files to appropriate directories
    for file in os.listdir('.'):
        if file.endswith('.pth') or file.endswith('.joblib') or file == 'model_versions.json':
            shutil.move(file, os.path.join(model_dir, file))
        elif file.endswith('.png'):
            shutil.move(file, os.path.join(training_dir, file))

if __name__ == "__main__":
    organize_files()
