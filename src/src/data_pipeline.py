import os
import json
import zipfile
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

def configure_kaggle_credentials():
    """
    Safely pulls sensitive credential constants from the local environment 
    and initializes the hidden system API configuration pathway.
    """
    print("Configuring Kaggle authentication parameters...")
    
    # Safely fetch strings from your private local .env file
    username = os.getenv("KAGGLE_USERNAME")
    token = os.getenv("KAGGLE_KEY")
    
    if not username or not token:
        print("Warning: Kaggle credentials missing from system environment variables.")
        return False

    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    credential_file = kaggle_dir / "kaggle.json"
    
    # Package parameters into standard secure JSON formatting
    credentials = {
        "username": username, 
        "key": token
    }
    
    with open(credential_file, "w") as f:
        json.dump(credentials, f)
        
    if os.name == "posix":
        credential_file.chmod(0o600)
        
    print("Credentials configured successfully from environment values!")
    return True

def download_and_extract_mri_dataset(target_data_dir):
    """
    Connects to the Kaggle API database, downloads the brain tumor imagery, 
    and handles programmatic system archive extraction.
    """
    base_data_path = Path(target_data_dir).resolve().parent
    base_data_path.mkdir(parents=True, exist_ok=True)
    zip_file_path = base_data_path / "brain-tumor-mri-dataset.zip"

    print("Connecting to Kaggle API and downloading dataset...")
    try:
        import kaggle
        kaggle.api.dataset_download_files(
            "masoudnickparvar/brain-tumor-mri-dataset",
            path=str(base_data_path),
            unzip=False
        )
    except Exception as e:
        print(f"Failed to complete download via Kaggle API. Error: {e}")
        return False

    if zip_file_path.exists():
        print("Extracting raw anatomical files to local directory...")
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(base_data_path)
        
        os.remove(zip_file_path)
        print("MRI Dataset is fully updated and ready!")
        return True
    else:
        print("Download file payload was not found.")
        return False

def generate_mri_metadata(base_dir, categories):
    """
    Ingests raw image files and extracts deep statistical
    and spatial metadata into a unified Pandas DataFrame.
    """
    all_metadata = []
    print("Extracting Biomedical Data Integrity Metrics...")

    for cat in categories:
        cat_path = os.path.join(base_dir, cat)
        if not os.path.exists(cat_path):
            continue
        images = os.listdir(cat_path)

        for img_name in images:
            img_path = os.path.join(cat_path, img_name)
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
                    matrix = np.array(img.convert('L'))
                    mean_brightness = np.mean(matrix)
                    std_brightness = np.std(matrix)
                    is_corrupt = 1 if mean_brightness < 2.0 or std_brightness < 1.0 else 0

                    all_metadata.append({
                        "FileName": img_name, "Class": cat, "Path": img_path,
                        "Width": width, "Height": height, "AspectRatio": width / height,
                        "MeanIntensity": mean_brightness, "StdIntensity": std_brightness,
                        "ArtifactFlag": is_corrupt
                    })
            except Exception:
                continue

    return pd.DataFrame(all_metadata)