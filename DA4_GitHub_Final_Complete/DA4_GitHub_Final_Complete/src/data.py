"""
Data loading and extraction utilities.
"""

import zipfile
from pathlib import Path
import pandas as pd
from .config import DATA_EXTRACT_DIR, CANDIDATE_DIRS


def find_file(filename):
    """
    Locate an uploaded ZIP file in candidate directories.
    
    Args:
        filename (str): Name of the file to find
        
    Returns:
        Path: Path to the found file
        
    Raises:
        FileNotFoundError: If file not found in any candidate directory
    """
    for d in CANDIDATE_DIRS:
        p = Path(d) / filename
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find '{filename}'. Upload it and place it in one of: {CANDIDATE_DIRS}"
    )


def extract_datasets():
    """
    Extract both ZIP archives (Dataset A and Dataset B).
    Dataset B contains nested zips that are also extracted.
    
    Returns:
        tuple: (path_to_dataset_A_csv, path_to_dataset_B_csv)
    """
    # Locate ZIP files
    zip_a = find_file("archive (1).zip")  # JanataHack Customer Segmentation
    zip_b = find_file("bank+marketing.zip")  # UCI Bank Marketing
    
    print("Dataset A zip:", zip_a.resolve())
    print("Dataset B zip:", zip_b.resolve())
    
    # Create extraction directories
    dir_a = DATA_EXTRACT_DIR / "dataset_A"
    dir_b_raw = DATA_EXTRACT_DIR / "dataset_B_raw"
    dir_b = DATA_EXTRACT_DIR / "dataset_B"
    for d in [dir_a, dir_b_raw, dir_b]:
        d.mkdir(exist_ok=True)
    
    # Extract Dataset A
    with zipfile.ZipFile(zip_a) as z:
        z.extractall(dir_a)
    
    # Extract Dataset B (and nested zips)
    with zipfile.ZipFile(zip_b) as z:
        z.extractall(dir_b_raw)
    
    # Extract nested zips in Dataset B
    for nested_zip in dir_b_raw.rglob("*.zip"):
        with zipfile.ZipFile(nested_zip) as z:
            z.extractall(dir_b / nested_zip.stem)
    
    print("Dataset A files:", [p.name for p in dir_a.rglob("*.csv")])
    print("Dataset B files:", [p.name for p in dir_b.rglob("*.csv")])
    
    # Resolve the exact CSV files
    train_a_path = next(dir_a.rglob("Train.csv"))
    bank_b_path = next(dir_b.rglob("bank-additional-full.csv"))
    
    print("Dataset A (customer segmentation) source:", train_a_path)
    print("Dataset B (bank marketing) source:", bank_b_path)
    
    return train_a_path, bank_b_path


def load_dataset_a(csv_path):
    """
    Load Dataset A (Customer Segmentation) from CSV.
    
    Args:
        csv_path (str or Path): Path to Train.csv
        
    Returns:
        pd.DataFrame: Customer segmentation data
    """
    return pd.read_csv(csv_path)


def load_dataset_b(csv_path):
    """
    Load Dataset B (Bank Marketing) from CSV.
    
    Args:
        csv_path (str or Path): Path to bank-additional-full.csv
        
    Returns:
        pd.DataFrame: Bank marketing data
    """
    return pd.read_csv(csv_path, sep=";")
