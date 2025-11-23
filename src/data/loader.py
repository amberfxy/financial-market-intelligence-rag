"""Data loading utilities for Kaggle dataset."""

import pandas as pd
import os
from pathlib import Path
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


def load_kaggle_dataset(data_dir: str = "data/raw") -> pd.DataFrame:
    """
    Load the Kaggle Stock Market News Dataset.
    
    Args:
        data_dir: Directory containing the dataset files
        
    Returns:
        DataFrame with columns: Date, News Headline, etc.
    """
    data_path = Path(data_dir)
    
    # Look for CSV files
    csv_files = list(data_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {data_dir}. "
            "Please download the dataset using: "
            "kaggle datasets download -d aaron7sun/stocknews"
        )
    
    # Load the first CSV file found
    df = pd.read_csv(csv_files[0])
    logger.info(f"Loaded {len(df)} entries from {csv_files[0]}")
    
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the dataset: clean text, remove duplicates, normalize.
    
    Args:
        df: Raw dataframe
        
    Returns:
        Cleaned dataframe
    """
    # Remove duplicates
    initial_count = len(df)
    df = df.drop_duplicates()
    logger.info(f"Removed {initial_count - len(df)} duplicate entries")
    
    # Clean text columns
    text_columns = df.select_dtypes(include=['object']).columns
    
    for col in text_columns:
        if col in df.columns:
            # Remove HTML tags and normalize whitespace
            df[col] = df[col].astype(str).str.replace(r'<[^>]+>', '', regex=True)
            df[col] = df[col].str.strip()
            df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
    
    # Remove rows with empty text
    if 'News Headline' in df.columns:
        df = df[df['News Headline'].str.len() > 0]
    
    logger.info(f"Final dataset size: {len(df)} entries")
    
    return df


def save_processed_data(df: pd.DataFrame, output_path: str = "data/processed/processed_data.csv"):
    """Save processed data to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved processed data to {output_path}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    df = load_kaggle_dataset()
    df_processed = preprocess_data(df)
    save_processed_data(df_processed)

