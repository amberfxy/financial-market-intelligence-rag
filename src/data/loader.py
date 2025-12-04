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
    Supports multiple CSV files with different structures:
    - RedditNews.csv: has "News" column
    - Combined_News_DJIA.csv: has Top1-Top25 columns (merged into single text)
    
    Args:
        data_dir: Directory containing the dataset files
        
    Returns:
        DataFrame with standardized columns: Date, News Headline, source
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
    
    all_dataframes = []
    
    for csv_file in csv_files:
        logger.info(f"Loading {csv_file.name}...")
        df = pd.read_csv(csv_file)
        
        # Handle different file structures
        if "Combined_News_DJIA" in csv_file.name:
            # Combine Top1-Top25 columns into a single text column
            top_columns = [col for col in df.columns if col.startswith("Top") and col[3:].isdigit()]
            if top_columns:
                # Sort columns: Top1, Top2, ..., Top25
                top_columns = sorted(top_columns, key=lambda x: int(x[3:]))
                
                # Combine all Top columns into a single text column
                df["News Headline"] = df[top_columns].apply(
                    lambda row: " ".join([str(val) for val in row if pd.notna(val) and str(val).strip()]),
                    axis=1
                )
                
                # Remove empty rows
                df = df[df["News Headline"].str.len() > 0]
                
                # Add source identifier
                df["source"] = "Combined_News_DJIA"
                df["source_type"] = "combined_news"
                
                logger.info(f"  Combined {len(top_columns)} Top columns into 'News Headline'")
                logger.info(f"  Result: {len(df)} entries with {len(df.columns)} columns")
                
        elif "RedditNews" in csv_file.name:
            # Rename "News" column to "News Headline" for consistency
            if "News" in df.columns:
                df["News Headline"] = df["News"]
                # Optionally drop the original "News" column
                # df = df.drop(columns=["News"])
            
            # Add source identifier
            df["source"] = "RedditNews"
            df["source_type"] = "reddit_news"
            
            logger.info(f"  Renamed 'News' column to 'News Headline'")
            logger.info(f"  Result: {len(df)} entries")
            
        else:
            # For other files, try to find a text column
            # Check for common column names
            text_column = None
            for col_name in ["News Headline", "News", "Headline", "Text", "Content"]:
                if col_name in df.columns:
                    if col_name != "News Headline":
                        df["News Headline"] = df[col_name]
                    text_column = "News Headline"
                    break
            
            if text_column is None:
                logger.warning(f"  Could not find text column in {csv_file.name}. Available columns: {list(df.columns)}")
                # Skip this file or use first text-like column
                continue
            
            # Add source identifier
            df["source"] = csv_file.stem
            df["source_type"] = "other"
        
        # Ensure Date column exists (use first date-like column if available)
        if "Date" not in df.columns:
            date_columns = [col for col in df.columns if "date" in col.lower() or "time" in col.lower()]
            if date_columns:
                df["Date"] = df[date_columns[0]]
            else:
                df["Date"] = ""
        
        # Ensure News Headline column exists
        if "News Headline" not in df.columns:
            logger.warning(f"  No 'News Headline' column found in {csv_file.name}. Skipping.")
            continue
        
        all_dataframes.append(df)
        logger.info(f"  Successfully processed {csv_file.name}: {len(df)} entries")
    
    if not all_dataframes:
        raise ValueError("No valid data files found. Please check CSV file formats.")
    
    # Combine all dataframes
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    logger.info(f"Combined {len(all_dataframes)} files into {len(combined_df)} total entries")
    logger.info(f"Sources: {combined_df['source'].unique().tolist()}")
    
    return combined_df


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
    df = df.drop_duplicates(subset=['News Headline'], keep='first')
    logger.info(f"Removed {initial_count - len(df)} duplicate entries")
    
    # Clean News Headline column
    if 'News Headline' in df.columns:
        # Convert to string and clean
        df['News Headline'] = df['News Headline'].astype(str)
        
        # Remove HTML tags
        df['News Headline'] = df['News Headline'].str.replace(r'<[^>]+>', '', regex=True)
        
        # Normalize whitespace
        df['News Headline'] = df['News Headline'].str.strip()
        df['News Headline'] = df['News Headline'].str.replace(r'\s+', ' ', regex=True)
        
        # Remove rows with empty or very short text
        df = df[df['News Headline'].str.len() > 10]
        
        # Remove rows that are just "nan" or similar
        df = df[~df['News Headline'].str.lower().isin(['nan', 'none', 'null', ''])]
    
    # Clean Date column if it exists
    if 'Date' in df.columns:
        df['Date'] = df['Date'].astype(str).str.strip()
    
    logger.info(f"Final dataset size: {len(df)} entries")
    if 'source' in df.columns:
        logger.info(f"Source distribution: {df['source'].value_counts().to_dict()}")
    
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

