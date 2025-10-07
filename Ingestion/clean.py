import pandas as pd
import os
import sys
from datetime import datetime
from pathlib import Path


if __name__ == "__main__":
    os.chdir(Path(__file__).parent.parent)

# Configuration
input_file = "Ingested_Data/combined_streaming_history.csv"
output_file = "Ingested_Data/cleaned_streaming_history.csv"

def load_data(file_path):
    """Load the CSV file into a pandas DataFrame."""
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} rows from {file_path}")
        return df
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def parse_datetime(df):
    """Parse endTime column to datetime format."""
    try:
        df['endTime'] = pd.to_datetime(df['endTime'], format='%Y-%m-%d %H:%M', errors='coerce')
        # Drop rows where endTime couldn't be parsed
        invalid_times = df['endTime'].isna().sum()
        if invalid_times > 0:
            print(f"Dropped {invalid_times} rows with invalid endTime")
            df = df.dropna(subset=['endTime'])
        print(f"Parsed endTime to datetime. Remaining rows: {len(df)}")
        return df
    except Exception as e:
        print(f"Error parsing datetime: {e}")
        return df

def convert_ms_to_time(df):
    """Convert msPlayed to secondsPlayed and add minutesPlayed."""
    try:
        df['secondsPlayed'] = df['msPlayed'] / 1000
        df['minutesPlayed'] = df['secondsPlayed'] / 60
        print("Converted msPlayed to secondsPlayed and minutesPlayed")
        return df
    except Exception as e:
        print(f"Error converting msPlayed: {e}")
        return df

def filter_data(df):
    """Filter out invalid and short plays."""
    initial_count = len(df)
    
    # Remove unknown artists/tracks
    df = df[~df['artistName'].str.contains('Unknown', na=False)]
    df = df[~df['trackName'].str.contains('Unknown', na=False)]
    
    # Filter out plays less than 30 seconds (Spotify's play threshold)
    df = df[df['secondsPlayed'] >= 30]
    
    filtered_count = initial_count - len(df)
    print(f"Filtered out {filtered_count} rows. Remaining rows: {len(df)}")
    return df

def clean_names(df):
    """Clean artist and track names."""
    try:
        # Remove extra quotes around names
        df['artistName'] = df['artistName'].str.strip('"')
        df['trackName'] = df['trackName'].str.strip('"')
        
        # Remove any leading/trailing whitespace
        df['artistName'] = df['artistName'].str.strip()
        df['trackName'] = df['trackName'].str.strip()
        
        print("Cleaned artist and track names")
        return df
    except Exception as e:
        print(f"Error cleaning names: {e}")
        return df

def add_datetime_features(df):
    """Add derived datetime features."""
    try:
        df['date'] = df['endTime'].dt.date
        df['hour'] = df['endTime'].dt.hour
        df['day_of_week'] = df['endTime'].dt.day_name()
        df['month'] = df['endTime'].dt.month
        df['year'] = df['endTime'].dt.year
        
        print("Added derived datetime features")
        return df
    except Exception as e:
        print(f"Error adding datetime features: {e}")
        return df

def remove_duplicates(df):
    """Remove duplicate entries."""
    initial_count = len(df)
    
    # Remove duplicates based on endTime, artistName, and trackName
    df = df.drop_duplicates(subset=['endTime', 'artistName', 'trackName'], keep='first')
    
    duplicates_removed = initial_count - len(df)
    print(f"Removed {duplicates_removed} duplicate entries. Remaining rows: {len(df)}")
    return df

def save_cleaned_data(df, output_path):
    """Save the cleaned DataFrame to a CSV file."""
    try:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to {output_path}")
        print(f"Total rows in cleaned dataset: {len(df)}")
        return True
    except Exception as e:
        print(f"Error saving data: {e}")
        return False

if __name__ == "__main__":
    # Load the data
    df = load_data(input_file)
    if df is None:
        exit(1)
    
    # Parse datetime
    df = parse_datetime(df)
    
    # Convert msPlayed to time units
    df = convert_ms_to_time(df)
    
    # Filter invalid and short plays
    df = filter_data(df)
    
    # Clean artist and track names
    df = clean_names(df)
    
    # Add derived datetime features
    df = add_datetime_features(df)
    
    # Remove duplicates
    df = remove_duplicates(df)
    
    # Save cleaned data
    save_cleaned_data(df, output_file)
    
    print("\n=== Data Cleaning Complete ===")
    print(f"Original columns: endTime, artistName, trackName, msPlayed")
    print(f"Added columns: secondsPlayed, minutesPlayed, date, hour, day_of_week, month, year")
    print(f"Final dataset shape: {df.shape}")