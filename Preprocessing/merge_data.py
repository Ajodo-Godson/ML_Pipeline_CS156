"""
Data Merger - Combine Streaming History with Audio Features
Merges listening events with extracted audio features to create master dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def clean_filename_to_track_artist(filename):
    """
    Extract track name and artist from filename.
    Format: "Artist - Track Name.mp3"
    
    Args:
        filename (str): Audio file name
        
    Returns:
        tuple: (artist_name, track_name)
    """
    # Remove "Audio_Samples/" prefix and ".mp3" suffix
    filename = Path(filename).stem
    
    # Split by " - " to get artist and track
    if " - " in filename:
        parts = filename.split(" - ", 1)
        artist = parts[0].strip()
        track = parts[1].strip()
        return artist, track
    else:
        return None, None


def normalize_text(text):
    """
    Normalize text for better matching (lowercase, remove extra spaces).
    """
    if pd.isna(text):
        return ""
    return str(text).lower().strip()


def merge_datasets(streaming_csv, audio_features_csv, output_csv):
    """
    Merge streaming history with audio features.
    
    Args:
        streaming_csv (str): Path to cleaned streaming history CSV
        audio_features_csv (str): Path to audio features CSV
        output_csv (str): Path to save merged dataset
        
    Returns:
        pd.DataFrame: Merged dataset
    """
    print("="*60)
    print(" Dataset Merger")
    print("="*60)
    
    # Load datasets
    print(f"\n  Loading streaming history: {streaming_csv}")
    streaming_df = pd.read_csv(streaming_csv)
    print(f"    Loaded {len(streaming_df):,} listening events")
    print(f"    Date range: {streaming_df['date'].min()} to {streaming_df['date'].max()}")
    
    print(f"\n  Loading audio features: {audio_features_csv}")
    audio_df = pd.read_csv(audio_features_csv)
    print(f"    Loaded features for {len(audio_df):,} tracks")
    print(f"    Features per track: {len(audio_df.columns)}")
    
    # Extract artist and track from audio_path
    print(f"\n Extracting artist and track info from audio filenames...")
    audio_df[['extracted_artist', 'extracted_track']] = audio_df['audio_path'].apply(
        lambda x: pd.Series(clean_filename_to_track_artist(x))
    )
    
    # Normalize for matching
    audio_df['artist_normalized'] = audio_df['extracted_artist'].apply(normalize_text)
    audio_df['track_normalized'] = audio_df['extracted_track'].apply(normalize_text)
    
    streaming_df['artist_normalized'] = streaming_df['artistName'].apply(normalize_text)
    streaming_df['track_normalized'] = streaming_df['trackName'].apply(normalize_text)
    
    # Drop rows where extraction failed
    audio_df_clean = audio_df[audio_df['extracted_artist'].notna()].copy()
    print(f"    Successfully extracted info from {len(audio_df_clean):,} tracks")
    
    # Merge datasets (inner join: only streaming events with audio features)
    print(f"\n  Merging datasets on artist + track name (only events with audio samples)...")
    merged_df = streaming_df.merge(
        audio_df_clean,
        left_on=['artist_normalized', 'track_normalized'],
        right_on=['artist_normalized', 'track_normalized'],
        how='inner'  # Only keep rows with audio features
    )

    matched_rows = len(merged_df)
    print(f"    Merged successfully!")
    print(f"    Matched listening events: {matched_rows:,} (100%)")

    unique_matched_tracks = merged_df[['trackName', 'artistName']].drop_duplicates()
    print(f" Unique tracks with features: {len(unique_matched_tracks):,}")

    # Clean up temporary columns
    columns_to_drop = ['artist_normalized', 'track_normalized', 'extracted_artist', 'extracted_track']
    merged_df = merged_df.drop(columns=columns_to_drop, errors='ignore')

    # Save merged dataset
    print(f"\n  Saving merged training dataset to: {output_csv}")
    merged_df.to_csv(output_csv, index=False)
    print(f"   Saved {len(merged_df):,} rows with {len(merged_df.columns)} columns")

    # Summary statistics
    print("\n" + "="*60)
    print(" Merge Summary")
    print("="*60)
    print(f"Total listening events (with audio): {len(merged_df):,}")
    print(f"\nColumns in merged dataset: {len(merged_df.columns)}")

    # Show column categories
    streaming_cols = ['endTime', 'artistName', 'trackName', 'msPlayed', 'secondsPlayed', 
                      'minutesPlayed', 'date', 'hour', 'day_of_week', 'month', 'year']
    audio_feature_cols = [col for col in merged_df.columns if col not in streaming_cols and col != 'audio_path']

    print(f"\nColumn breakdown:")
    print(f"  - Streaming data columns: {len([c for c in streaming_cols if c in merged_df.columns])}")
    print(f"  - Audio feature columns: {len(audio_feature_cols)}")

    # Sample of audio features available
    if len(audio_feature_cols) > 0:
        print(f"\nAudio features included:")
        for col in ['tempo', 'danceability', 'energy', 'valence', 'acousticness', 'instrumentalness']:
            if col in merged_df.columns:
                non_null = merged_df[col].notna().sum()
                print(f"  - {col}: {non_null:,} non-null values")

    print("\n Training dataset merge complete!")
    print("="*60)

    return merged_df


def analyze_merge_quality(merged_df):
    """
    Analyze the quality of the merge and identify potential issues.
    """
    print("\n" + "="*60)
    print(" Merge Quality Analysis")
    print("="*60)
    
    # Check for unmatched popular tracks
    unmatched = merged_df[merged_df['audio_path'].isna()].copy()
    
    if len(unmatched) > 0:
        print(f"\n  Unmatched listening events: {len(unmatched):,}")
        
        # Top unmatched tracks
        top_unmatched = unmatched.groupby(['trackName', 'artistName']).size().sort_values(ascending=False).head(10)
        print(f"\nTop 10 most-listened unmatched tracks:")
        for (track, artist), count in top_unmatched.items():
            print(f"  - {artist} - {track}: {count} plays")
        
        print(f"\n Tip: These tracks don't have audio previews available from Spotify")
    
    # Check matched tracks
    matched = merged_df[merged_df['audio_path'].notna()]
    if len(matched) > 0:
        print(f"\n Matched listening events: {len(matched):,}")
        
        # Most listened tracks with features
        top_matched = matched.groupby(['trackName', 'artistName']).size().sort_values(ascending=False).head(10)
        print(f"\nTop 10 most-listened tracks (with features):")
        for (track, artist), count in top_matched.items():
            print(f"  - {artist} - {track}: {count} plays")


if __name__ == "__main__":
    # File paths
    streaming_csv = "Ingested_Data/cleaned_streaming_history.csv"
    audio_features_csv = "Ingested_Data/audio_features.csv"
    output_csv = "Ingested_Data/merged_dataset.csv"
    
    # Check if input files exist
    if not Path(streaming_csv).exists():
        print(f" Error: {streaming_csv} not found!")
        sys.exit(1)
    
    if not Path(audio_features_csv).exists():
        print(f" Error: {audio_features_csv} not found!")
        sys.exit(1)
    
    # Merge datasets
    merged_df = merge_datasets(streaming_csv, audio_features_csv, output_csv)
    
    # Analyze merge quality
    analyze_merge_quality(merged_df)
    
    print(f"\n Completed! Merged dataset is ready at: {output_csv}")
