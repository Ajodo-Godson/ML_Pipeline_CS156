"""
Audio Feature Extraction - Improved Batch Processing
Saves progress incrementally to avoid data loss
"""

import librosa
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
import sys
warnings.filterwarnings('ignore')

# Import the AudioFeatureExtractor class
sys.path.insert(0, str(Path(__file__).parent))
from extract_audio_features import AudioFeatureExtractor


def extract_features_batch(audio_dir, output_csv, batch_size=100):
    """
    Extract features from all audio files with incremental saving.
    
    Args:
        audio_dir (str): Path to directory containing audio files
        output_csv (str): Path to save extracted features CSV
        batch_size (int): Save progress every N files
    """
    audio_dir = Path(audio_dir)
    output_csv = Path(output_csv)
    
    # Supported audio formats
    audio_extensions = ['.mp3', '.wav', '.flac', '.ogg', '.m4a']
    
    # Find all audio files
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(audio_dir.glob(f'**/*{ext}'))
    
    print(f"Found {len(audio_files)} audio files")
    
    if len(audio_files) == 0:
        print("No audio files found!")
        return
    
    # Initialize extractor
    extractor = AudioFeatureExtractor()
    
    # Check if output file already exists (resume mode)
    if output_csv.exists():
        print(f"\nResuming from existing file: {output_csv}")
        existing_df = pd.read_csv(output_csv)
        processed_paths = set(existing_df['audio_path'].values)
        print(f"Already processed: {len(processed_paths)} files")
    else:
        existing_df = None
        processed_paths = set()
    
    # Extract features from each file
    features_list = []
    successful = 0
    failed = 0
    skipped = 0
    
    for idx, audio_file in enumerate(audio_files, start=1):
        audio_path_str = str(audio_file)
        
        # Skip if already processed
        if audio_path_str in processed_paths:
            skipped += 1
            if idx % 100 == 0:
                print(f"[{idx}/{len(audio_files)}] Skipping (already processed): {audio_file.name}")
            continue
        
        print(f"[{idx}/{len(audio_files)}] Processing: {audio_file.name}")
        
        features = extractor.extract_all_features(audio_file)
        
        if features:
            features_list.append(features)
            successful += 1
        else:
            failed += 1
            print(f" Failed to extract features")
        
        # Save progress every batch_size files
        if len(features_list) >= batch_size:
            print(f"\n Saving progress... ({len(features_list)} new features)")
            batch_df = pd.DataFrame(features_list)
            
            if existing_df is not None:
                # Append to existing
                combined_df = pd.concat([existing_df, batch_df], ignore_index=True)
            else:
                combined_df = batch_df
            
            combined_df.to_csv(output_csv, index=False)
            print(f" Saved to {output_csv}")
            print(f"   Total processed so far: {len(combined_df)}\n")
            
            # Update existing_df and reset features_list
            existing_df = combined_df
            features_list = []
    
    # Save any remaining features
    if len(features_list) > 0:
        print(f"\n Saving final batch... ({len(features_list)} features)")
        batch_df = pd.DataFrame(features_list)
        
        if existing_df is not None:
            combined_df = pd.concat([existing_df, batch_df], ignore_index=True)
        else:
            combined_df = batch_df
        
        combined_df.to_csv(output_csv, index=False)
    elif existing_df is not None:
        combined_df = existing_df
    else:
        print("No features extracted!")
        return None
    
    # Final summary
    print("\n" + "="*60)
    print(" Feature Extraction Complete!")
    print("="*60)
    print(f"Features saved to: {output_csv}")
    print(f"Total tracks processed: {len(combined_df)}")
    print(f"Total features per track: {len(combined_df.columns)}")
    print(f"\n Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"⏭  Skipped (already done): {skipped}")
    
    # Print summary statistics
    print("\n=== Feature Statistics ===")
    if 'tempo' in combined_df.columns:
        print(f"Average Tempo: {combined_df['tempo'].mean():.2f} BPM")
    if 'danceability' in combined_df.columns:
        print(f"Average Danceability: {combined_df['danceability'].mean():.3f}")
    if 'energy' in combined_df.columns:
        print(f"Average Energy: {combined_df['energy'].mean():.3f}")
    if 'valence' in combined_df.columns:
        print(f"Average Valence: {combined_df['valence'].mean():.3f}")
    
    return combined_df


if __name__ == "__main__":
    import sys
    
    # Example usage
    if len(sys.argv) > 1:
        audio_directory = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else "extracted_audio_features.csv"
        batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    else:
        # Default
        audio_directory = "Audio_Samples"
        output_file = "Ingested_Data/audio_features.csv"
        batch_size = 100
    
    extract_features_batch(audio_directory, output_file, batch_size)
