"""
Track Information Fetcher
Fetches track metadata and genres from Spotify API
Audio features are extracted locally using signal processing (Spotify API deprecated)
"""

import pandas as pd
import time
import requests
import os
import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import sp

# Import local audio feature extractor
try:
    from Ingestion.extract_audio_features import AudioFeatureExtractor
    AUDIO_EXTRACTION_AVAILABLE = True
except ImportError:
    print("Warning: Audio feature extraction not available. Install librosa: pip install librosa")
    AUDIO_EXTRACTION_AVAILABLE = False

class SpotifyTrackFetcher:
    """Handles fetching track information from Spotify API and extracting audio features locally"""
    
    def __init__(self, rate_limit_delay=0.1, audio_dir=None):
        """
        Initialize the track fetcher.
        
        Args:
            rate_limit_delay (float): Delay between API calls in seconds
            audio_dir (str): Directory containing audio files for feature extraction
        """
        self.sp = sp
        self.rate_limit_delay = rate_limit_delay
        self.cache = {}  # Cache to avoid duplicate API calls
        self.audio_dir = Path(audio_dir) if audio_dir else None
        
        # Initialize audio feature extractor if available
        if AUDIO_EXTRACTION_AVAILABLE and self.audio_dir:
            self.audio_extractor = AudioFeatureExtractor()
        else:
            self.audio_extractor = None
    
    def search_track(self, track_name, artist_name, retries=3):
        """
        Search for a track on Spotify.
        
        Args:
            track_name (str): Name of the track
            artist_name (str): Name of the artist
            retries (int): Number of retry attempts
            
        Returns:
            dict: Track information or None if not found
        """
        # Create cache key
        cache_key = f"{artist_name}||{track_name}".lower()
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        query = f"track:{track_name} artist:{artist_name}"
        
        for attempt in range(retries):
            try:
                results = self.sp.search(q=query, type='track', limit=1)
                
                if results['tracks']['items']:
                    track = results['tracks']['items'][0]
                    self.cache[cache_key] = track
                    time.sleep(self.rate_limit_delay)
                    return track
                else:
                    self.cache[cache_key] = None
                    return None
                    
            except Exception as e:
                if attempt < retries - 1:
                    wait_time = (attempt + 1) * 2  # Exponential backoff
                    print(f"Error searching for '{track_name}' by '{artist_name}': {e}")
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"Failed to fetch '{track_name}' by '{artist_name}' after {retries} attempts")
                    self.cache[cache_key] = None
                    return None
        
        return None
    
    def get_track_features(self, track_id, audio_file_path=None, retries=3):
        """
        Get audio features for a track using local extraction from audio file.
        
        Note: Spotify's audio features API endpoint is deprecated for new applications.
        This method extracts features from local audio files using signal processing.
        
        Args:
            track_id (str): Spotify track ID
            audio_file_path (str): Path to audio file for feature extraction
            retries (int): Number of retry attempts
            
        Returns:
            dict: Audio features or None if failed
        """
        # If audio file provided, extract features locally
        if audio_file_path and self.audio_extractor:
            try:
                print(f"Extracting audio features from: {audio_file_path}")
                features = self.audio_extractor.extract_all_features(audio_file_path)
                if features:
                    # Map our extracted features to Spotify-like format
                    return {
                        'danceability': features.get('danceability'),
                        'energy': features.get('energy'),
                        'valence': features.get('valence'),
                        'acousticness': features.get('acousticness'),
                        'instrumentalness': features.get('instrumentalness'),
                        'tempo': features.get('tempo'),
                        'loudness': features.get('loudness'),
                        'spectral_centroid': features.get('spectral_centroid_mean'),
                        'duration_ms': features.get('duration_ms'),
                        # Additional features not in Spotify API
                        'rms_energy': features.get('rms_mean'),
                        'key': features.get('dominant_pitch_class'),
                    }
            except Exception as e:
                print(f"Error extracting audio features locally: {e}")
        
        return None
    
    def get_artist_genres(self, artist_id, retries=3):
        """
        Get genres for an artist.
        
        Args:
            artist_id (str): Spotify artist ID
            retries (int): Number of retry attempts
            
        Returns:
            list: List of genres or empty list if failed
        """
        for attempt in range(retries):
            try:
                artist_info = self.sp.artist(artist_id)
                time.sleep(self.rate_limit_delay)
                return artist_info.get('genres', [])
            except Exception as e:
                if attempt < retries - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"Error fetching artist genres: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"Failed to fetch genres for artist {artist_id}")
                    return []
        
        return []
    
    def get_track_info(self, track_name, artist_name):
        """
        Get comprehensive track information including metadata, audio features, and genres.
        
        Args:
            track_name (str): Name of the track
            artist_name (str): Name of the artist
            
        Returns:
            dict: Dictionary with track info, or None if not found
        """
        track = self.search_track(track_name, artist_name)
        
        if not track:
            return None
        
        track_id = track['id']
        artist_id = track['artists'][0]['id']
        
        # Get audio features
        audio_features = self.get_track_features(track_id)
        
        # Get artist genres
        genres = self.get_artist_genres(artist_id)
        
        # Compile all information
        track_info = {
            'track_id': track_id,
            'track_uri': track['uri'],
            'track_name_spotify': track['name'],
            'artist_name_spotify': track['artists'][0]['name'],
            'artist_id': artist_id,
            'album_name': track['album']['name'],
            'release_date': track['album']['release_date'],
            'popularity': track['popularity'],
            'preview_url': track.get('preview_url'),
            'duration_ms': track['duration_ms'],
            'explicit': track['explicit'],
            'genres': ', '.join(genres) if genres else 'Unknown',
            'primary_genre': genres[0] if genres else 'Unknown'
        }
        
        # Add audio features if available
        if audio_features:
            track_info.update({
                'danceability': audio_features.get('danceability'),
                'energy': audio_features.get('energy'),
                'key': audio_features.get('key'),
                'loudness': audio_features.get('loudness'),
                'mode': audio_features.get('mode'),
                'speechiness': audio_features.get('speechiness'),
                'acousticness': audio_features.get('acousticness'),
                'instrumentalness': audio_features.get('instrumentalness'),
                'liveness': audio_features.get('liveness'),
                'valence': audio_features.get('valence'),
                'tempo': audio_features.get('tempo'),
                'time_signature': audio_features.get('time_signature')
            })
        
        return track_info


def enrich_streaming_data(cleaned_csv_path, output_csv_path, max_tracks=None):
    """
    Enrich cleaned streaming history with Spotify track information.
    
    Args:
        cleaned_csv_path (str): Path to cleaned streaming history CSV
        output_csv_path (str): Path to save enriched data
        max_tracks (int): Maximum number of tracks to process (for testing)
    """
    print("Loading cleaned streaming history...")
    df = pd.read_csv(cleaned_csv_path)
    
    if max_tracks:
        df = df.head(max_tracks)
        print(f"Processing first {max_tracks} tracks for testing...")
    
    print(f"Total tracks to enrich: {len(df)}")
    
    # Get unique track-artist combinations
    unique_tracks = df[['trackName', 'artistName']].drop_duplicates()
    print(f"Unique tracks: {len(unique_tracks)}")
    
    # Initialize fetcher
    fetcher = SpotifyTrackFetcher(rate_limit_delay=0.2)
    
    # Fetch track info for each unique track
    track_info_list = []
    
    for idx, row in unique_tracks.iterrows():
        track_name = row['trackName']
        artist_name = row['artistName']
        
        if idx % 10 == 0:
            print(f"Processing track {idx + 1}/{len(unique_tracks)}...")
        
        track_info = fetcher.get_track_info(track_name, artist_name)
        
        if track_info:
            track_info['original_track_name'] = track_name
            track_info['original_artist_name'] = artist_name
            track_info_list.append(track_info)
    
    print(f"Successfully fetched info for {len(track_info_list)} tracks")
    
    # Create DataFrame from track info
    track_info_df = pd.DataFrame(track_info_list)
    
    # Merge with original streaming data
    enriched_df = df.merge(
        track_info_df,
        left_on=['trackName', 'artistName'],
        right_on=['original_track_name', 'original_artist_name'],
        how='left'
    )
    
    # Save enriched data
    enriched_df.to_csv(output_csv_path, index=False)
    print(f"\nEnriched data saved to: {output_csv_path}")
    print(f"Total rows: {len(enriched_df)}")
    print(f"Columns: {len(enriched_df.columns)}")
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Tracks with genre info: {enriched_df['primary_genre'].notna().sum()}")
    print(f"Tracks with preview URL: {enriched_df['preview_url'].notna().sum()}")
    print(f"Tracks with audio features: {enriched_df['danceability'].notna().sum()}")
    
    if enriched_df['primary_genre'].notna().sum() > 0:
        print("\nTop 10 Genres:")
        print(enriched_df['primary_genre'].value_counts().head(10))
    
    return enriched_df


if __name__ == "__main__":
    # Change to project root directory
    os.chdir(Path(__file__).parent.parent)
    
    # Test with a small sample first
    cleaned_data_path = "Ingested_Data/cleaned_streaming_history.csv"
    output_path = "Ingested_Data/enriched_streaming_history.csv"
    
    # Check if cleaned data exists
    if not os.path.exists(cleaned_data_path):
        print(f"Error: {cleaned_data_path} not found.")
        print("Please run 'python Ingestion/clean.py' first to create the cleaned data.")
        sys.exit(1)
    
    # Start with 50 tracks for testing
    print("Testing with first 50 tracks...")
    enrich_streaming_data(cleaned_data_path, output_path, max_tracks=50)
    
    print("\nTest complete! Review the output and then run with max_tracks=None to process all data.")
