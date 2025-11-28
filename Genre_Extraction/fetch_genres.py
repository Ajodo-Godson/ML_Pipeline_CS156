"""
Fetch Genre Labels from Spotify API
Retrieves artist genre information to create labels for classification
FIXED VERSION with improved caching, parsing, and matching
"""

import pandas as pd
import sys
from pathlib import Path
import time
from collections import Counter
import json
import ast

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import sp


class GenreFetcher:
    """Fetch and process genre information from Spotify API"""
    
    def __init__(self):
        """Initialize Spotify client"""
        print("="*60)
        print(" Genre Fetcher - Spotify API (IMPROVED)")
        print("="*60)
        
        self.sp = sp
        print(" Spotify client initialized")
    
    def get_artist_genres(self, artist_name):
        """
        Get genres for an artist from Spotify API.
        Uses improved matching: searches multiple results, picks best match by name + popularity,
        then fetches full artist data by ID for complete genre information.
        
        Args:
            artist_name (str): Name of the artist
            
        Returns:
            tuple: (list of genres, status_message)
                   Status can be: 'found', 'not_found', 'no_genres'
        """
        try:
            # Search for artist (get top 10 results for better matching)
            results = self.sp.search(q=artist_name, type='artist', limit=10)
            
            if not results['artists']['items']:
                return [], 'not_found'
            
            # Find best match using fuzzy matching and popularity
            best_match = None
            best_score = -1
            
            artist_name_lower = artist_name.lower().strip()
            
            for artist in results['artists']['items']:
                artist_result_name_lower = artist['name'].lower().strip()
                
                # Calculate match score
                if artist_result_name_lower == artist_name_lower:
                    # Exact match: high base score + popularity bonus
                    score = 1000 + artist['popularity']
                elif artist_name_lower in artist_result_name_lower or artist_result_name_lower in artist_name_lower:
                    # Partial match: medium score + popularity bonus
                    score = 500 + artist['popularity']
                else:
                    # No match: just popularity
                    score = artist['popularity']
                
                if score > best_score:
                    best_match = artist
                    best_score = score
            
            if best_match:
                # Fetch full artist data by ID for complete genre information
                full_artist = self.sp.artist(best_match['id'])
                genres = full_artist.get('genres', [])
                
                if not genres:
                    return [], 'no_genres'
                
                return genres, 'found'
            
            return [], 'not_found'
        
        except Exception as e:
            print(f" Error fetching genres for {artist_name}: {e}")
            return [], 'error'
    
    def parse_genre_list(self, genre_str):
        """
        Safely parse genre list from string representation.
        
        Args:
            genre_str: String representation of list or actual list
            
        Returns:
            list: Parsed genre list
        """
        # Already a list
        if isinstance(genre_str, list):
            return genre_str
        
        # NaN or empty
        if pd.isna(genre_str) or genre_str == '' or genre_str == '[]':
            return []
        
        try:
            # Try JSON parsing first (safer)
            return json.loads(genre_str.replace("'", '"'))
        except:
            pass
        
        try:
            # Fall back to ast.literal_eval (safer than eval)
            return ast.literal_eval(genre_str)
        except:
            pass
        
        # Last resort: manual parsing
        try:
            # Remove brackets and split
            cleaned = genre_str.strip('[]').replace("'", "").replace('"', '')
            if not cleaned:
                return []
            return [g.strip() for g in cleaned.split(',') if g.strip()]
        except:
            return []
    
    def save_cache(self, artist_genres, cache_file):
        """
        Save artist genres to cache with proper JSON encoding.
        
        Args:
            artist_genres (dict): Dictionary of artist -> genres
            cache_file (str): Path to cache file
        """
        cache_df = pd.DataFrame([
            {
                'artist': k, 
                'genres': json.dumps(v) if isinstance(v, list) else v  # Properly serialize
            } 
            for k, v in artist_genres.items()
        ])
        cache_df.to_csv(cache_file, index=False)
    
    def load_cache(self, cache_file):
        """
        Load artist genres from cache with proper parsing.
        
        Args:
            cache_file (str): Path to cache file
            
        Returns:
            dict: Dictionary of artist -> genres
        """
        if not Path(cache_file).exists():
            return {}
        
        try:
            cache_df = pd.read_csv(cache_file)
            artist_genres = {}
            
            for _, row in cache_df.iterrows():
                artist = row['artist']
                genres = self.parse_genre_list(row['genres'])
                artist_genres[artist] = genres
            
            return artist_genres
        
        except Exception as e:
            print(f" Warning: Could not load cache: {e}")
            return {}
    
    def fetch_genres_for_dataset(self, input_csv, output_csv, rate_limit_delay=0.35):
        """
        Fetch genres for all unique artists in the dataset.
        
        Args:
            input_csv (str): Path to input CSV with artist information
            output_csv (str): Path to save output CSV with genre information
            rate_limit_delay (float): Delay between API calls (0.35s = ~170 req/min, safely under 180 limit)
        """
        print(f"\n Loading dataset: {input_csv}")
        df = pd.read_csv(input_csv)
        
        print(f" Loaded {len(df):,} listening events")
        
        # Get unique artists
        unique_artists = df['artistName'].unique()
        print(f" Found {len(unique_artists):,} unique artists")
        
        # Load cache
        cache_file = "Modelling/artist_genres_cache.csv"
        artist_genres = self.load_cache(cache_file)
        
        if artist_genres:
            print(f" Loaded {len(artist_genres)} cached artist genres")
            
            # Find artists we still need to fetch
            remaining_artists = [a for a in unique_artists if a not in artist_genres]
            print(f" Need to fetch {len(remaining_artists)} remaining artists")
        else:
            artist_genres = {}
            remaining_artists = list(unique_artists)
        
        # Track statistics
        stats = {
            'found': 0,
            'not_found': 0,
            'no_genres': 0,
            'error': 0
        }
        
        # Fetch genres for remaining artists
        if len(remaining_artists) > 0:
            print(f"\n Fetching genres from Spotify API...")
            
            for idx, artist in enumerate(remaining_artists, 1):
                if idx % 10 == 0 or idx == 1:
                    elapsed_pct = (idx/len(remaining_artists)*100)
                    print(f"  Progress: {idx}/{len(remaining_artists)} artists... ({elapsed_pct:.1f}%)")
                
                genres, status = self.get_artist_genres(artist)
                artist_genres[artist] = genres
                stats[status] = stats.get(status, 0) + 1
                
                # Save cache every 50 artists
                if idx % 50 == 0:
                    self.save_cache(artist_genres, cache_file)
                    print(f"   Cache checkpoint: {len(artist_genres)} artists saved")
                
                # Rate limiting (safely under Spotify's 180/min limit)
                time.sleep(rate_limit_delay)
            
            # Final cache save
            self.save_cache(artist_genres, cache_file)
            
            # Print fetch statistics
            print(f"\n Genre Fetch Statistics:")
            print(f"  Found with genres: {stats['found']}")
            print(f"  Found but no genres: {stats['no_genres']}")
            print(f"  Not found: {stats['not_found']}")
            print(f"  Errors: {stats['error']}")
        
        # Analyze genre distribution
        print("\n Analyzing genre distribution...")
        all_genres = []
        artists_with_genres = 0
        
        for artist, genres in artist_genres.items():
            if genres:
                artists_with_genres += 1
                all_genres.extend(genres)
        
        print(f"  Artists with genres: {artists_with_genres}/{len(artist_genres)} ({artists_with_genres/len(artist_genres)*100:.1f}%)")
        print(f"  Total genre tags: {len(all_genres)}")
        print(f"  Unique genres: {len(set(all_genres))}")
        
        # Show top genres
        genre_counts = Counter(all_genres)
        print(f"\n Top 20 Most Common Genres:")
        for genre, count in genre_counts.most_common(20):
            print(f"  {genre}: {count} artists")
        
        # Add genres to dataframe
        print("\n Mapping genres to listening events...")
        df['artist_genres'] = df['artistName'].map(artist_genres)
        
        # Create primary genre (most common genre for the artist)
        df['primary_genre'] = df['artist_genres'].apply(
            lambda x: x[0] if x and len(x) > 0 else None
        )
        
        # Count genres per row
        df['genre_count'] = df['artist_genres'].apply(lambda x: len(x) if x else 0)
        
        # Statistics
        events_with_genres = df['primary_genre'].notna().sum()
        print(f"  Events with genre labels: {events_with_genres:,}/{len(df):,} ({events_with_genres/len(df)*100:.1f}%)")
        
        # Save to CSV with proper serialization
        print(f"\n Saving to: {output_csv}")
        # Convert lists to JSON strings for CSV storage
        df['artist_genres'] = df['artist_genres'].apply(lambda x: json.dumps(x) if isinstance(x, list) else x)
        df.to_csv(output_csv, index=False)
        
        # Convert back for return value
        df['artist_genres'] = df['artist_genres'].apply(self.parse_genre_list)
        
        print(f" Saved {len(df):,} rows with genre information")
        
        return df
    
    def create_genre_summary(self, df_with_genres, output_file):
        """
        Create a summary report of genre distribution.
        
        Args:
            df_with_genres (pd.DataFrame): Dataset with genre information
            output_file (str): Path to save summary report
        """
        print("\n Creating genre summary report...")
        
        with open(output_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("GENRE DISTRIBUTION REPORT\n")
            f.write("="*60 + "\n\n")
            
            # Overall statistics
            f.write("1. OVERALL STATISTICS\n")
            f.write("-"*40 + "\n")
            f.write(f"Total listening events: {len(df_with_genres):,}\n")
            events_with_genres = df_with_genres['primary_genre'].notna().sum()
            f.write(f"Events with genre labels: {events_with_genres:,} ({events_with_genres/len(df_with_genres)*100:.1f}%)\n")
            f.write(f"Unique genres: {df_with_genres['primary_genre'].nunique()}\n\n")
            
            # Genre distribution
            f.write("2. GENRE DISTRIBUTION (by listening events)\n")
            f.write("-"*40 + "\n")
            genre_dist = df_with_genres['primary_genre'].value_counts()
            for idx, (genre, count) in enumerate(genre_dist.head(30).items(), 1):
                pct = (count / len(df_with_genres)) * 100
                f.write(f"{idx:2}. {genre:30} - {count:5} events ({pct:5.2f}%)\n")
            
            f.write("\n3. ARTIST GENRE COVERAGE\n")
            f.write("-"*40 + "\n")
            artists_with_genres = df_with_genres[df_with_genres['genre_count'] > 0]['artistName'].nunique()
            total_artists = df_with_genres['artistName'].nunique()
            f.write(f"Artists with genres: {artists_with_genres}/{total_artists} ({artists_with_genres/total_artists*100:.1f}%)\n")
            
            f.write("\n4. TOP ARTISTS BY GENRE\n")
            f.write("-"*40 + "\n")
            for genre in genre_dist.head(10).index:
                f.write(f"\n{genre.upper()}:\n")
                genre_df = df_with_genres[df_with_genres['primary_genre'] == genre]
                top_artists = genre_df['artistName'].value_counts().head(5)
                for artist, count in top_artists.items():
                    f.write(f"  - {artist}: {count} plays\n")
        
        print(f" Summary saved to: {output_file}")


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fetch genre labels from Spotify API')
    parser.add_argument('--refresh-cache', action='store_true', 
                       help='Delete existing cache and refetch all genres (fixes incorrect cached data)')
    args = parser.parse_args()
    
    # File paths
    input_csv = "Ingested_Data/merged_dataset.csv"
    output_csv = "Ingested_Data/dataset_with_genres.csv"
    summary_file = "Modelling/genre_summary.txt"
    cache_file = "Modelling/artist_genres_cache.csv"
    
    # Check if input exists
    if not Path(input_csv).exists():
        print(f" Error: {input_csv} not found!")
        sys.exit(1)
    
    # Handle cache refresh
    if args.refresh_cache and Path(cache_file).exists():
        print(f"\n --refresh-cache flag detected")
        print(f" Deleting old cache: {cache_file}")
        Path(cache_file).unlink()
        print(" Cache deleted. Will refetch all genres with improved matching.\n")
    
    # Fetch genres
    fetcher = GenreFetcher()
    df_with_genres = fetcher.fetch_genres_for_dataset(input_csv, output_csv)
    
    # Create summary
    fetcher.create_genre_summary(df_with_genres, summary_file)
    
    print("\n" + "="*60)
    print(" Genre fetching complete!")
    print("="*60)
    print(f"\n Outputs:")
    print(f"  - Dataset with genres: {output_csv}")
    print(f"  - Summary report: {summary_file}")
    print(f"  - Cache file: {cache_file}")
    print("\n Ready for genre classification modeling!")


if __name__ == "__main__":
    main()