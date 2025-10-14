"""
Fetch Genre Labels from Spotify API
Retrieves artist genre information to create labels for classification
"""

import pandas as pd
import sys
from pathlib import Path
import time
from collections import Counter

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import sp


class GenreFetcher:
    """Fetch and process genre information from Spotify API"""
    
    def __init__(self):
        """Initialize Spotify client"""
        print("="*60)
        print("🎵 Genre Fetcher - Spotify API")
        print("="*60)
        
        self.sp = sp
        print(" Spotify client initialized")
    
    def get_artist_genres(self, artist_name):
        """
        Get genres for an artist from Spotify API.
        
        Args:
            artist_name (str): Name of the artist
            
        Returns:
            list: List of genre strings, empty list if not found
        """
        try:
            # Search for artist
            results = self.sp.search(q=f'artist:{artist_name}', type='artist', limit=1)
            
            if results['artists']['items']:
                artist = results['artists']['items'][0]
                return artist.get('genres', [])
            return []
        
        except Exception as e:
            print(f" Error fetching genres for {artist_name}: {e}")
            return []
    
    def fetch_genres_for_dataset(self, input_csv, output_csv, rate_limit_delay=0.05):
        """
        Fetch genres for all unique artists in the dataset.
        
        Args:
            input_csv (str): Path to input CSV with artist information
            output_csv (str): Path to save output CSV with genre information
            rate_limit_delay (float): Delay between API calls (seconds)
        """
        print(f"\n Loading dataset: {input_csv}")
        df = pd.read_csv(input_csv)
        
        print(f" Loaded {len(df):,} listening events")
        
        # Get unique artists
        unique_artists = df['artistName'].unique()
        print(f" Found {len(unique_artists):,} unique artists")
        
        # Check if we can resume from a cached file
        cache_file = "Modelling/artist_genres_cache.csv"
        if Path(cache_file).exists():
            print(f"\n Loading cached genre data from {cache_file}")
            cache_df = pd.read_csv(cache_file)
            artist_genres = dict(zip(cache_df['artist'], cache_df['genres'].apply(eval)))
            print(f"Loaded {len(artist_genres)} cached artist genres")
            
            # Find artists we still need to fetch
            remaining_artists = [a for a in unique_artists if a not in artist_genres]
            print(f" Need to fetch {len(remaining_artists)} remaining artists")
        else:
            artist_genres = {}
            remaining_artists = list(unique_artists)
        
        # Fetch genres for remaining artists
        if len(remaining_artists) > 0:
            print(f"\n Fetching genres from Spotify API...")
            
            for idx, artist in enumerate(remaining_artists, 1):
                if idx % 10 == 0 or idx == 1:
                    print(f"  Progress: {idx}/{len(remaining_artists)} artists... ({(idx/len(remaining_artists)*100):.1f}%)")
                
                genres = self.get_artist_genres(artist)
                artist_genres[artist] = genres
                
                # Save cache every 50 artists
                if idx % 50 == 0:
                    cache_df = pd.DataFrame([
                        {'artist': k, 'genres': v} for k, v in artist_genres.items()
                    ])
                    cache_df.to_csv(cache_file, index=False)
                    print(f"   Saved cache checkpoint ({len(artist_genres)} artists)")
                
                # Rate limiting
                time.sleep(rate_limit_delay)
            
            # Final cache save
            cache_df = pd.DataFrame([
                {'artist': k, 'genres': v} for k, v in artist_genres.items()
            ])
            cache_df.to_csv(cache_file, index=False)
            print(f" Fetched and cached genres for {len(artist_genres)} artists")
        
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
        
        # Save to CSV
        print(f"\n Saving to: {output_csv}")
        df.to_csv(output_csv, index=False)
        print(f" Saved {len(df):,} rows with genre information")
        
        return df
    
    def create_genre_summary(self, df_with_genres, output_file):
        """
        Create a summary report of genre distribution.
        
        Args:
            df_with_genres (pd.DataFrame): Dataset with genre information
            output_file (str): Path to save summary report
        """
        print("\n📝 Creating genre summary report...")
        
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
    # File paths
    input_csv = "Ingested_Data/merged_dataset.csv"
    output_csv = "Ingested_Data/dataset_with_genres.csv"
    summary_file = "Modelling/genre_summary.txt"
    
    # Check if input exists
    if not Path(input_csv).exists():
        print(f" Error: {input_csv} not found!")
        sys.exit(1)
    
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
    print("\n Ready for genre classification modeling!")


if __name__ == "__main__":
    main()
