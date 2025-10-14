"""
Audio Sample Downloader
Downloads audio preview clips using spotify-preview-finder npm package
"""

import pandas as pd
import requests
import time
import os
import sys
import json
import subprocess
from pathlib import Path

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import sp


class SpotifyAudioDownloader:
    """Downloads audio preview samples using spotify-preview-finder npm package"""
    
    def __init__(self, output_dir="Audio_Samples", rate_limit_delay=0.2):
        """
        Initialize the audio downloader.
        
        Args:
            output_dir (str): Directory to save audio samples
            rate_limit_delay (float): Delay between download requests
        """
        self.sp = sp
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.rate_limit_delay = rate_limit_delay
        self.download_log = []
        
        # Path to Node.js script
        self.node_script = Path(__file__).parent / "spotify_preview_finder.js"
        
        # Verify Node.js is available
        try:
            subprocess.run(['node', '--version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("Node.js is not installed. Please install Node.js to use this feature.")
        
    def sanitize_filename(self, filename):
        """
        Sanitize filename to be filesystem-safe.
        
        Args:
            filename (str): Original filename
            
        Returns:
            str: Safe filename
        """
        # Remove or replace invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        
        # Limit length
        if len(filename) > 200:
            filename = filename[:200]
        
        return filename
    
    def find_preview_urls(self, track_name, artist_name):
        """
        Find preview URLs using the spotify-preview-finder npm package.
        
        Args:
            track_name (str): Name of the track
            artist_name (str): Name of the artist
            
        Returns:
            dict: Track information with preview URLs or None if not found
        """
        try:
            # Call Node.js script
            result = subprocess.run(
                ['node', str(self.node_script), track_name, artist_name, '5'],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=Path(__file__).parent.parent  # Run from project root to find .env
            )
            
            if result.returncode != 0:
                print(f"Node.js error: {result.stderr}")
                return None
            
            # Parse JSON output 
            output_lines = result.stdout.strip().split('\n')
            json_line = None
            for line in output_lines:
                if line.startswith('{'):
                    json_line = line
                    break
            
            if not json_line:
                print(f"No JSON output found")
                return None
            
            data = json.loads(json_line)
            
            if data['success'] and data['tracks']:
                # Return the first track with preview URLs
                for track in data['tracks']:
                    if track['previewUrls'] and len(track['previewUrls']) > 0:
                        return {
                            'track_id': track['trackId'],
                            'name': track['name'],
                            'album': track['albumName'],
                            'release_date': track['releaseDate'],
                            'popularity': track['popularity'],
                            'duration_ms': track['durationMs'],
                            'spotify_url': track['spotifyUrl'],
                            'preview_urls': track['previewUrls']
                        }
                
                # If no tracks have preview URLs
                return None
            else:
                return None
                
        except subprocess.TimeoutExpired:
            print(f"Timeout searching for '{track_name}' by '{artist_name}'")
            return None
        except json.JSONDecodeError as e:
            print(f"Failed to parse response: {e}")
            print(f"JSON line: {json_line if 'json_line' in locals() else 'None'}")
            return None
        except Exception as e:
            print(f"Error finding preview: {e}")
            return None
    
    def download_preview(self, preview_url, output_path, retries=3):
        """
        Download audio preview from URL.
        
        Args:
            preview_url (str): URL of the preview audio
            output_path (Path): Path to save the audio file
            retries (int): Number of retry attempts
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not preview_url:
            return False
        
        for attempt in range(retries):
            try:
                response = requests.get(preview_url, timeout=30)
                response.raise_for_status()
                
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                
                time.sleep(self.rate_limit_delay)
                return True
                
            except Exception as e:
                if attempt < retries - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"Download error: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"Failed to download from {preview_url}")
                    return False
        
        return False
    
    def download_track_sample(self, track_name, artist_name):
        """
        Search for a track and download its preview using spotify-preview-finder.
        
        Args:
            track_name (str): Name of the track
            artist_name (str): Name of the artist
            
        Returns:
            dict: Download result information
        """
        result = {
            'track_name': track_name,
            'artist_name': artist_name,
            'success': False,
            'file_path': None,
            'track_id': None,
            'spotify_url': None,
            'reason': None
        }
        
        # Search for track with preview finder
        track_info = self.find_preview_urls(track_name, artist_name)
        
        if not track_info:
            result['reason'] = 'Track not found or no preview available'
            return result
        
        result['track_id'] = track_info['track_id']
        result['spotify_url'] = track_info['spotify_url']
        preview_urls = track_info.get('preview_urls', [])
        
        if not preview_urls:
            result['reason'] = 'No preview URLs found'
            return result
        
        # Create filename
        safe_artist = self.sanitize_filename(artist_name)
        safe_track = self.sanitize_filename(track_name)
        filename = f"{safe_artist} - {safe_track}.mp3"
        output_path = self.output_dir / filename
        
        # Skip if already downloaded
        if output_path.exists():
            result['success'] = True
            result['file_path'] = str(output_path)
            result['reason'] = 'Already downloaded'
            return result
        
        # Try downloading from each preview URL until one works
        for preview_url in preview_urls:
            success = self.download_preview(preview_url, output_path)
            if success:
                result['success'] = True
                result['file_path'] = str(output_path)
                result['reason'] = 'Downloaded successfully'
                return result
        
        result['reason'] = 'All download attempts failed'
        return result
    
    def download_from_csv(self, csv_path, max_tracks=None, skip_existing=True):
        """
        Download audio samples for tracks in a CSV file.
        
        Args:
            csv_path (str): Path to CSV with streaming history
            max_tracks (int): Maximum number of unique tracks to download (None = all)
            skip_existing (bool): Skip tracks that are already downloaded
            
        Returns:
            pd.DataFrame: Log of download results
        """
        print(f"Loading streaming history from: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Get unique track-artist combinations and reset index
        unique_tracks = df[['trackName', 'artistName']].drop_duplicates().reset_index(drop=True)
        print(f"Found {len(unique_tracks)} unique tracks")
        
        if max_tracks:
            unique_tracks = unique_tracks.head(max_tracks)
            print(f"Limiting to first {max_tracks} tracks")
        
        # Download samples
        results = []
        successful = 0
        no_preview = 0
        already_downloaded = 0
        failed = 0
        
        for counter, (idx, row) in enumerate(unique_tracks.iterrows(), start=1):
            track_name = row['trackName']
            artist_name = row['artistName']
            
            print(f"\n[{counter}/{len(unique_tracks)}] {artist_name} - {track_name}")
            
            result = self.download_track_sample(track_name, artist_name)
            results.append(result)
            
            if result['success']:
                if result['reason'] == 'Already downloaded':
                    already_downloaded += 1
                    print(f" Already exists")
                else:
                    successful += 1
                    print(f"  Downloaded: {result['file_path']}")
            elif result['reason'] == 'No preview available':
                no_preview += 1
                print(f"  No preview available")
            else:
                failed += 1
                print(f" {result['reason']}")
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Save download log
        log_path = self.output_dir / "download_log.csv"
        results_df.to_csv(log_path, index=False)
        
        # Print summary
        print("\n" + "="*60)
        print("Download Summary")
        print("="*60)
        print(f"Total unique tracks: {len(unique_tracks)}")
        print(f"Successfully downloaded: {successful}")
        print(f" Already downloaded: {already_downloaded}")
        print(f"  No preview available: {no_preview}")
        print(f" Failed: {failed}")
        print(f"\n Audio samples saved to: {self.output_dir.absolute()}")
        print(f" Download log saved to: {log_path.absolute()}")
        
        return results_df


def main():
    """Main function to run the audio sample downloader"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download Spotify preview samples')
    parser.add_argument('--csv', type=str, 
                       default='Ingested_Data/cleaned_streaming_history.csv',
                       help='Path to streaming history CSV')
    parser.add_argument('--output', type=str, 
                       default='Audio_Samples',
                       help='Output directory for audio samples')
    parser.add_argument('--max-tracks', type=int, 
                       default=None,
                       help='Maximum number of tracks to download (for testing)')
    
    args = parser.parse_args()
    
    # Check if CSV exists
    if not os.path.exists(args.csv):
        print(f"Error: CSV file not found: {args.csv}")
        print("Please run 'python Ingestion/clean.py' first to create the cleaned data.")
        sys.exit(1)
    
    # Create downloader
    downloader = SpotifyAudioDownloader(output_dir=args.output)
    
    # Download samples
    results = downloader.download_from_csv(args.csv, max_tracks=args.max_tracks)
    
    print("\n Download process complete!")


if __name__ == "__main__":
    main()
