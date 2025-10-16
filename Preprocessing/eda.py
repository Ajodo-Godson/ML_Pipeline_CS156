"""
Exploratory Data Analysis (EDA) - Spotify Listening History
Comprehensive analysis and visualization of merged dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class SpotifyEDA:
    """Comprehensive EDA for Spotify listening history with audio features"""
    
    def __init__(self, merged_csv):
        """
        Initialize EDA with merged dataset.
        
        Args:
            merged_csv (str): Path to merged dataset CSV
        """
        print("="*60)
        print(" Spotify Listening History - EDA")
        print("="*60)
        
        print(f"\n Loading dataset: {merged_csv}")
        self.df = pd.read_csv(merged_csv)
        
        # Convert date columns
        if 'endTime' in self.df.columns:
            self.df['endTime'] = pd.to_datetime(self.df['endTime'])
        if 'date' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['date'])
        
        # Separate data with and without features
        self.df_with_features = self.df[self.df['audio_path'].notna()].copy()
        self.df_without_features = self.df[self.df['audio_path'].isna()].copy()
        
        print(f" Loaded {len(self.df):,} listening events")
        print(f"   - With audio features: {len(self.df_with_features):,}")
        print(f"   - Without audio features: {len(self.df_without_features):,}")
        
        # Create output directory for plots
        self.output_dir = Path("Preprocessing/EDA_Outputs")
        self.output_dir.mkdir(exist_ok=True)
        print(f"\n📁 Plots will be saved to: {self.output_dir}")
    
    def basic_statistics(self):
        """Display basic statistics about the dataset"""
        print("\n" + "="*60)
        print("📈 Basic Statistics")
        print("="*60)
        
        print(f"\n🎵 Music Listening Overview:")
        print(f"  Total listening events: {len(self.df):,}")
        print(f"  Unique tracks: {self.df['trackName'].nunique():,}")
        print(f"  Unique artists: {self.df['artistName'].nunique():,}")
        
        if 'date' in self.df.columns:
            print(f"  Date range: {self.df['date'].min().date()} to {self.df['date'].max().date()}")
            days = (self.df['date'].max() - self.df['date'].min()).days + 1
            print(f"  Total days: {days}")
            print(f"  Avg listens per day: {len(self.df) / days:.1f}")
        
        if 'minutesPlayed' in self.df.columns:
            total_hours = self.df['minutesPlayed'].sum() / 60
            print(f"\n⏱️  Listening Time:")
            print(f"  Total hours: {total_hours:,.1f} hours")
            print(f"  Total minutes: {self.df['minutesPlayed'].sum():,.1f} minutes")
            print(f"  Avg minutes per listen: {self.df['minutesPlayed'].mean():.2f} minutes")
            print(f"  Median minutes per listen: {self.df['minutesPlayed'].median():.2f} minutes")
        
        # Audio features statistics (for matched tracks only)
        if len(self.df_with_features) > 0:
            print(f"\n🎼 Audio Features Summary:")
            feature_cols = ['tempo', 'danceability', 'energy', 'valence', 'acousticness', 'instrumentalness']
            
            for col in feature_cols:
                if col in self.df_with_features.columns:
                    mean_val = self.df_with_features[col].mean()
                    median_val = self.df_with_features[col].median()
                    print(f"  {col.capitalize():20} - Mean: {mean_val:.3f}, Median: {median_val:.3f}")
    
    def top_tracks_and_artists(self, top_n=10):
        """Analyze top tracks and artists"""
        print("\n" + "="*60)
        print(f"🏆 Top {top_n} Tracks and Artists")
        print("="*60)
        
        # Top tracks
        print(f"\n🎵 Top {top_n} Most Played Tracks:")
        top_tracks = self.df.groupby(['trackName', 'artistName']).size().sort_values(ascending=False).head(top_n)
        for idx, ((track, artist), count) in enumerate(top_tracks.items(), 1):
            print(f"  {idx:2}. {artist} - {track}: {count} plays")
        
        # Top artists
        print(f"\n🎤 Top {top_n} Most Played Artists:")
        top_artists = self.df.groupby('artistName').size().sort_values(ascending=False).head(top_n)
        for idx, (artist, count) in enumerate(top_artists.items(), 1):
            unique_tracks = self.df[self.df['artistName'] == artist]['trackName'].nunique()
            print(f"  {idx:2}. {artist}: {count} plays ({unique_tracks} unique tracks)")
        
        # Visualize top tracks
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Top tracks bar plot
        top_tracks_plot = top_tracks.head(10)
        track_labels = [f"{artist[:20]} - {track[:20]}" for (track, artist) in top_tracks_plot.index]
        ax1.barh(track_labels, top_tracks_plot.values, color='#1DB954')
        ax1.set_xlabel('Number of Plays')
        ax1.set_title(f'Top {top_n} Most Played Tracks')
        ax1.invert_yaxis()
        
        # Top artists bar plot
        top_artists_plot = top_artists.head(10)
        ax2.barh(list(top_artists_plot.index), top_artists_plot.values, color='#1ed760')
        ax2.set_xlabel('Number of Plays')
        ax2.set_title(f'Top {top_n} Most Played Artists')
        ax2.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'top_tracks_artists.png', dpi=300, bbox_inches='tight')
        print(f"\n💾 Saved plot: {self.output_dir / 'top_tracks_artists.png'}")
        plt.close()
    
    def temporal_analysis(self):
        """Analyze listening patterns over time"""
        print("\n" + "="*60)
        print("⏰ Temporal Analysis")
        print("="*60)
        
        if 'hour' not in self.df.columns:
            print("⚠️  Hour information not available")
            return
        
        # Listening by hour
        print(f"\n🕐 Listening by Hour of Day:")
        hourly_plays = self.df.groupby('hour').size()
        peak_hour = hourly_plays.idxmax()
        print(f"  Peak listening hour: {peak_hour}:00 ({hourly_plays[peak_hour]} plays)")
        
        # Listening by day of week
        if 'day_of_week' in self.df.columns:
            print(f"\n📅 Listening by Day of Week:")
            day_plays = self.df.groupby('day_of_week').size().sort_values(ascending=False)
            print(f"  Most active day: {day_plays.index[0]} ({day_plays.iloc[0]} plays)")
            print(f"  Least active day: {day_plays.index[-1]} ({day_plays.iloc[-1]} plays)")
        
        # Create temporal visualizations
        fig = plt.figure(figsize=(16, 10))
        
        # Hour of day
        ax1 = plt.subplot(2, 2, 1)
        hourly_plays.plot(kind='bar', color='#1DB954', ax=ax1)
        ax1.set_xlabel('Hour of Day')
        ax1.set_ylabel('Number of Plays')
        ax1.set_title('Listening Activity by Hour of Day')
        ax1.grid(axis='y', alpha=0.3)
        
        # Day of week
        if 'day_of_week' in self.df.columns:
            ax2 = plt.subplot(2, 2, 2)
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_plays_ordered = self.df.groupby('day_of_week').size().reindex(day_order)
            day_plays_ordered.plot(kind='bar', color='#1ed760', ax=ax2)
            ax2.set_xlabel('Day of Week')
            ax2.set_ylabel('Number of Plays')
            ax2.set_title('Listening Activity by Day of Week')
            ax2.set_xticklabels(day_order, rotation=45)
            ax2.grid(axis='y', alpha=0.3)
        
        # Month
        if 'month' in self.df.columns:
            ax3 = plt.subplot(2, 2, 3)
            monthly_plays = self.df.groupby('month').size()
            monthly_plays.plot(kind='bar', color='#1DB954', ax=ax3)
            ax3.set_xlabel('Month')
            ax3.set_ylabel('Number of Plays')
            ax3.set_title('Listening Activity by Month')
            ax3.grid(axis='y', alpha=0.3)
        
        # Daily trend (if date available)
        if 'date' in self.df.columns:
            ax4 = plt.subplot(2, 2, 4)
            daily_plays = self.df.groupby('date').size()
            daily_plays.plot(color='#1ed760', ax=ax4)
            ax4.set_xlabel('Date')
            ax4.set_ylabel('Number of Plays')
            ax4.set_title('Daily Listening Trend')
            ax4.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'temporal_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n💾 Saved plot: {self.output_dir / 'temporal_analysis.png'}")
        plt.close()
    
    def audio_features_analysis(self):
        """Analyze audio features distribution"""
        print("\n" + "="*60)
        print("🎼 Audio Features Analysis")
        print("="*60)
        
        if len(self.df_with_features) == 0:
            print("⚠️  No audio features available for analysis")
            return
        
        print(f"\n📊 Analyzing {len(self.df_with_features):,} listening events with audio features")
        
        feature_cols = ['tempo', 'danceability', 'energy', 'valence', 'acousticness', 'instrumentalness']
        available_features = [col for col in feature_cols if col in self.df_with_features.columns]
        
        if len(available_features) == 0:
            print("⚠️  No standard audio features found")
            return
        
        # Create feature distribution plots
        n_features = len(available_features)
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()
        
        for idx, feature in enumerate(available_features):
            if feature in self.df_with_features.columns:
                ax = axes[idx]
                self.df_with_features[feature].hist(bins=30, color='#1DB954', alpha=0.7, ax=ax)
                ax.set_xlabel(feature.capitalize())
                ax.set_ylabel('Frequency')
                ax.set_title(f'{feature.capitalize()} Distribution')
                ax.axvline(self.df_with_features[feature].mean(), color='red', linestyle='--', 
                          label=f'Mean: {self.df_with_features[feature].mean():.2f}')
                ax.legend()
                ax.grid(alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_features, 6):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'audio_features_distribution.png', dpi=300, bbox_inches='tight')
        print(f"\n💾 Saved plot: {self.output_dir / 'audio_features_distribution.png'}")
        plt.close()
        
        # Correlation heatmap
        if len(available_features) > 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            corr_matrix = self.df_with_features[available_features].corr()
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                       square=True, ax=ax, cbar_kws={'shrink': 0.8})
            ax.set_title('Audio Features Correlation Matrix')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'feature_correlation.png', dpi=300, bbox_inches='tight')
            print(f"💾 Saved plot: {self.output_dir / 'feature_correlation.png'}")
            plt.close()
    
    def listening_duration_analysis(self):
        """Analyze listening duration patterns"""
        print("\n" + "="*60)
        print("⏱️  Listening Duration Analysis")
        print("="*60)
        
        if 'minutesPlayed' not in self.df.columns:
            print("⚠️  Duration information not available")
            return
        
        # Duration statistics
        print(f"\n📊 Duration Statistics:")
        print(f"  Mean: {self.df['minutesPlayed'].mean():.2f} minutes")
        print(f"  Median: {self.df['minutesPlayed'].median():.2f} minutes")
        print(f"  Min: {self.df['minutesPlayed'].min():.2f} minutes")
        print(f"  Max: {self.df['minutesPlayed'].max():.2f} minutes")
        
        # Skip detection (songs listened < 30 seconds)
        if 'secondsPlayed' in self.df.columns:
            skips = (self.df['secondsPlayed'] < 30).sum()
            skip_rate = (skips / len(self.df)) * 100
            print(f"\n⏭️  Skip Analysis (< 30 seconds):")
            print(f"  Skipped tracks: {skips:,} ({skip_rate:.1f}%)")
            print(f"  Completed tracks: {len(self.df) - skips:,} ({100-skip_rate:.1f}%)")
        
        # Visualize duration distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Duration histogram
        self.df['minutesPlayed'].hist(bins=50, color='#1DB954', alpha=0.7, ax=ax1)
        ax1.set_xlabel('Minutes Played')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Listening Duration Distribution')
        mean_minutes = self.df['minutesPlayed'].mean()
        ax1.axvline(mean_minutes, color='red', linestyle='--', 
                   label=f'Mean: {mean_minutes:.2f} min')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Box plot by hour
        if 'hour' in self.df.columns:
            hour_duration = self.df.groupby('hour')['minutesPlayed'].apply(list)
            ax2.boxplot([hour_duration[h] for h in sorted(hour_duration.index)], 
                       labels=sorted(hour_duration.index))
            ax2.set_xlabel('Hour of Day')
            ax2.set_ylabel('Minutes Played')
            ax2.set_title('Listening Duration by Hour of Day')
            ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'duration_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n💾 Saved plot: {self.output_dir / 'duration_analysis.png'}")
        plt.close()
    
    def generate_summary_report(self):
        """Generate a comprehensive text summary report"""
        print("\n" + "="*60)
        print("📝 Generating Summary Report")
        print("="*60)
        
        report_path = self.output_dir / 'EDA_Summary_Report.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("SPOTIFY LISTENING HISTORY - EDA SUMMARY REPORT\n")
            f.write("="*60 + "\n\n")
            
            # Basic stats
            f.write("1. DATASET OVERVIEW\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total listening events: {len(self.df):,}\n")
            f.write(f"Events with audio features: {len(self.df_with_features):,}\n")
            f.write(f"Events without audio features: {len(self.df_without_features):,}\n")
            f.write(f"Unique tracks: {self.df['trackName'].nunique():,}\n")
            f.write(f"Unique artists: {self.df['artistName'].nunique():,}\n\n")
            
            # Temporal info
            if 'date' in self.df.columns:
                f.write("2. TEMPORAL INFORMATION\n")
                f.write("-" * 40 + "\n")
                f.write(f"Date range: {self.df['date'].min().date()} to {self.df['date'].max().date()}\n")
                days = (self.df['date'].max() - self.df['date'].min()).days + 1
                f.write(f"Total days: {days}\n")
                f.write(f"Average listens per day: {len(self.df) / days:.1f}\n\n")
            
            # Listening time
            if 'minutesPlayed' in self.df.columns:
                f.write("3. LISTENING TIME\n")
                f.write("-" * 40 + "\n")
                total_hours = self.df['minutesPlayed'].sum() / 60
                f.write(f"Total hours: {total_hours:,.1f}\n")
                f.write(f"Average minutes per listen: {self.df['minutesPlayed'].mean():.2f}\n")
                f.write(f"Median minutes per listen: {self.df['minutesPlayed'].median():.2f}\n\n")
            
            # Top tracks
            f.write("4. TOP 10 MOST PLAYED TRACKS\n")
            f.write("-" * 40 + "\n")
            top_tracks = self.df.groupby(['trackName', 'artistName']).size().sort_values(ascending=False).head(10)
            for idx, ((track, artist), count) in enumerate(top_tracks.items(), 1):
                f.write(f"{idx:2}. {artist} - {track}: {count} plays\n")
            f.write("\n")
            
            # Top artists
            f.write("5. TOP 10 MOST PLAYED ARTISTS\n")
            f.write("-" * 40 + "\n")
            top_artists = self.df.groupby('artistName').size().sort_values(ascending=False).head(10)
            for idx, (artist, count) in enumerate(top_artists.items(), 1):
                f.write(f"{idx:2}. {artist}: {count} plays\n")
            f.write("\n")
            
            # Audio features
            if len(self.df_with_features) > 0:
                f.write("6. AUDIO FEATURES SUMMARY\n")
                f.write("-" * 40 + "\n")
                feature_cols = ['tempo', 'danceability', 'energy', 'valence', 'acousticness', 'instrumentalness']
                for col in feature_cols:
                    if col in self.df_with_features.columns:
                        mean_val = self.df_with_features[col].mean()
                        median_val = self.df_with_features[col].median()
                        f.write(f"{col.capitalize():20} - Mean: {mean_val:.3f}, Median: {median_val:.3f}\n")
        
        print(f"✅ Summary report saved to: {report_path}")
    
    def run_full_eda(self):
        """Run complete EDA analysis"""
        print("\n🚀 Running Full EDA Analysis...\n")
        
        self.basic_statistics()
        self.top_tracks_and_artists()
        self.temporal_analysis()
        self.audio_features_analysis()
        self.listening_duration_analysis()
        self.generate_summary_report()
        
        print("\n" + "="*60)
        print("✅ EDA Complete!")
        print("="*60)
        print(f"\n📁 All outputs saved to: {self.output_dir}")
        print(f"\nGenerated files:")
        for file in sorted(self.output_dir.glob('*')):
            print(f"  - {file.name}")


if __name__ == "__main__":
    # File path
    merged_csv = "Ingested_Data/merged_dataset.csv"
    
    # Check if file exists
    if not Path(merged_csv).exists():
        print(f"❌ Error: {merged_csv} not found!")
        print(f"Please run 'python Preprocessing/merge_data.py' first")
        sys.exit(1)
    
    # Run EDA
    eda = SpotifyEDA(merged_csv)
    eda.run_full_eda()
    
    print("\n🎉 EDA complete! Check the EDA_Outputs folder for visualizations.")
