"""
Audio Feature Extraction using Signal Processing and ML
Extracts audio features from audio files as an alternative to deprecated Spotify API
"""

import librosa
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class AudioFeatureExtractor:
    """Extract audio features from audio files using signal processing"""
    
    def __init__(self, sample_rate=22050):
        """
        Initialize the audio feature extractor.
        
        Args:
            sample_rate (int): Sample rate for audio processing
        """
        self.sample_rate = sample_rate
    
    def load_audio(self, audio_path, duration=30):
        """
        Load audio file.
        
        Args:
            audio_path (str): Path to audio file
            duration (int): Duration to load in seconds (30s matches Spotify previews)
            
        Returns:
            tuple: (audio_data, sample_rate)
        """
        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate, duration=duration)
            return y, sr
        except Exception as e:
            print(f"Error loading audio: {e}")
            return None, None
    
    def extract_tempo(self, y, sr):
        """
        Extract tempo (BPM) from audio.
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            float: Tempo in BPM
        """
        try:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            return float(tempo)
        except:
            return None
    
    def extract_spectral_features(self, y, sr):
        """
        Extract spectral features from audio.
        
        Returns:
            dict: Dictionary of spectral features
        """
        features = {}
        
        try:
            # Spectral centroid (brightness)
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
            features['spectral_centroid_std'] = float(np.std(spectral_centroids))
            
            # Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
            
            # Spectral bandwidth
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            features['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features['zero_crossing_rate_mean'] = float(np.mean(zcr))
            
        except Exception as e:
            print(f"Error extracting spectral features: {e}")
            
        return features
    
    def extract_mfcc(self, y, sr, n_mfcc=13):
        """
        Extract MFCC (Mel-frequency cepstral coefficients).
        
        Returns:
            dict: MFCC statistics
        """
        features = {}
        
        try:
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
            
            # Store mean and std for each MFCC coefficient
            for i in range(n_mfcc):
                features[f'mfcc_{i}_mean'] = float(np.mean(mfccs[i]))
                features[f'mfcc_{i}_std'] = float(np.std(mfccs[i]))
                
        except Exception as e:
            print(f"Error extracting MFCC: {e}")
            
        return features
    
    def extract_chroma(self, y, sr):
        """
        Extract chroma features (pitch class).
        
        Returns:
            dict: Chroma features
        """
        features = {}
        
        try:
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features['chroma_mean'] = float(np.mean(chroma))
            features['chroma_std'] = float(np.std(chroma))
            
            # Dominant pitch class
            features['dominant_pitch_class'] = int(np.argmax(np.mean(chroma, axis=1)))
            
        except Exception as e:
            print(f"Error extracting chroma: {e}")
            
        return features
    
    def extract_energy_features(self, y):
        """
        Extract energy-based features.
        
        Returns:
            dict: Energy features
        """
        features = {}
        
        try:
            # RMS energy
            rms = librosa.feature.rms(y=y)[0]
            features['rms_mean'] = float(np.mean(rms))
            features['rms_std'] = float(np.std(rms))
            
            # Overall loudness (in dB)
            features['loudness'] = float(20 * np.log10(np.mean(rms) + 1e-10))
            
        except Exception as e:
            print(f"Error extracting energy features: {e}")
            
        return features
    
    def estimate_danceability(self, y, sr, tempo):
        """
        Estimate danceability (0-1 scale) based on rhythm and beat strength.
        
        Using a heuristic combining tempo and onset strength.

        Couldn't discover a specific formula from  to calculate this.



        """
        try:
            # Beat strength
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            beat_strength = float(np.mean(onset_env))
            
            # Tempo factor (120 BPM is ideal for dancing)
            tempo_factor = 1 - abs(tempo - 120) / 120
            tempo_factor = max(0, min(1, tempo_factor))
            
            # Rhythm regularity (autocorrelation of onset envelope)
            ac = librosa.autocorrelate(onset_env)
            rhythm_regularity = float(np.max(ac[1:100]) / (ac[0] + 1e-10))
            
            # Combined danceability score
            danceability = (beat_strength * 0.4 + tempo_factor * 0.3 + rhythm_regularity * 0.3)
            danceability = max(0, min(1, danceability))
            
            return float(danceability)
            
        except Exception as e:
            print(f"Error estimating danceability: {e}")
            return None
    
    def estimate_energy(self, y, sr):
        """
        Estimate energy (0-1 scale) based on loudness, tempo, and dynamics.
        """
        try:
            # RMS energy
            rms = np.mean(librosa.feature.rms(y=y))
            
            # Spectral centroid (brightness correlates with energy)
            centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            centroid_normalized = centroid / (sr / 2)  # Normalize by Nyquist frequency
            
            # Dynamic range
            dynamic_range = np.std(librosa.feature.rms(y=y))
            
            # Combined energy score
            energy = (rms * 0.5 + centroid_normalized * 0.3 + dynamic_range * 0.2)
            energy = max(0, min(1, energy))
            
            return float(energy)
            
        except Exception as e:
            print(f"Error estimating energy: {e}")
            return None
    
    def estimate_valence(self, y, sr):
        """
        Estimate valence/mood (0-1 scale, where 1 is positive/happy).
        Based on mode (major/minor) and brightness.
        """
        try:
            # Chroma features for mode detection
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            
            # Major chords (C, E, G positions: 0, 4, 7) vs Minor (C, Eb, G: 0, 3, 7)
            chroma_mean = np.mean(chroma, axis=1)
            major_score = chroma_mean[4]  # E (major third)
            minor_score = chroma_mean[3]  # Eb (minor third)
            
            mode_valence = major_score / (major_score + minor_score + 1e-10)
            
            # Brightness (spectral centroid correlates with happiness)
            centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            brightness = centroid / (sr / 2)
            
            # Combined valence
            valence = (mode_valence * 0.6 + brightness * 0.4)
            valence = max(0, min(1, valence))
            
            return float(valence)
            
        except Exception as e:
            print(f"Error estimating valence: {e}")
            return None
    
    def estimate_acousticness(self, y, sr):
        """
        Estimate acousticness (0-1 scale).
        Based on harmonic-percussive separation and spectral features.
        """
        try:
            # Separate harmonic and percussive components
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            
            # Ratio of harmonic to total energy
            harmonic_energy = np.sum(y_harmonic ** 2)
            total_energy = np.sum(y ** 2)
            harmonic_ratio = harmonic_energy / (total_energy + 1e-10)
            
            # Lower spectral centroid suggests acoustic instruments
            centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            centroid_factor = 1 - (centroid / (sr / 2))
            
            acousticness = (harmonic_ratio * 0.6 + centroid_factor * 0.4)
            acousticness = max(0, min(1, acousticness))
            
            return float(acousticness)
            
        except Exception as e:
            print(f"Error estimating acousticness: {e}")
            return None
    
    def estimate_instrumentalness(self, y, sr):
        """
        Estimate instrumentalness (0-1 scale).
        Based on detecting vocal-like frequencies.
        """
        try:
            # Vocal frequency range is typically 80-300 Hz (male) and 165-255 Hz (female)
            
            
            # Spectral flux (measure of spectral change - vocals have more variation)
            spectral_flux = np.mean(np.abs(np.diff(librosa.feature.spectral_centroid(y=y, sr=sr))))
            
            # MFCC variance (vocals have characteristic MFCC patterns)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_variance = np.mean(np.var(mfcc, axis=1))
            
            # Lower flux and variance suggests instrumental
            # This is a rough heuristic 
            # Hopefully, I have a better way to do this in next iteration
            vocal_likelihood = (spectral_flux + mfcc_variance) / 2
            instrumentalness = 1 - min(1, vocal_likelihood / 1000)  # Normalize
            
            return float(instrumentalness)
            
        except Exception as e:
            print(f"Error estimating instrumentalness: {e}")
            return None
    
    def extract_all_features(self, audio_path):
        """
        Extract all audio features from an audio file.
        
        Args:
            audio_path (str): Path to audio file
            
        Returns:
            dict: Dictionary containing all extracted features
        """
        # Load audio
        y, sr = self.load_audio(audio_path)
        
        if y is None:
            return None
        
        features = {
            'audio_path': str(audio_path),
            'duration_ms': int(len(y) / sr * 1000)
        }
        
        # Extract tempo
        tempo = self.extract_tempo(y, sr)
        if tempo:
            features['tempo'] = tempo
        
        # Extract spectral features
        features.update(self.extract_spectral_features(y, sr))
        
        # Extract MFCC
        features.update(self.extract_mfcc(y, sr))
        
        # Extract chroma
        features.update(self.extract_chroma(y, sr))
        
        # Extract energy features
        features.update(self.extract_energy_features(y))
        
        # Estimate Spotify-like features
        if tempo:
            danceability = self.estimate_danceability(y, sr, tempo)
            if danceability is not None:
                features['danceability'] = danceability
        
        energy = self.estimate_energy(y, sr)
        if energy is not None:
            features['energy'] = energy
        
        valence = self.estimate_valence(y, sr)
        if valence is not None:
            features['valence'] = valence
        
        acousticness = self.estimate_acousticness(y, sr)
        if acousticness is not None:
            features['acousticness'] = acousticness
        
        instrumentalness = self.estimate_instrumentalness(y, sr)
        if instrumentalness is not None:
            features['instrumentalness'] = instrumentalness
        
        return features


def extract_features_from_directory(audio_dir, output_csv):
    """
    Extract features from all audio files in a directory.
    
    Args:
        audio_dir (str): Path to directory containing audio files
        output_csv (str): Path to save extracted features CSV
    """
    audio_dir = Path(audio_dir)
    
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
    
    # Extract features from each file
    features_list = []
    
    for idx, audio_file in enumerate(audio_files):
        print(f"Processing {idx + 1}/{len(audio_files)}: {audio_file.name}")
        
        features = extractor.extract_all_features(audio_file)
        
        if features:
            features_list.append(features)
    
    # Create DataFrame
    df = pd.DataFrame(features_list)
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"\nFeatures extracted and saved to: {output_csv}")
    print(f"Total tracks processed: {len(df)}")
    print(f"Total features per track: {len(df.columns)}")
    
    # Print summary statistics
    print("\n=== Feature Statistics ===")
    if 'tempo' in df.columns:
        print(f"Average Tempo: {df['tempo'].mean():.2f} BPM")
    if 'danceability' in df.columns:
        print(f"Average Danceability: {df['danceability'].mean():.3f}")
    if 'energy' in df.columns:
        print(f"Average Energy: {df['energy'].mean():.3f}")
    if 'valence' in df.columns:
        print(f"Average Valence: {df['valence'].mean():.3f}")
    
    return df


if __name__ == "__main__":
    import sys
    
    # Example usage
    if len(sys.argv) > 1:
        audio_directory = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else "extracted_audio_features.csv"
    else:
        
        audio_directory = "Audio_Samples"
        output_file = "Ingested_Data/extracted_audio_features.csv"
    
    extract_features_from_directory(audio_directory, output_file)
