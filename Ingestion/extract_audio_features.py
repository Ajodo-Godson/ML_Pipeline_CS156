"""
Audio Feature Extraction using Signal Processing and ML
Extracts audio features from audio files as an alternative to deprecated Spotify API
IMPROVED VERSION with better danceability, energy, and instrumentalness estimation
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
        IMPROVED: Estimate danceability (0-1 scale) based on beat regularity and strength.
        
        Uses beat interval consistency as primary metric, combined with tempo optimality.
        Higher regularity = more danceable.
        """
        try:
            # Get onset strength envelope
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            
            # Extract beats with tight tracking for better regularity measurement
            _, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, tightness=100)
            
            # Need sufficient beats for reliable analysis
            if len(beats) < 10:
                return 0.3
            
            # Calculate beat interval regularity (inverse of coefficient of variation)
            intervals = np.diff(beats)
            if len(intervals) == 0 or np.mean(intervals) == 0:
                return 0.3
            
            # Lower CV = more regular rhythm = more danceable
            cv = np.std(intervals) / np.mean(intervals)
            regularity = 1.0 / (1.0 + cv)
            
            # Tempo score: Gaussian distribution peaked at 120 BPM
            tempo_score = np.exp(-((tempo - 120)**2) / (2 * 25**2))
            
            # Beat strength (use 80th percentile to avoid outliers)
            strength = np.percentile(onset_env, 80) / 100.0
            
            # Weighted combination: regularity is most important for dancing
            danceability = 0.5 * regularity + 0.3 * tempo_score + 0.2 * strength
            
            # Clamp to reasonable range (avoid extremes)
            return float(max(0.1, min(0.95, danceability)))
            
        except Exception as e:
            print(f"Error estimating danceability: {e}")
            return 0.4
    
    def estimate_energy(self, y, sr):
        """
        IMPROVED: Estimate energy (0-1 scale) using RMS and frequency band analysis.
        
        Combines overall loudness with bass and treble energy for better accuracy.
        """
        try:
            # RMS energy (overall loudness)
            rms = np.sqrt(np.mean(y**2))
            
            # Frequency domain analysis
            S = np.abs(librosa.stft(y))
            
            # Bass energy (low frequencies contribute heavily to perceived energy)
            # For sr=22050, n_fft=2048: bin 50 ≈ 537 Hz
            low_energy = np.sum(S[:50, :])
            total_energy = np.sum(S) + 1e-8
            bass_ratio = low_energy / total_energy
            
            # High frequency energy (brightness adds to energy perception)
            high_energy = np.sum(S[200:, :])  # Above ~2 kHz
            high_ratio = high_energy / total_energy
            
            # Weighted combination: RMS dominates, bass and treble provide nuance
            energy = 0.7 * rms + 0.2 * bass_ratio + 0.1 * high_ratio
            
            return float(max(0.0, min(1.0, energy)))
            
        except Exception as e:
            print(f"Error estimating energy: {e}")
            return 0.5
    
    def estimate_valence(self, y, sr):
        """
        IMPROVED: Estimate valence/mood (0-1 scale, where 1 is positive/happy).
        
        Combines mode detection (major/minor), brightness, and tempo influence.
        """
        try:
            # Chroma features for mode detection
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)
            
            # Major third (index 4) vs minor third (index 3)
            major_score = chroma_mean[4]
            minor_score = chroma_mean[3]
            mode_valence = major_score / (major_score + minor_score + 1e-10)
            
            # Spectral brightness (higher frequencies correlate with happiness)
            centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            brightness = centroid / (sr / 2)
            
            # Tempo influence (faster tempo often = happier, but not always)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            tempo_valence = 0.5 + 0.5 * np.tanh((tempo - 100) / 50)
            
            # Weighted combination: mode is most important
            valence = 0.5 * mode_valence + 0.3 * brightness + 0.2 * tempo_valence
            
            return float(max(0.0, min(1.0, valence)))
            
        except Exception as e:
            print(f"Error estimating valence: {e}")
            return 0.5
    
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
            
            # Spectral rolloff (how quickly spectrum drops off)
            rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            rolloff_factor = 1 - (rolloff / (sr / 2))
            
            # Weighted combination
            acousticness = 0.5 * harmonic_ratio + 0.3 * centroid_factor + 0.2 * rolloff_factor
            
            return float(max(0.0, min(1.0, acousticness)))
            
        except Exception as e:
            print(f"Error estimating acousticness: {e}")
            return 0.5
    
    def estimate_instrumentalness(self, y, sr):
        """
        IMPROVED: Estimate instrumentalness (0-1 scale).
        
        Uses HPSS and vocal frequency range analysis for better vocal detection.
        Higher score = more likely to be instrumental (no vocals).
        """
        try:
            # Harmonic-percussive separation with higher margin
            y_harm, y_perc = librosa.effects.hpss(y, margin=8.0)
            
            # Percussive energy ratio (vocals have percussive attack characteristics)
            perc_energy = np.mean(y_perc**2)
            total_energy = np.mean(y**2) + 1e-8
            perc_ratio = perc_energy / total_energy
            
            # Analyze vocal frequency range (fundamental frequencies: 85-500 Hz)
            S = np.abs(librosa.stft(y))
            n_fft = (S.shape[0] - 1) * 2
            freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
            
            # Find bins in vocal range
            vocal_bins = np.where((freqs >= 85) & (freqs <= 500))[0]
            vocal_energy = np.sum(S[vocal_bins, :])
            total_spec_energy = np.sum(S) + 1e-8
            vocal_ratio = vocal_energy / total_spec_energy
            
            # MFCC variance (vocals have characteristic temporal patterns)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_var = np.mean(np.var(mfcc, axis=1))
            mfcc_factor = 1.0 / (1.0 + mfcc_var / 100.0)
            
            # Combine indicators (higher = more instrumental)
            instrumentalness = 0.4 * (1 - perc_ratio) + 0.4 * (1 - vocal_ratio) + 0.2 * mfcc_factor
            
            return float(max(0.0, min(1.0, instrumentalness)))
            
        except Exception as e:
            print(f"Error estimating instrumentalness: {e}")
            return 0.5
    
    def estimate_speechiness(self, y, sr):
        """
        NEW: Estimate speechiness (0-1 scale).
        
        Detects spoken word content (podcasts, audiobooks, speech) vs music.
        """
        try:
            # Zero crossing rate (speech has higher ZCR than music)
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            zcr_mean = np.mean(zcr)
            
            # Spectral flatness (speech spectrum is flatter than music)
            flatness = np.mean(librosa.feature.spectral_flatness(y=y))
            
            # MFCC analysis (speech has characteristic patterns in lower coefficients)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            # Speech typically has higher variance in lower MFCCs
            speech_indicator = np.mean(np.var(mfcc[:5], axis=1)) / (np.mean(np.var(mfcc[5:], axis=1)) + 1e-10)
            speech_score = min(1.0, speech_indicator / 10.0)
            
            # Rhythm regularity (speech is less rhythmically regular than music)
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            ac = librosa.autocorrelate(onset_env)
            regularity = np.max(ac[1:100]) / (ac[0] + 1e-10)
            irregularity = 1 - min(1.0, regularity)
            
            # Weighted combination
            speechiness = 0.3 * zcr_mean * 10 + 0.3 * flatness + 0.2 * speech_score + 0.2 * irregularity
            
            return float(max(0.0, min(1.0, speechiness)))
            
        except Exception as e:
            print(f"Error estimating speechiness: {e}")
            return 0.1
    
    def estimate_liveness(self, y, sr):
        """
        NEW: Estimate liveness (0-1 scale).
        
        Detects live performance characteristics (audience, ambience, reverb).
        """
        try:
            # Spectral flatness (audience noise has flatter spectrum)
            flatness = np.mean(librosa.feature.spectral_flatness(y=y))
            
            # Dynamic range (live recordings often have more variation)
            rms = librosa.feature.rms(y=y)[0]
            dynamic_range = np.std(rms) / (np.mean(rms) + 1e-10)
            
            # Spectral rolloff variance (applause/crowd noise has unique signature)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            rolloff_var = np.var(rolloff) / (np.mean(rolloff)**2 + 1e-10)
            
            # Background noise floor (look at quieter sections)
            rms_sorted = np.sort(rms)
            noise_floor = np.mean(rms_sorted[:len(rms_sorted)//4])
            noise_ratio = noise_floor / (np.mean(rms) + 1e-10)
            
            # Weighted combination
            liveness = 0.3 * flatness + 0.3 * min(1.0, dynamic_range) + 0.2 * min(1.0, rolloff_var * 100) + 0.2 * noise_ratio
            
            return float(max(0.0, min(1.0, liveness)))
            
        except Exception as e:
            print(f"Error estimating liveness: {e}")
            return 0.1
    
    def detect_key_and_mode(self, y, sr):
        """
        NEW: Detect musical key and mode (major/minor).
        
        Returns:
            tuple: (key, mode) where key is 0-11 (C to B) and mode is 0 (minor) or 1 (major)
        """
        try:
            # Extract chroma features using CQT for better pitch resolution
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            
            # Average over time to get overall key profile
            chroma_mean = np.mean(chroma, axis=1)
            chroma_mean = chroma_mean / (np.sum(chroma_mean) + 1e-10)
            
            # Krumhansl-Schmuckler key profiles
            major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
            minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
            
            # Normalize profiles
            major_profile = major_profile / np.sum(major_profile)
            minor_profile = minor_profile / np.sum(minor_profile)
            
            # Find best matching key and mode
            max_corr = -1
            best_key = 0
            best_mode = 1
            
            for key in range(12):
                rotated = np.roll(chroma_mean, -key)
                
                # Test major
                corr_major = np.corrcoef(rotated, major_profile)[0, 1]
                if corr_major > max_corr:
                    max_corr = corr_major
                    best_key = key
                    best_mode = 1
                
                # Test minor
                corr_minor = np.corrcoef(rotated, minor_profile)[0, 1]
                if corr_minor > max_corr:
                    max_corr = corr_minor
                    best_key = key
                    best_mode = 0
            
            return int(best_key), int(best_mode)
            
        except Exception as e:
            print(f"Error detecting key/mode: {e}")
            return None, None
    
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
        
        # Extract basic features
        features.update(self.extract_spectral_features(y, sr))
        features.update(self.extract_mfcc(y, sr))
        features.update(self.extract_chroma(y, sr))
        features.update(self.extract_energy_features(y))
        
        # Estimate Spotify-like features (IMPROVED VERSIONS)
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
        
        # NEW features
        speechiness = self.estimate_speechiness(y, sr)
        if speechiness is not None:
            features['speechiness'] = speechiness
        
        liveness = self.estimate_liveness(y, sr)
        if liveness is not None:
            features['liveness'] = liveness
        
        # Detect key and mode
        key, mode = self.detect_key_and_mode(y, sr)
        if key is not None:
            features['key'] = key
            features['mode'] = mode
        
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
    print(f"\n{'='*60}")
    print(f"Features extracted and saved to: {output_csv}")
    print(f"Total tracks processed: {len(df)}")
    print(f"Total features per track: {len(df.columns)}")
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("FEATURE STATISTICS")
    print(f"{'='*60}")
    
    if 'tempo' in df.columns:
        print(f"Tempo:            {df['tempo'].mean():.2f} BPM (range: {df['tempo'].min():.1f}-{df['tempo'].max():.1f})")
    if 'danceability' in df.columns:
        print(f"Danceability:     {df['danceability'].mean():.3f} (range: {df['danceability'].min():.3f}-{df['danceability'].max():.3f})")
    if 'energy' in df.columns:
        print(f"Energy:           {df['energy'].mean():.3f} (range: {df['energy'].min():.3f}-{df['energy'].max():.3f})")
    if 'valence' in df.columns:
        print(f"Valence:          {df['valence'].mean():.3f} (range: {df['valence'].min():.3f}-{df['valence'].max():.3f})")
    if 'acousticness' in df.columns:
        print(f"Acousticness:     {df['acousticness'].mean():.3f} (range: {df['acousticness'].min():.3f}-{df['acousticness'].max():.3f})")
    if 'instrumentalness' in df.columns:
        print(f"Instrumentalness: {df['instrumentalness'].mean():.3f} (range: {df['instrumentalness'].min():.3f}-{df['instrumentalness'].max():.3f})")
    if 'speechiness' in df.columns:
        print(f"Speechiness:      {df['speechiness'].mean():.3f} (range: {df['speechiness'].min():.3f}-{df['speechiness'].max():.3f})")
    if 'liveness' in df.columns:
        print(f"Liveness:         {df['liveness'].mean():.3f} (range: {df['liveness'].min():.3f}-{df['liveness'].max():.3f})")
    
    print(f"{'='*60}\n")
    
    return df


if __name__ == "__main__":
    import sys
    
    print("="*60)
    print("IMPROVED Audio Feature Extractor")
    print("Alternative to Spotify API with enhanced algorithms")
    print("="*60)
    print("\nImprovements:")
    print("✓ Better danceability using beat interval regularity")
    print("✓ Improved energy with frequency band analysis")
    print("✓ Enhanced instrumentalness using HPSS + vocal detection")
    print("✓ Added speechiness for podcast/speech detection")
    print("✓ Added liveness for live performance detection")
    print("✓ Added key and mode detection")
    print("="*60 + "\n")
    
    # Example usage
    if len(sys.argv) > 1:
        audio_directory = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else "extracted_audio_features.csv"
    else:
        audio_directory = "Audio_Samples"
        output_file = "Ingested_Data/extracted_audio_features.csv"
    
    extract_features_from_directory(audio_directory, output_file)