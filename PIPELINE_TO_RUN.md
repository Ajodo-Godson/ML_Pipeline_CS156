---

## High-level pipeline (order to run)
1. Ingest raw Spotify JSON files into a combined CSV
   - Script: `Ingestion/parse.py`
   - Purpose: Read `StreamingHistory_*.json` files and combine into `Ingested_Data/combined_streaming_history.csv` (or similar CSV)
   - Run:
     python Ingestion/parse.py
   - Expected output: `Ingested_Data/combined_streaming_history.csv`

2. Clean and preprocess the combined streaming history
   - Script: `Ingestion/clean.py`
   - Purpose: Parse timestamps, convert msPlayed to minutes/seconds, filter short plays, clean names, remove duplicates.
   - Run:
     python Ingestion/clean.py
   - Expected output: `Ingested_Data/cleaned_streaming_history.csv`

3.  Download audio preview samples
   - Script: `Ingestion/extract_audio_samples.py`
   - Purpose: Use `spotify-preview-finder` (Node.js) to download preview clips for tracks listed in the cleaned CSV. Saves files to `Audio_Samples/` and a `download_log.csv`.
   - Note: This step requires Node.js and may not find previews for all tracks.
   - Run (example, limit for testing):
     python Ingestion/extract_audio_samples.py --csv Ingested_Data/cleaned_streaming_history.csv --output Audio_Samples --max-tracks 200
   - Expected output: `Audio_Samples/*.mp3` and `Audio_Samples/download_log.csv`

4. Extract audio features from downloaded audio samples (batch-safe)
   - Script: `Ingestion/extract_audio_features_batch.py`
   - Purpose: Extract audio features using `librosa` for each audio file and save incremental progress to `Ingested_Data/audio_features.csv` (default path used by script).
   - Run:
     python Ingestion/extract_audio_features_batch.py Audio_Samples Ingested_Data/audio_features.csv 100
   - Expected output: `Ingested_Data/audio_features.csv`

5. (Alternative) Extract audio features from a single file or directory
   - Script: `Ingestion/extract_audio_features.py`
   - Purpose: Defines `AudioFeatureExtractor` class and helpers. Can be used directly for single-file feature extraction in custom scripts or debugging.

6. Merge streaming history with audio features
   - Script: `Preprocessing/merge_data.py`
   - Purpose: Join `Ingested_Data/cleaned_streaming_history.csv` with `Ingested_Data/audio_features.csv` (matching by normalized artist and track names) to produce `Ingested_Data/merged_dataset.csv`
   - Run:
     python Preprocessing/merge_data.py
   - Expected output: `Ingested_Data/merged_dataset.csv`

7. Fetch artist genres from Spotify API (labels)
   - Script: `Genre_Extraction/fetch_genres.py`
   - Purpose: Use the Spotify API to fetch genres for artists in the dataset and produce `Ingested_Data/dataset_with_genres.csv` (and a cache at `Modelling/artist_genres_cache.csv`). Also creates `Modelling/genre_summary.txt`.
   - Run:
     python Genre_Extraction/fetch_genres.py
   - Expected outputs: `Ingested_Data/dataset_with_genres.csv`, `Modelling/artist_genres_cache.csv`, `Modelling/genre_summary.txt`


NOTE: PROCEED TO CHECK OUT THE JUPYTER NOTEBOOK. THE OTHER STEPS ARE JUST MY PERSONAL WORK TO CREATE A PIPELINE TO TEST INTERACTIVELY: THAT IS, I HAVE A SONG SAMPLE AS INPUT AND THE SAVED MODEL PREDICTS INSTANTLY. 
SO THE MODELLING/ WON'T BE INCLUDED HERE



8. Run exploratory data analysis (EDA)
   - Script: `Preprocessing/eda.py` or open the notebooks
   - Purpose: Generate EDA plots and summary reports; saves outputs in `Preprocessing/EDA_Outputs`.
   - Run:
     python Preprocessing/eda.py
   - Expected outputs: plots and `EDA_Summary_Report.txt` in `Preprocessing/EDA_Outputs`

9. Train and evaluate genre classification model
   - Script: `Modelling/genre_classifier.py`
   - Purpose: Train multiple models (Logistic Regression, Random Forest, Gradient Boosting), evaluate, save best model and artifacts in `Modelling/Genre_Classification_Results/`.
   - Run:
     python Modelling/genre_classifier.py
   - Expected outputs: `Modelling/Genre_Classification_Results/*` (models, reports, plots)

10. Test model predictions on real audio samples
    - Script: `Modelling/test_genre_predictions.py`
    - Purpose: Load saved model and predict genres for audio files (random, specific, or validation against known dataset).
    - Run:
      python Modelling/test_genre_predictions.py
    - Expected output: console predictions and any saved reports.

