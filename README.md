
This document lists the scripts in this repository and the recommended order to run them to generate the final dataset used by the notebooks (`ML_Pipeline_Final_Report_NEW.ipynb`, etc.). It includes prerequisites, commands, expected outputs, and quick verification steps.

## Assumptions
- You have Python 3.8+ installed and a virtual environment activated.
- Install Python dependencies from `requirements.txt` before running the scripts:

  pip install -r requirements.txt

- You have Node.js installed if you want to download audio previews (used by `Ingestion/extract_audio_samples.py`).
- Your Spotify API credentials are set in environment variables or a `.env` file with keys `CLIENT_ID` and `CLIENT_SECRET`. The `config.py` uses `dotenv` to load these.
- Raw Spotify data (StreamingHistory JSON files) are placed in a directory referenced by `Ingestion/parse.py` (default: `../Spotify Account Data`).

## Project structure
A compact, easy-to-read project tree describing the repository layout.

```
ML_Pipeline_CS156/
├─ Audio_Samples/                  # downloaded preview clips (.mp3)
├─ Genre_Extraction/               # fetch artist genres (Spotify API)
│  └─ fetch_genres.py
├─ Ingestion/                      # ingestion & feature extraction helpers
│  ├─ __init__.py
│  ├─ parse.py                      # combine StreamingHistory JSON -> CSV
│  ├─ clean.py                      # clean streaming CSV (timestamps, filters)
│  ├─ extract_audio_samples.py      # (optional) download preview clips (Node + helper)
│  ├─ extract_audio_features.py     # AudioFeatureExtractor (single-file)
│  └─ extract_audio_features_batch.py  # batch extractor 
with incremental saves
|   └─ spotify_preview_finder.js  # batch extractor 
├─ Ingested_Data/                  # pipeline outputs (CSVs)
│  ├─ combined_streaming_history.csv
│  ├─ cleaned_streaming_history.csv
│  ├─ audio_features.csv
│  ├─ merged_dataset.csv
│  └─ dataset_with_genres.csv
├─ Preprocessing/                  # preprocessing & EDA
│  ├─ merge_data.py
│  └─ eda.py
├─ Modelling/                      # modelling and prediction utilities
│  ├─ genre_classifier.py
│  ├─ test_genre_predictions.py
│  └─ Genre_Classification_Results/  # saved models, plots, reports
├─ config.py                       # Spotipy client setup (loads .env)
├─ requirements.txt                # Python dependencies
├─ package.json / node_modules/     # node helper used for preview-finder (optional)
└─ ML_Pipeline_Final_Report_NEW.ipynb
```

Short notes:
- Use files under `Ingestion/` to build the initial datasets (CSV output goes to `Ingested_Data/`).
- `Audio_Samples/` is only required if you want to extract local audio features from previews.
- `Genre_Extraction/fetch_genres.py` adds genre labels (writes into `Ingested_Data/`).
- Final modelling and evaluation happen under `Modelling/` and save artifacts under `Modelling/Genre_Classification_Results/`.


