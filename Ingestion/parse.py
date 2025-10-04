import pandas as pd
import glob
import os

# --- Configuration ---

data_path = "/Users/godsonajodo/Documents/Fall 2025/CS156/Assignments/Spotify Account Data"
output_path = "Ingested_Data"

# --- Loading the Data ---
# Find all StreamingHistory*.json files in the specified folder
all_files = glob.glob(os.path.join(data_path, "StreamingHistory_music*.json"))


def create_dataset():
    # Create a list to hold the individual dataframes
    list_of_dfs = []
    for file in all_files:
        df = pd.read_json(file)
        list_of_dfs.append(df)
    # Concatenate all dataframes in the list into a single one
    spotify_df = pd.concat(list_of_dfs, ignore_index=True)
    return spotify_df

    # Loop through the files, load each into a dataframe, and append to the list
    for file in all_files:
        df = pd.read_json(file)
        list_of_dfs.append(df)

    # Concatenate all dataframes in the list into a single one
    spotify_df = pd.concat(list_of_dfs, ignore_index=True)

    return spotify_df

def save_to_csv(df, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_file = os.path.join(output_path, "combined_streaming_history.csv")
    df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")



if __name__ == "__main__":
    spotify_df = create_dataset()
    # --- Save the combined dataframe to a CSV file ---
    save_to_csv(spotify_df, output_path)
    print("Successfully loaded the data!")
    print(f"Total tracks listened to: {len(spotify_df)}")
    # --- Display the first few rows to verify ---
    print(spotify_df.head())