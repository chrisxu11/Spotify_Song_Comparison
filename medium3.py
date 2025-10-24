import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import re

CSV_PATH = "Popular_Spotify_Songs.csv"
OUTPUT_DIR = "results"
TOP_K = 10
QUERY_SONGS = [
    ("Flowers", "Miley Cyrus"),
    ("vampire", "Olivia Rodrigo"),
    ("Blinding Lights", "The Weeknd")
]
FEATURES = [
    "bpm", "danceability_%", "energy_%", "valence_%",
    "acousticness_%", "instrumentalness_%", "liveness_%",
    "speechiness_%", "mode"
]

def load_data(filepath, feature_cols):
    """Loads, cleans, and prepares the Spotify dataset."""
    df = pd.read_csv(filepath, encoding='latin1')
    df.columns = df.columns.str.strip()
    required_cols = feature_cols + ["track_name", "artist(s)_name"]
    df = df.dropna(subset=required_cols).copy()

    # Convert 'mode' from categorical to numerical (Major=1, Minor=0)
    df['mode'] = (df['mode'] == 'Major').astype(int)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])
    
    return df.reset_index(drop=True), X_scaled

def find_track(df, song, artist):
    """Finds the index of a track in the DataFrame, returns index and row."""
    matches = df[
        (df["track_name"].str.lower() == song.lower()) &
        (df["artist(s)_name"].str.lower().str.contains(artist.lower()))
    ]
    if matches.empty:
        matches = df[df["track_name"].str.lower().str.contains(song.lower())]

    return matches.index[0], matches.iloc[0]

def get_recommendations(X_scaled, query_idx, k):
    """Calculates cosine similarity and returns the top k indices and scores."""
    query_vector = X_scaled[query_idx].reshape(1, -1)
    similarities = cosine_similarity(query_vector, X_scaled).flatten()
    similarities[query_idx] = -np.inf 
    top_indices = np.argsort(-similarities)[:k]
    
    return top_indices, similarities[top_indices]

def plot(df, X_scaled, query_idx, rec_indices, song_name, artist_name, filepath):
    """Generates and saves a PCA plot for the query and recommended songs."""
    all_indices = [query_idx] + list(rec_indices)
    pca_data = X_scaled[all_indices]
    
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(pca_data)
    
    plt.figure(figsize=(12, 8))

    plt.scatter(pca_result[1:, 0], pca_result[1:, 1], c='blue', label=f'Top {TOP_K} Recommendations')

    plt.scatter(pca_result[0, 0], pca_result[0, 1], c='orange', s=150, edgecolor='black', zorder=5, label=f'Query: {song_name}')
    

    labels = df.loc[all_indices, 'track_name']
    for i, label in enumerate(labels):
        plt.annotate(label, (pca_result[i, 0], pca_result[i, 1]), textcoords="offset points", xytext=(0,10), ha='center')

    plt.title(f'PCA of Similar Songs for "{song_name}" by {artist_name}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(True)
    plt.savefig(filepath)
    plt.close() # Free up memory

def main():
    """Main function to run the song recommendation process."""
    df, X_scaled = load_data(CSV_PATH, FEATURES)
    if df is None:
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Data loaded successfully. Total songs after cleaning: {len(df)}")

    for song, artist in QUERY_SONGS:
            query_idx, query_row = find_track(df, song, artist)
            print(f"\nrecommendations for: '{query_row['track_name']}' by {query_row['artist(s)_name']}")
       

        rec_indices, rec_scores = get_recommendations(X_scaled, query_idx, k=TOP_K)
        
        results_df = df.loc[rec_indices].copy()
        results_df['similarity'] = rec_scores
        
        safe_filename = re.sub(r'[^\w\s-]', '', f"{song}_{artist}").strip().replace(' ', '_')
        
        # Save csv
        csv_path = os.path.join(OUTPUT_DIR, f"top_{TOP_K}_for_{safe_filename}.csv")
        results_df[["track_name", "artist(s)_name", "released_year", "similarity"]].to_csv(csv_path, index=False)
        print(f"Saved top {TOP_K} recommendations to {csv_path}")

        # Save PCA plot
        plot_path = os.path.join(OUTPUT_DIR, f"pca_plot_for_{safe_filename}.png")
        plot(df, X_scaled, query_idx, rec_indices, song, artist, plot_path)

if __name__ == "__main__":
    main()
