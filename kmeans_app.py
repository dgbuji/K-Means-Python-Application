import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

# Define the dataset and features based on your Parquet file
FILE_PATH = 'yellow_tripdata_2025-08.parquet'
FEATURES = ['trip_distance', 'fare_amount', 'total_amount']
OPTIMAL_K = 4  # Placeholder: This should be determined by the Elbow Method visually


# --- Function Definitions (Required for Unit Tests) ---

def load_data(file_path):
    """Loads data from a local Parquet file."""
    try:
        df = pd.read_parquet(file_path, engine='pyarrow')
        print(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns.")
        return df
    except Exception as e:
        print(f"Error loading Parquet file: {e}")
        return None


def preprocess_data(df, features):
    """
    Performs feature selection, imputation, and scaling.
    Returns a NumPy array of scaled data (X_scaled).
    """
    # 1. Feature Selection
    X = df[features].copy()

    # 2. Handle Missing Values (Imputation)
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # 3. Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    return X_scaled


def find_optimal_k(data, max_k=10):
    """Uses the Elbow Method to find the optimal number of clusters K."""
    inertia = []
    # Test K from 1 up to max_k
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(data)
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_k + 1), inertia, marker='o')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia (Sum of Squared Errors)')
    plt.grid(True)
    plt.savefig('elbow_method.png')  # Save plot for analysis in the summary
    print("Saved Elbow Method plot to 'elbow_method.png'")
    # Note: This function focuses on visualization, returning only for completeness
    # of the test, but the K choice is manual after viewing the plot.
    return inertia


def run_kmeans(data, k):
    """Fits the final K-Means model and returns cluster labels and the model object."""
    # Use n_init='auto' to avoid warnings and leverage smart initialization
    final_kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = final_kmeans.fit_predict(data)
    return labels, final_kmeans


# --- Main Application Execution ---

if __name__ == '__main__':
    data_df = load_data(FILE_PATH)

    if data_df is not None:
        # Preprocessing
        X_scaled = preprocess_data(data_df, FEATURES)
        print("\nData preprocessing and scaling complete.")

        # Step 1: Find Optimal K
        # Note: You must manually look at the saved 'elbow_method.png' to set the OPTIMAL_K variable.
        print("Running Elbow Method...")
        find_optimal_k(X_scaled)

        # Step 2: Run Final K-Means
        print(f"\nRunning final K-Means with K={OPTIMAL_K}...")
        cluster_labels, kmeans_model = run_kmeans(X_scaled, OPTIMAL_K)

        # Attach clusters to the original DataFrame for analysis
        data_df['Cluster'] = cluster_labels

        # Step 3: Analyze Cluster Centroids (for Executive Summary)
        cluster_analysis = data_df.groupby('Cluster')[FEATURES].mean()
        print("\nCluster Centroid Analysis (Mean values for each cluster):")
        print(cluster_analysis)

        # Save the clustered data for further use/verification
        data_df.to_csv('clustered_data.csv', index=False)
        print("\nClustering complete. Results saved to 'clustered_data.csv' and 'elbow_method.png'.")