import unittest
import pandas as pd
import numpy as np
import os
# Import all functions from your main application file
from kmeans_app import load_data, preprocess_data, run_kmeans
from sklearn.cluster import KMeans


class TestKMeansApplication(unittest.TestCase):
    # Define a constant for the temporary test file
    TEST_FILE_PATH = 'test_dummy.parquet'

    # Define the features that your main script uses for clustering
    FEATURES = ['trip_distance', 'fare_amount', 'total_amount']

    def setUp(self):
        """Setup a dummy Parquet file for testing before each test method."""
        # Create mock data that resembles the structure of the actual Parquet file
        # We include a NaN value to explicitly test the SimpleImputer in preprocessing.
        dummy_data = pd.DataFrame({
            'trip_distance': [1.0, 5.5, 0.5, 10.0],
            'fare_amount': [10.0, 25.0, 5.0, np.nan],  # Introduce a missing value
            'total_amount': [15.0, 30.0, 8.0, 50.0],
            'unrelated_col': ['A', 'B', 'C', 'D']
        })

        # Write the dummy data to a Parquet file
        dummy_data.to_parquet(self.TEST_FILE_PATH, index=False)

    def tearDown(self):
        """Clean up the dummy file after each test method."""
        if os.path.exists(self.TEST_FILE_PATH):
            os.remove(self.TEST_FILE_PATH)

    # --- Test 1: Data Loading ---

    def test_load_data_success(self):
        """Test successful loading and checking for essential columns."""
        df = load_data(self.TEST_FILE_PATH)
        self.assertIsInstance(df, pd.DataFrame, "Loaded object should be a DataFrame.")
        self.assertTrue(all(col in df.columns for col in self.FEATURES),
                        f"DataFrame should contain all clustering features: {self.FEATURES}")

    def test_load_data_failure(self):
        """Test failure when file path is invalid."""
        df = load_data('non_existent_file.parquet')
        self.assertIsNone(df, "Should return None on load failure.")

    # --- Test 2: Preprocessing and Scaling ---

    def test_preprocess_data(self):
        """Test that data is correctly scaled and imputed."""
        df = load_data(self.TEST_FILE_PATH)
        X_scaled = preprocess_data(df, self.FEATURES)

        self.assertIsInstance(X_scaled, np.ndarray, "Output must be a NumPy array.")
        self.assertEqual(X_scaled.shape, (4, len(self.FEATURES)),
                         "Output shape should match (rows, features).")

        # Check for standard scaling properties (mean near 0, std dev near 1)
        # We use 'almost equal' assertion due to floating point math
        self.assertTrue(np.all(np.isclose(X_scaled.mean(axis=0), 0, atol=1e-10)),
                        "Scaled data mean should be close to 0.")
        self.assertTrue(np.all(np.isclose(X_scaled.std(axis=0), 1, atol=1e-10)),
                        "Scaled data standard deviation should be close to 1.")

    # --- Test 3: K-Means Clustering ---

    def test_run_kmeans_labels(self):
        """Test that K-Means runs and returns the correct number of labels."""
        df = load_data(self.TEST_FILE_PATH)
        X_scaled = preprocess_data(df, self.FEATURES)

        K = 2  # Test with 2 clusters
        labels, model = run_kmeans(X_scaled, K)

        self.assertIsInstance(model, KMeans, "Returned object should be a KMeans instance.")
        self.assertEqual(len(labels), X_scaled.shape[0], "Labels length must match data points count.")
        self.assertEqual(len(np.unique(labels)), K, "Should create the specified number of clusters K.")


if __name__ == '__main__':
    # This runs the tests when you execute the file in PyCharm
    unittest.main(argv=['first-arg-is-ignored'], exit=False)