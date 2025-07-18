# load test + signature test + performance test
import logging
import unittest
import mlflow
import os
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tiktoken

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logging.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logging.error('YAML error: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error: %s', e)
        raise

logging.basicConfig(level=logging.INFO)
CONFIG = load_params("params.yaml")["test_model"]


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise

def load_padded_sequence(file_path: str) -> np.ndarray:
    """Load padded sequence data from a .npy file."""
    try:
        data = np.load(file_path)
        logging.info('Padded sequence loaded from %s', file_path)
        return data
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the padded sequence: %s', e)
        raise

class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set up DagsHub credentials for MLflow tracking
        dagshub_token = os.getenv("CAPSTONE_TEST")
        if not dagshub_token:
            raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner = "harshitneverdebugs"
        repo_name = "testing"

        # Set up MLflow tracking URI
        mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

        # Load the new model from MLflow model registry
        cls.new_model_name = "my_model"
        cls.new_model_version = cls.get_latest_model_version(cls.new_model_name)
        cls.new_model_uri = f'models:/{cls.new_model_name}/{cls.new_model_version}'
        cls.new_model = mlflow.pyfunc.load_model(cls.new_model_uri)

        # Load the vectorizer
        cls.vectorizer = load_model('models/model.h5')

        # Load holdout test data
        cls.holdout_data = pd.read_csv('data/interim/test_processed.csv')

    @staticmethod
    def get_latest_model_version(model_name, stage="Staging"):
        client = mlflow.MlflowClient()
        latest_version = client.get_latest_versions(model_name, stages=[stage])
        return latest_version[0].version if latest_version else None

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.new_model)

    def test_model_signature(self):
        enc=tiktoken.get_encoding("cl100k_base")
        input_text = "hi how are you"
        input_seq=enc.encode(input_text)

        padded_input=pad_sequences([input_seq], maxlen=CONFIG["max_len"], padding='post', truncating='post')

        input_array = np.array(padded_input, dtype=np.float32)  
        input_df = pd.DataFrame(input_array)

        prediction = self.vectorizer.predict(input_array)

        y_train = load_data('data/interim/test_bpe.csv')['label'].values
        num_classes = len(np.unique(y_train))
        # Assertions
        self.assertEqual(padded_input.shape[1], CONFIG["max_len"])
        self.assertEqual(prediction.shape[0], 1)  # 1 sample
        self.assertEqual(prediction.shape[1], num_classes)

    def test_model_performance(self):
        enc = tiktoken.get_encoding("cl100k_base")
        
        # Tokenize and pad the holdout texts
        tokenized = self.holdout_data['text'].apply(lambda text: enc.encode(text))
        padded_sequences = pad_sequences(tokenized.tolist(), maxlen=CONFIG["max_len"], padding="post", truncating="post")
        
        X_holdout = np.array(padded_sequences, dtype=np.float32)
        y_holdout = self.holdout_data['label'].values

        # Predict using the new model
        y_pred_new = self.new_model.predict(X_holdout)
        y_pred_classes = np.argmax(y_pred_new, axis=1)

        # Calculate performance metrics
        accuracy_new = accuracy_score(y_holdout, y_pred_classes)
        precision_new = precision_score(y_holdout, y_pred_classes, average='weighted')
        recall_new = recall_score(y_holdout, y_pred_classes, average='weighted')
        f1_new = f1_score(y_holdout, y_pred_classes, average='weighted')

        # Define expected thresholds
        expected_accuracy = 0.40
        expected_precision = 0.40
        expected_recall = 0.40
        expected_f1 = 0.40

        # Assert performance
        self.assertGreaterEqual(accuracy_new, expected_accuracy)
        self.assertGreaterEqual(precision_new, expected_precision)
        self.assertGreaterEqual(recall_new, expected_recall)
        self.assertGreaterEqual(f1_new, expected_f1)



if __name__ == "__main__":
    unittest.main()
    