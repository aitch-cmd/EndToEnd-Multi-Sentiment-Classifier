import time
import logging
import src.logger
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np
import pandas as pd
import tiktoken
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dropout, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tiktoken
import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")
from src.data.data_preprocessing import preprocess_dataframe
import yaml

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

# Set up logger
logging.basicConfig(level=logging.INFO)
CONFIG = load_params("params.yaml")["feature_engineering"]

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        logging.info('Data loaded and NaNs filled from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise

def feature_df(df, col='text'):
    enc = tiktoken.get_encoding("cl100k_base")

    # Data Preprocessing(Cleaning)
    df = preprocess_dataframe(df, col)

    # Feature Engineering: Tokenization
    logging.info("Tokenizing text using tiktoken...")
    df["token_ids"] = df[col].apply(lambda text: enc.encode(text))
    logging.info("Tokenization completed.")

    # Feature Engineering: Padding
    logging.info("Padding sequences...")
    padded_sequences = pad_sequences(df["token_ids"].tolist(), maxlen=CONFIG["max_len"], padding="post", truncating="post")
    logging.info("Padding completed.")

    return df, padded_sequences

def main():
    try:
        # Load data
        train_df = load_data('data/interim/train_processed.csv')
        test_df = load_data('data/interim/test_processed.csv')
        val_df = load_data('data/interim/validation_processed.csv')
        logging.info('Data loaded successfully.')

        # Use preprocess_dataframe to clean, tokenize and pad 
        train_df, train_padded = feature_df(train_df, 'text')
        test_df, test_padded = feature_df(test_df, 'text')
        val_df, val_padded = feature_df(val_df, 'text')

        # Save processed data and padded sequences
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)

        train_df.to_csv(os.path.join(data_path, "train_bpe.csv"), index=False)
        test_df.to_csv(os.path.join(data_path, "test_bpe.csv"), index=False)
        val_df.to_csv(os.path.join(data_path, "validation_bpe.csv"), index=False)

        np.save(os.path.join(data_path, "train_bpe.npy"), train_padded)
        np.save(os.path.join(data_path, "test_bpe.npy"), test_padded)
        np.save(os.path.join(data_path, "validation_bpe.npy"), val_padded)

        logging.info('Processed data saved at %s', data_path)
        
    except Exception as e:
        logging.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()

