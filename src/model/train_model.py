import yaml
import os
import logging
import src.logger
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dropout, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tiktoken
import os

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
CONFIG = load_params("params.yaml")["train_model"]


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

def building_model(vocab_size, num_classes, CONFIG):
    """Build the BiLSTM model."""
    logging.info("Building the model...")
    
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=CONFIG["embedding_dim"], input_length=CONFIG["max_len"]),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    logging.info("Model compilation complete.")
    return model

def train_model(model, X_train, y_train_cat, X_val, y_valid_cat, epochs, batch_size):
    """Training the model."""
    logging.info("Starting model training...")
    model.fit(
        X_train, 
        y_train_cat, 
        validation_data=(X_val, y_valid_cat),
        epochs=CONFIG["epochs"], 
        batch_size=CONFIG["batch_size"]
    )
    logging.info("Model training complete.")

def save_model(model, file_path: str) -> None:
    """Save the trained Keras model to a file. If the file exists, replace it."""
    try:
        # Check if file exists, remove it
        if os.path.exists(file_path):
            os.remove(file_path)
            logging.info(f"Existing model file {file_path} removed.")

        model.save(file_path)
        logging.info(f'Model saved to {file_path}')
    except Exception as e:
        logging.error(f'Error occurred while saving the model: {e}')
        raise



def main():
    try:
        logging.info("Starting the model building process...")

        # Load padded input sequences
        X_train = load_padded_sequence(r"D:\projects\testing\data\interim\train_bpe.npy")
        logging.info(f"Loaded training padded sequences: {X_train.shape}")

        X_val = load_padded_sequence(r"D:\projects\testing\data\interim\validation_bpe.npy")
        logging.info(f"Loaded validation padded sequences: {X_val.shape}")

        # Load labels
        y_train = load_data(r"D:\projects\testing\data\interim\train_bpe.csv")['label'].values
        logging.info(f"Loaded training labels: {y_train.shape}")

        y_valid = load_data(r"D:\projects\testing\data\interim\validation_bpe.csv")['label'].values
        logging.info(f"Loaded validation labels: {y_valid.shape}")

        # Get number of classes
        num_classes = len(np.unique(y_train))
        logging.info(f"Number of classes: {num_classes}")

        # Convert labels to one-hot encoded format 
        y_train_cat = to_categorical(y_train, num_classes)
        y_valid_cat = to_categorical(y_valid, num_classes)
        logging.info("Converted labels to one-hot encoding.")

        # Load training parameters from YAML
        training_config = load_params("params.yaml")["train_model"]
        epochs = training_config["epochs"]
        batch_size = training_config["batch_size"]
        logging.info(f"Training parameters - epochs: {epochs}, batch_size: {batch_size}")

        enc = tiktoken.get_encoding("cl100k_base")
        vocab_size = enc.max_token_value + 1
        logging.info(f"Vocabulary size: {vocab_size}")

        # Build the model
        model = building_model(vocab_size, num_classes, CONFIG)
        logging.info("Model built successfully")

        # Train the model
        clf=train_model(model, X_train, y_train_cat, X_val, y_valid_cat, epochs, batch_size)
        logging.info("Model training completed")

        # Save the model in HDF5 format
        save_model(model, 'models/model.h5')
        logging.info("Model saved to 'models/model.h5'")

    except Exception as e:
        logging.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()


