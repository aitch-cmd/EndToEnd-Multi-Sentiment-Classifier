import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import os
import yaml
import logging
# from src.connections import s3_connection

def load_params(params_path:str)-> dict:
    """Loads parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params= yaml.safe_load(file)
        logging.debug(f"Parameters loaded successfully from {params_path}")
        return params
    except FileNotFoundError:
        logging.error(f"Parameters file not found at {params_path}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file at {params_path}: {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading parameters: {e}")
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """Loads data from a CSV file."""
    try:
        df=pd.read_csv(file_path)
        logging.debug(f"Data loaded successfully from {file_path}")
        return df
    except pd.errors.ParserError as e:
        logging.error(f"Error parsing CSV file at {file_path}: {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading data: {e}")
        raise

def save_data(train_data: pd.DataFrame, valid_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Saves the training, validation and testing datasets."""
    try:
        raw_data_path =os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, 'train.csv'), index=False)
        valid_data.to_csv(os.path.join(raw_data_path, 'valid.csv'), index=False)
        test_data.to_csv(os.path.join(raw_data_path, 'test.csv'), index=False)
        logging.debug(f"Test data saved successfully to {raw_data_path}")
    except Exception as e:
        logging.error(f"An error occurred while saving data: {e}")
        raise

def main():
    try:
        params=load_params('params.yaml')
        train_data=load_data(r"D:\projects\testing\notebooks\emotions\training.csv")
        valid_data=load_data(r"D:\projects\testing\notebooks\emotions\validation.csv")
        test_data=load_data(r"D:\projects\testing\notebooks\emotions\test.csv")

        # s3 = s3_connection.s3_operations("emotion-df-s3", "AKIAS2VS4C2QDGEUSDOW", "CKz0HsiH/c2NtY7uYkQFFRS/0Uu3oaoe2bXOIfr+")
        # train_data = s3.fetch_file_from_s3("training.csv")
        # valid_data = s3.fetch_file_from_s3("validation.csv")
        # test_data = s3.fetch_file_from_s3("test.csv")

        save_data(train_data, valid_data, test_data, data_path='./data')
    except Exception as e:
        logging.error(f"An error occurred in the main function: {e}")
        raise

if __name__ == '__main__':
    main()