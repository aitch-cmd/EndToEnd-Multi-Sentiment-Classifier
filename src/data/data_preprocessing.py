import os
import re
import string
import time
import logging
import src.logger
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dropout, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tiktoken
import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

nltk.download('wordnet')
nltk.download('stopwords')

def preprocess_dataframe(df, col='text'):
    """Preprocesses the input DataFrame by cleaning text data, tokenizing, and padding sequences."""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    def lemmatization(text):
        """Lemmatizes the input text."""
        lemmatizer=WordNetLemmatizer()
        words=text.split()
        lemmatized_words=[]

        for word in words:
            word=lemmatizer.lemmatize(word)
            lemmatized_words.append(word)
        
        result_text=' '.join(lemmatized_words)
        return result_text

    def lower_case(text):
        """Converts the input text to lower case."""
        return text.lower()

    def remove_stop_words(text):
        """Removes stopwords from the input text."""
        stop_words=set(stopwords.words('english'))
        words=text.split()
        filtered_words=[]

        for word in words:
            if word not in stop_words:
                filtered_words.append(word)
        
        result_text=' '.join(filtered_words)
        return result_text

    def removing_numbers(text):
        """Removes numbers from input_text."""
        filtered_words=[]
        words=text.split()

        for word in words:
            if not word.isdigit():
                filtered_words.append(word)
        result_text=' '.join(filtered_words)
        return result_text

    def removing_urls(text):
        """Removes URLs from the text"""
        url_pattern = r'https?://\S+|www.\S+'
        result_text= re.sub(url_pattern, ' ', text)
        return result_text

    def removing_punctuations(text):
        """Removes punctuation form text"""
        punctuation_pattern = f"[{re.escape(string.punctuation)}]"
        result_text = re.sub(punctuation_pattern, ' ', text)
        return result_text
    
    logging.info("Applying text cleaning steps...")
    df[col] = df[col].astype(str)
    df[col] = df[col].apply(lower_case)
    df[col] = df[col].apply(remove_stop_words)
    df[col] = df[col].apply(removing_numbers)
    df[col] = df[col].apply(removing_punctuations)
    df[col] = df[col].apply(removing_urls)
    df[col] = df[col].apply(lemmatization)
    logging.info("Text cleaning completed.")

 
    return df

def main():
    try:
        # Fetch the data from data/raw
        train_data = pd.read_csv(r'data\raw\train.csv')
        test_data = pd.read_csv(r'data\raw\test.csv')
        validation_data = pd.read_csv(r'data\raw\valid.csv')
        logging.info('data loaded properly')

        # Transform the data
        train_processed_data = preprocess_dataframe(train_data, 'text')
        test_processed_data = preprocess_dataframe(test_data, 'text')
        validation_processed_data = preprocess_dataframe(validation_data, 'text')

        # Store the data inside data/processed
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)
        
        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
        validation_processed_data.to_csv(os.path.join(data_path, "validation_processed.csv"), index=False)

        logging.info('Processed data saved to %s', data_path)
    except Exception as e:
        logging.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()



