import tiktoken
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dropout, Dense
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import logging
import mlflow
import time
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import dagshub

mlflow.set_tracking_uri('https://dagshub.com/harshitneverdebugs/testing.mlflow')
dagshub.init(repo_owner='harshitneverdebugs', repo_name='testing', mlflow=True)

mlflow.set_experiment("TF Tokenizer vs Tiktoken BPE")

CONFIG = {
    "training_data_path": r"D:\projects\testing\notebooks\emotions\training.csv",
    "validation_data_path": r"D:\projects\testing\notebooks\emotions\validation.csv",
    "test_data_path": r"D:\projects\testing\notebooks\emotions\test.csv",
    "mlflow_tracking_uri": "https://dagshub.com/harshitneverdebugs/testing.mlflow",
    "dagshub_repo_owner": "harshitneverdebugs",
    "dagshub_repo_name": "testing",
    "experiment_name": "TF Tokenizer vs Tiktoken BPE",
    "max_len": 100,
    "embedding_dim": 64,
    "num_words": 10000,
    "batch_size": 64,
    "epochs": 5,
    "random_seed": 42
}

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

def normalize_text(df):
    """Normalize the text data."""
    try:
        df['text'] = df['text'].apply(lower_case)
        df['text'] = df['text'].apply(remove_stop_words)
        df['text'] = df['text'].apply(removing_numbers)
        df['text'] = df['text'].apply(removing_punctuations)
        df['text'] = df['text'].apply(removing_urls)
        df['text'] = df['text'].apply(lemmatization)
        return df
    except Exception as e:
        print(f'Error during text normalization: {e}')
        raise

train_df= pd.read_csv(CONFIG['training_data_path'])
test_df=pd.read_csv(CONFIG['test_data_path'])
valid_df=pd.read_csv(CONFIG['validation_data_path'])

train_df=normalize_text(train_df)
test_df=normalize_text(test_df)
valid_df=normalize_text(valid_df)

enc=tiktoken.get_encoding("cl100k_base")

def encode_with_tf_tokenizer(texts, tokenizer=None):
    """Encodes texts using TensorFlow tokenizer."""
    if tokenizer is None:
        tokenizer=Tokenizer(num_words=CONFIG['num_words'], oov_token='<OOV>')
        tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    pad=pad_sequences(sequences, maxlen=CONFIG['max_len'], padding='post', truncating='post')
    return pad, tokenizer, sequences


def encode_with_tiktoken(texts):
    """Encodes texts using Tiktoken BPE."""
    sequences=[enc.encode(text) for text in texts]
    padded=pad_sequences(sequences, maxlen=CONFIG['max_len'], padding='post', truncating='post')
    return padded

def build_model(vocab_size, num_classes):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=CONFIG["embedding_dim"], input_length=CONFIG["max_len"]),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_and_evaluate(train_df, valid_df, test_df, tokenizer_type="tf"):
    with mlflow.start_run(run_name=f"BiLSTM with {tokenizer_type} tokenizer"):
        start_time = time.time()

        if tokenizer_type == "tf":
            X_train, tf_tokenizer, sequences = encode_with_tf_tokenizer(train_df["text"])
            X_valid = pad_sequences(tf_tokenizer.texts_to_sequences(valid_df["text"]), maxlen=CONFIG["max_len"], padding='post', truncating='post')
            X_test = pad_sequences(tf_tokenizer.texts_to_sequences(test_df["text"]), maxlen=CONFIG["max_len"], padding='post', truncating='post')
            vocab_size = min(len(tf_tokenizer.word_index) + 1, CONFIG["num_words"])
        elif tokenizer_type == "bpe":
            X_train = encode_with_tiktoken(train_df["text"])
            X_valid = encode_with_tiktoken(valid_df["text"])
            X_test = encode_with_tiktoken(test_df["text"])
            vocab_size = enc.max_token_value + 1

        else:
            raise ValueError("Unsupported tokenizer_type")

        y_train = train_df["label"].values
        y_valid = valid_df["label"].values
        y_test = test_df["label"].values

        num_classes = len(np.unique(y_train))
        y_train_cat = to_categorical(y_train, num_classes)
        y_valid_cat = to_categorical(y_valid, num_classes)
        y_test_cat = to_categorical(y_test, num_classes)

        mlflow.log_params({
            "tokenizer_type": tokenizer_type,
            "vocab_size": vocab_size,
            "embedding_dim": CONFIG["embedding_dim"],
            "max_len": CONFIG["max_len"],
            "epochs": CONFIG["epochs"],
            "batch_size": CONFIG["batch_size"]
        })

        model = build_model(vocab_size, num_classes)
        model.fit(
            X_train, y_train_cat,
            validation_data=(X_valid, y_valid_cat),
            epochs=CONFIG["epochs"],
            batch_size=CONFIG["batch_size"]
        )

        y_pred_prob = model.predict(X_test)
        y_pred = np.argmax(y_pred_prob, axis=1)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })

        model_name = f"bilstm_{tokenizer_type}_model.h5"
        model.save(model_name)
        mlflow.log_artifact(model_name)

        end_time = time.time()
        logging.info(f"Run for {tokenizer_type} tokenizer took {end_time - start_time:.2f} seconds")
        logging.info(f"Metrics - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")



if __name__ == "__main__":
    mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])
    mlflow.set_experiment(CONFIG["experiment_name"])

    train_and_evaluate(train_df, valid_df, test_df, tokenizer_type="tf")
    train_and_evaluate(train_df, valid_df, test_df, tokenizer_type="bpe")
