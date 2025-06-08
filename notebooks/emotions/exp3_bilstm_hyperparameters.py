import os
import re
import string
import time
import logging
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dropout, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from scikeras.wrappers import KerasClassifier
import mlflow
import tiktoken
import dagshub
import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)

mlflow.set_tracking_uri('https://dagshub.com/harshitneverdebugs/testing.mlflow')
dagshub.init(repo_owner='harshitneverdebugs', repo_name='testing', mlflow=True)

mlflow.set_experiment("BiLSTM Hyperparameter Tuning")

CONFIG = {
    "training_data_path": r"D:\projects\testing\notebooks\emotions\training.csv",
    "validation_data_path": r"D:\projects\testing\notebooks\emotions\validation.csv",
    "test_data_path": r"D:\projects\testing\notebooks\emotions\test.csv",
    "mlflow_tracking_uri": "https://dagshub.com/harshitneverdebugs/testing.mlflow",
    "experiment_name": "BiLSTM Hyperparameter Tuning",
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

train_df = pd.read_csv(CONFIG['training_data_path'])
valid_df = pd.read_csv(CONFIG['validation_data_path'])
test_df = pd.read_csv(CONFIG['test_data_path'])

train_df = normalize_text(train_df)
valid_df = normalize_text(valid_df)
test_df = normalize_text(test_df)

# Initialize Tiktoken encoder
enc = tiktoken.get_encoding("cl100k_base")

def encode_with_tiktoken(texts):
    sequences = [enc.encode(text) for text in texts]
    padded = pad_sequences(sequences, maxlen=CONFIG['max_len'], padding='post', truncating='post')
    return padded

X_train = encode_with_tiktoken(train_df["text"])
X_valid = encode_with_tiktoken(valid_df["text"])
X_test = encode_with_tiktoken(test_df["text"])

y_train = train_df["label"].values
y_valid = valid_df["label"].values
y_test = test_df["label"].values

num_classes = len(np.unique(y_train))
y_train_cat = to_categorical(y_train, num_classes)
y_valid_cat = to_categorical(y_valid, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

def build_model(embedding_dim=64, lstm_units=64, dropout_rate=0.5, num_lstm_layers=1):
    """Builds the BiLSTM model."""
    model=Sequential()
    model.add(Embedding(input_dim=enc.max_token_value+1, output_dim=embedding_dim, input_length=CONFIG['max_len']))

    for _ in range(num_lstm_layers):
        model.add(Bidirectional(LSTM(lstm_units, return_sequences=False)))
        model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Wrap the model for use with scikit-learn
model = KerasClassifier(
    build_fn=build_model,
    epochs=CONFIG["epochs"],
    batch_size=CONFIG["batch_size"],
    verbose=0,
    embedding_dim=CONFIG["embedding_dim"],
    lstm_units=64,  
    dropout_rate=0.5,  
    num_lstm_layers=1  
)

param_dist = {
    'embedding_dim': [32, 64, 128],
    'lstm_units': [32, 64, 128],
    'dropout_rate': [0.3, 0.5, 0.7],
    'num_lstm_layers': [1, 2]
}

random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=5, cv=3, verbose=1, random_state=CONFIG["random_seed"])

with mlflow.start_run(run_name="BiLSTM with Tiktoken BPE and RandomizedSearchCV"):
    start_time = time.time()
    
    # Fit the model
    random_search_result = random_search.fit(X_train, y_train_cat)
    
    # Log best parameters
    mlflow.log_params(random_search_result.best_params_)
    
    # Evaluate on validation set
    val_accuracy = random_search_result.score(X_valid, y_valid_cat)
    mlflow.log_metric("val_accuracy", val_accuracy)
    
    # Predict on test set
    y_pred_prob = random_search_result.best_estimator_.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Log metrics
    mlflow.log_metrics({
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1_score": f1
    })
    
    
    # Save the model
    model_path = "models/bilstm_tiktoken_model.h5"
    best_model = random_search_result.best_estimator_.model_
    best_model.save(model_path)

    mlflow.log_artifact(model_path)

    
    end_time = time.time()
    logging.info(f"Total time for RandomizedSearchCV: {end_time - start_time:.2f} seconds")
    logging.info(f"Test Metrics - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}"   )