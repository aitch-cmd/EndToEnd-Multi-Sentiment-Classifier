from flask import Flask, render_template, request
import mlflow
import pickle
import os
import numpy as np
import pandas as pd
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
import time
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import re
import dagshub
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tiktoken
import yaml

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        return params
    except Exception as e:
        print(f"Error loading params: {e}")
        # Return default values if params file not found
        return {"feature_engineering": {"max_len": 100}}

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

def remove_small_sentences(df):
    """Remove sentences with less than 3 words."""
    for i in range(len(df)):
        text_value = df.text.iloc[i]
        if isinstance(text_value, str) and len(text_value.split()) < 3:
            df.text.iloc[i] = np.nan
        elif pd.isna(text_value):
            continue


def normalize_text(text):
    text = lower_case(text)
    text = remove_stop_words(text)
    text = removing_numbers(text)
    text = removing_punctuations(text)
    text = removing_urls(text)
    text = lemmatization(text)
    return text

def preprocess_text_for_prediction(text, max_len=100):
    """
    Preprocess text for prediction using the same pipeline as training.
    """
    # Normalize text
    text = normalize_text(text)
    
    # Tokenize using tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    token_ids = enc.encode(text)
    
    # Pad sequences
    padded_sequence = pad_sequences([token_ids], maxlen=max_len, padding="post", truncating="post")
    
    return padded_sequence

# Below code block is for local use
# -------------------------------------------------------------------------------------
# mlflow.set_tracking_uri('https://dagshub.com/harshitneverdebugs/testing.mlflow')
# dagshub.init(repo_owner='harshitneverdebugs', repo_name='testing', mlflow=True)
# -------------------------------------------------------------------------------------

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
# Initialize Flask app
app = Flask(__name__)

# Create a custom registry
registry = CollectorRegistry()

# Define your custom metrics using this registry
REQUEST_COUNT = Counter(
    "app_request_count", "Total number of requests to the app", ["method", "endpoint"], registry=registry
)
REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds", "Latency of requests in seconds", ["endpoint"], registry=registry
)
PREDICTION_COUNT = Counter(
    "model_prediction_count", "Count of predictions for each class", ["prediction"], registry=registry
)

# ------------------------------------------------------------------------------------------
# Model setup
model_name = "my_model"

def get_latest_model_version(model_name):
    client = mlflow.MlflowClient()
    latest_version = client.get_latest_versions(model_name, stages=["Production", "Staging"])
    if not latest_version:
        latest_version = client.get_latest_versions(model_name, stages=["None"])
    return latest_version[0].version if latest_version else None

# Load model
model_version = get_latest_model_version(model_name)
model_uri = f'models:/{model_name}/{model_version}'
print(f"Fetching model from: {model_uri}")
model = mlflow.pyfunc.load_model(model_uri)

# Load configuration for max_len parameter
try:
    CONFIG = load_params("params.yaml")["feature_engineering"]
    MAX_LEN = CONFIG["max_len"]
except:
    MAX_LEN = 100  # Default value
    print("Using default max_len value of 100")

# Routes
@app.route("/")
def home():
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    start_time = time.time()
    response = render_template("index.html", result=None)
    REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start_time)
    return response

def safe_extract_scalar(arr):
    arr = np.array(arr)
    if arr.size == 1:
        return int(arr.item())
    elif arr.ndim == 2 and arr.shape[0] == 1:
        # 2D with one row, possibly multiple columns (e.g., [[1,2,3]])
        return int(np.argmax(arr[0]))
    elif arr.ndim == 1 and arr.size > 1:
        # 1D array with multiple elements (e.g., [1,2,3])
        return int(np.argmax(arr))
    else:
        # Fallback: try to convert the first element
        return int(arr.flat[0])

@app.route("/predict", methods=["POST"])
def predict():
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    start_time = time.time()

    text = request.form["text"]
    processed_features = preprocess_text_for_prediction(text, MAX_LEN)
    feature_names = [f"feature_{i}" for i in range(processed_features.shape[1])]
    features_df = pd.DataFrame(processed_features, columns=feature_names)

    result = model.predict(features_df)

    def extract_prediction(result):
        if isinstance(result, (pd.Series, pd.DataFrame)):
            arr = result.values
        else:
            arr = np.array(result)
        arr = arr.flatten()
        if arr.size > 1:
            return int(np.argmax(arr))
        elif arr.size == 1:
            return int(arr[0])
        else:
            raise ValueError("Model output is empty or not understood.")

    prediction = extract_prediction(result)

    print("Predicted label:", prediction)

    PREDICTION_COUNT.labels(prediction=str(prediction)).inc()
    REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)
    return render_template("index.html", result=prediction)


@app.route("/metrics", methods=["GET"])
def metrics():
    """Expose only custom Prometheus metrics."""
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}

if __name__ == "__main__":
    # app.run(debug=True) # for local use
    app.run(debug=True, host="0.0.0.0", port=5000)  # Accessible from outside Docker
    