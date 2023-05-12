import pickle
import os
import numpy as np
import json

def load_model():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(dir_path, "weights/gbr_model.pkl")
    with open(file_path, "rb") as f:
        clf = pickle.load(f)
    return clf

def get_genres_encoding(genres, config):
    genre_list = config["genres"]
    one_hot_encoding = [1 if genre in genres else 0 for genre in genre_list]
    return one_hot_encoding

def get_date_features(dt):
    weekday = dt.weekday()
    month = dt.month
    day = dt.day
    quarter = (month - 1) // 3 + 1
    return [weekday, month, day, quarter]

def predict(X, twitter_score, reddit_score):
    # Load the config file
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Open the file and load the model
    clf = load_model()
    
    # Get the genre_text and date from X
    genres = X[0][5]
    date = X[0][4]

    # Get the one-hot encoding for the genre
    genre_encoding = get_genres_encoding(genres, config)

    # Get the date features
    date_features = get_date_features(date)

    # Create a dummy input for the model
    # Replace the values in the array with the actual values for the features
    model_input = np.array([[ X[0][3], 10.892, X[0][6], 3.966, 250.0, 0.0, twitter_score[0], twitter_score[1], twitter_score[2], reddit_score[0], reddit_score[1], reddit_score[2], 2.73857322141759, 3.216707972959764, 1.0, *date_features, *genre_encoding ]])

    # Make a prediction using the dummy input
    revenue_prediction = clf.predict(model_input)[0]
    print(f"Predicted revenue: {revenue_prediction}")

    return revenue_prediction