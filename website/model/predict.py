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
    dummy_input = np.array([[
        340000000.0, # budget
        10.892, # popularity
        81.0, # runtime
        3.966, # vote_average
        250.0, # vote_count
        0.0, # rating
        0.0, # twitter_movie_score
        0.044497194648252, # twitter_cast_average_score
        0.3129770992366412, # twitter_director_score
        0.4369369369369369, # reddit_movie_score
        0.5351854283457296, # reddit_cast_average_score
        0.2815533980582524, # reddit_director_score
        2.73857322141759, # log_revenue
        3.216707972959764, # log_budget
        1.0, # has_homepage (True, since there's a homepage in the JSON)
        *date_features, # Unpack the date_features list
        *genre_encoding # Unpack the genre_encoding list
    ]])

    # Make a prediction using the dummy input
    revenue_prediction = clf.predict(dummy_input)[0]
    print(f"Predicted revenue: {revenue_prediction}")

    return revenue_prediction