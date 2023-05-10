import streamlit as st
import pandas as pd
import json
import numpy as np
import social_media_scores as scores
import model.predict as predict
import datetime


with open('config.json', 'r') as f:
    config = json.load(f)


def display_results(revenue, reddit_scores, twitter_scores):
    # create a dictionary of the sentiment scores
    twitter_scores = {
        'Movie': twitter_scores[0],
        'Director': twitter_scores[1],
        'Casts': twitter_scores[2]
    }

    reddit_scores = {
        'Movie': reddit_scores[0],
        'Director': reddit_scores[1],
        'Casts': reddit_scores[2]
    }

    # create dataframes from the dictionaries
    twitter_df = pd.DataFrame(twitter_scores, index=['Twitter'])
    reddit_df = pd.DataFrame(reddit_scores, index=['Reddit'])

    # concatenate the dataframes along the rows axis
    sentiment_df = pd.concat([twitter_df, reddit_df])

    # display the dataframe as a table
    st.header("Sentiment scores:")
    st.table(sentiment_df)

    st.header("Estimated Box Office Revenue :")
    revenue_text = f'<span style="color:green; font-size:40px">$ {revenue:.2f}</span>'
    st.markdown(revenue_text, unsafe_allow_html=True)


def get_results(movie_name, director, casts, budget, released_date, genres, runtime):
    X = np.array(
        [[movie_name, director, casts, budget, released_date, genres, runtime]])

    print("Fetching sentiment scores")
    twitter_scores, reddit_scores = scores.get_sentiment_scores(
        movie_name, director, casts)
    print("Predicting revenue")
    revenue = predict.predict(X, reddit_scores, twitter_scores)
    return revenue, reddit_scores, twitter_scores


def render_input_form():
    movie_name = st.text_input(
        "Movie Name", config['prepopulated_example']['movie_name'], key="movie_name_input")
    director = st.text_input("Directors (separate names with commas)",
                             config['prepopulated_example']['directors'], key="director_input")
    casts = st.text_input("Casts (separate names with commas)",
                          config['prepopulated_example']['casts'], key="casts_input")
    budget = st.number_input("Budget (in USD)", min_value=1,
                             value=1000000, step=1000, key="budget_input", format="%i")
    released_date = st.date_input("Released Date", value=datetime.date.today(
    ), key="released_date_input", min_value=datetime.date(1900, 1, 1))
    genres = st.multiselect("Genres", options=config['genres'], default=[
                            'Action', 'Comedy'], key="genres_input")
    runtime = st.number_input("Runtime (in minutes)", min_value=1,
                              value=120, step=1, key="runtime_input", format="%i")

    if st.button("Predict Box Office Revenue"):
        revenue, reddit_scores, twitter_scores = get_results(
            movie_name, director, casts, budget, released_date, genres, runtime)

        display_results(revenue, reddit_scores, twitter_scores)


def render_predict_page():

    st.write("""### We need some information to predict the revenue""")
    render_input_form()


render_predict_page()
