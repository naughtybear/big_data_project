import pandas as pd
from sqlalchemy import create_engine
import psycopg2

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class SentimentScore(metaclass=Singleton):
    def __init__(self) -> None:
        # Establish a connection to the PostgreSQL database
        print("Connecting to the Google Cloud SQL...")
        self.conn = psycopg2.connect(
            host="34.30.45.126",
            port=5432,
            database="finalproject",
            user="postgres",
            password="teamrocket"
        )

        # Create a cursor
        cur = self.conn.cursor()

        # Create a SQLAlchemy engine
        engine = create_engine('postgresql+psycopg2://', creator=lambda: self.conn)

        # Define the table names for the data
        movie_table_name = "movie_data"
        twitter_data = ['twitter_movie_score',
                        'twitter_cast_score', 'twitter_director_score']
        reddit_data = ['reddit_movie_title', 'reddit_director',
                    'reddit_cast_score', 'reddit_cast_avg_score']

        # Read these data from the table as dataframes
        with engine.connect() as con:
            self.df_movie = pd.read_sql_table(movie_table_name, con=con)
            self.df_twitter_movie_score = pd.read_sql_table(twitter_data[0], con=con)
            self.df_twitter_cast_score = pd.read_sql_table(twitter_data[1], con=con)
            self.df_twitter_director_score = pd.read_sql_table(twitter_data[2], con=con)
            self.df_reddit_movie_title = pd.read_sql_table(reddit_data[0], con=con)
            self.df_reddit_director = pd.read_sql_table(reddit_data[1], con=con)
            self.df_reddit_cast_score = pd.read_sql_table(reddit_data[2], con=con)
            self.df_reddit_average_score = pd.read_sql_table(reddit_data[3], con=con)


    def find_director_scores(self, director_name):
        director_twitter_score = self.df_twitter_director_score.query(
            'director_name == @director_name')['score'].mean()
        if pd.isna(director_twitter_score):
            director_twitter_score = 0.0
        director_reddit_score = self.df_reddit_director.query(
            'director == @director_name')['director_l/d_ratio'].mean()
        if pd.isna(director_reddit_score):
            director_reddit_score = 0.0
        return director_twitter_score, director_reddit_score


    def find_cast_scores(self, cast_name):
        cast_twitter_score = self.df_twitter_cast_score.query(
            'cast_name == @cast_name')['score'].mean()
        if pd.isna(cast_twitter_score):
            cast_twitter_score = 0.0
        cast_reddit_score = self.df_reddit_cast_score.query(
            'Name == @cast_name')['score'].mean()
        if pd.isna(cast_reddit_score):
            cast_reddit_score = 0.0
        return cast_twitter_score, cast_reddit_score


    def find_movie_scores(self, movie_name):
        movie_twitter_score = self.df_twitter_movie_score.query(
            'movie_title == @movie_name')['score'].mean()
        if pd.isna(movie_twitter_score):
            movie_twitter_score = 0.0
        movie_reddit_score = self.df_reddit_movie_title.query(
            'original_title == @movie_name')['title_l/d_ratio'].mean()
        if pd.isna(movie_reddit_score):
            movie_reddit_score = 0.0
        return movie_twitter_score, movie_reddit_score


    # Close the connection
    def __del__(self):
        self.conn.close()
