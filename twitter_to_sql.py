import psycopg2
import pandas as pd
import sqlalchemy
import numpy as np
from twitter_function import get_twitter_data
from twitter_sentiment import twitter_sentiment
import re
import tweepy
import time

def connect_sql():
    conn = psycopg2.connect(
        host="34.30.45.126",
        port=5432,
        database="finalproject",
        user="postgres",
        password="teamrocket")

    # create a cursor
    cur = conn.cursor()

    engine = sqlalchemy.create_engine(
        sqlalchemy.engine.url.URL.create(
            drivername="postgresql",
            username="postgres",
            password="teamrocket",
            host="34.30.45.126",
            port=5432,
            database="finalproject",
        ),
    )
    return cur, conn, engine

def get_twitter_movie_raw_data(cur, conn, engine):
    query = "SELECT t1.id, t1.movie_title, t1.release_date, t1.tweet_end_id\
        FROM twitter_movie_score t1\
        where t1.release_date is not NULL and t1.tweet_end_id = 0\
        order by t1.release_date DESC\
        limit 1"
    cur.execute(query)
    twitter_list = cur.fetchall()
    # 
    # print(twitter_list[0][0])
    # input()
    api = get_twitter_data()
    max_id = 2650506631677968385
    current_id = api.get_current_id()
    time.sleep(0.5)
    count = 0

    while True:
        print(twitter_list)
        if len(twitter_list[0][1]) < 15:
            movie_tweets = api.get_search_data(twitter_list[0][1] + " movie", max_id = max_id)
        else:
            movie_tweets = api.get_search_data(twitter_list[0][1], max_id = max_id)
        if movie_tweets is None:
            if count == 0:
                query = f"update twitter_movie_score\
                    set tweet_end_id = {current_id}\
                    where twitter_movie_score.id = {twitter_list[0][0]}"
                print(query)
                cur.execute(query)
                conn.commit()
            break

        max_id = movie_tweets.iloc[-1].tweet_id - 1
        movie_tweets['movie_id'] = twitter_list[0][0]
        movie_tweets.to_sql("twitter_movie_raw_data", engine, if_exists="append", index=False)

        if count == 0 and movie_tweets.iloc[0].tweet_id + 1 > twitter_list[0][3]:
            query = f"update twitter_movie_score\
                set tweet_end_id = {movie_tweets.iloc[0].tweet_id + 1}\
                where twitter_movie_score.id = {twitter_list[0][0]}"
            print(query)
            cur.execute(query)
            conn.commit()

        count = count + len(movie_tweets.index)
        query = f"update twitter_movie_score\
            set tweet_count = tweet_count + {len(movie_tweets.index)}\
            where twitter_movie_score.id = {twitter_list[0][0]}"
        print(query)
        cur.execute(query)
        conn.commit()
        print(count)

def get_twitter_cast_raw_data(cur, conn, engine):
    query = "SELECT t1.id, t1.cast_name, t1.tweet_end_id\
        FROM twitter_cast_score t1\
        where t1.tweet_end_id = 0\
        limit 1"
    cur.execute(query)
    twitter_list = cur.fetchall()
    # 
    # print(twitter_list[0][0])
    # input()
    api = get_twitter_data()
    max_id = 2650506631677968385
    current_id = api.get_current_id()
    time.sleep(0.5)
    
    count = 0

    # print(current_id)
    while True:
        print(twitter_list)
        try:
            cast_tweets = api.get_search_data(twitter_list[0][1], max_id = max_id)
        except tweepy.errors.TwitterServerError as e:
            # Handle the error
            print("Twitter server error occurred: ", e)
            continue

        if cast_tweets is None:
            if count == 0:
                query = f"update twitter_cast_score\
                    set tweet_end_id = {current_id}\
                    where twitter_cast_score.id = {twitter_list[0][0]}"
                print(query)
                cur.execute(query)
                conn.commit()
            break

        max_id = cast_tweets.iloc[-1].tweet_id - 1
        cast_tweets['cast_id'] = twitter_list[0][0]
        cast_tweets.to_sql("twitter_cast_raw_data", engine, if_exists="append", index=False)

        if count == 0 and cast_tweets.iloc[0].tweet_id + 1 > twitter_list[0][2]:
            query = f"update twitter_cast_score\
                set tweet_end_id = {cast_tweets.iloc[0].tweet_id + 1}\
                where twitter_cast_score.id = {twitter_list[0][0]}"
            print(query)
            cur.execute(query)
            conn.commit()

        count = count + len(cast_tweets.index)
        query = f"update twitter_cast_score\
            set tweet_count = tweet_count + {len(cast_tweets.index)}\
            where twitter_cast_score.id = {twitter_list[0][0]}"
        print(query)
        cur.execute(query)
        conn.commit()
        print(count)

def get_movie_raw_sentiment(cur, conn, engine):
    query = "SELECT t1.id, t1.text\
            FROM twitter_cast_raw_data t1\
            Where score is NULL\
            limit 32"
    cur.execute(query)
    data = cleaner(cur.fetchall())
    df = pd.DataFrame(data, columns = ['id', 'text'])
    trainer = twitter_sentiment("drive/MyDrive/Colab Notebooks/model/bert_v2")

    df["score"] = trainer.predict(df)
    for i in range(32):
        query = f"update twitter_cast_raw_data\
            set score = {df.iloc[i]['score']}\
            where twitter_cast_raw_data.id = {df.iloc[i]['id']}"
        print(query)
        cur.execute(query)
        conn.commit()

def cleaner(tweets):
    result = []
    for tweet in tweets:
        tmp_tweet = tweet[1].lower()
        tmp_tweet = re.sub("@[A-Za-z0-9]+","",tmp_tweet) #Remove @ sign
        tmp_tweet = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", tmp_tweet) #Remove http links
        tmp_tweet = " ".join(tmp_tweet.split())
        tmp_tweet = tmp_tweet.replace("#", "").replace("_", " ") #Remove hashtag sign but keep the text
        result.append([tweet[0], tmp_tweet])

    return result
    
if __name__ == "__main__":
    cur, conn, engine = connect_sql()
    for i in range(100):
        get_twitter_cast_raw_data(cur, conn, engine)
    # api = get_twitter_data()
    # current_id = api.get_current_id()
    # print(current_id)