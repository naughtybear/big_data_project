import psycopg2
import pandas as pd
import sqlalchemy
import numpy as np
from twitter_function import get_twitter_data
from twitter_sentiment import twitter_sentiment
import re
import tweepy
import time
import dask.dataframe as dd

def connect_sql():
    '''
    built connection to postgres by psycopg2 and sqlalchemy
    return:
        cur: psycopg2 cursor
        conn: psycopg2 connection
        engine: sqlalchemy engine
    '''
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

def get_twitter_movie_raw_data(cur, conn, engine, movie_id=None):
    '''
    Get the twitter data from twitter search api and store it to prosgres sql.
    It will return after getting more than 3000 tweets per moive to accelerate the speed to get data from every movie.

    parameters:
        cur: psycopg2 cursor
        conn: psycopg2 connection
        engine: sqlalchemy engine
        movie_id: get the movie data from this movie id. If movie id is None, will get the tweets for those movie have no data.
    return:
        twitter_list[0][0]: the movie_id get the tweets
    '''
    # get the movie data from twitter_movie_score
    if movie_id is None:
        query = "SELECT t1.id, t1.movie_title, t1.release_date, t1.tweet_end_id\
            FROM twitter_movie_score t1\
            where t1.release_date is not NULL and t1.tweet_end_id = 0\
            order by t1.release_date DESC\
            limit 1"
    else:
        query = f"SELECT t1.id, t1.movie_title, t1.release_date, t1.tweet_end_id\
            FROM twitter_movie_score t1\
            where t1.id >= {movie_id}\
            order by t1.id DESC\
            limit 1"

    cur.execute(query)
    twitter_list = cur.fetchall()

    api = get_twitter_data()
    max_id = 2650506631677968385
    current_id = api.get_current_id()
    time.sleep(3)
    count = 0

    while True:
        print(twitter_list)
        if len(twitter_list[0][1]) < 15:
            movie_tweets = api.get_search_data(twitter_list[0][1] + " movie", max_id = max_id)
        else:
            movie_tweets = api.get_search_data(twitter_list[0][1], max_id = max_id)
        if movie_tweets is None:
            if count == 0:
                if current_id is not None:
                    query = f"update twitter_movie_score\
                        set tweet_end_id = {current_id}\
                        where twitter_movie_score.id = {twitter_list[0][0]}"
                else:
                    print("default twitter end id")
                    query = f"update twitter_cast_score\
                        set tweet_end_id = 1654959876240441345\
                        where twitter_cast_score.id = {twitter_list[0][0]}"
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
        if count > 3000:
            break
    
    return twitter_list[0][0]

def get_twitter_cast_raw_data(cur, conn, engine, cast_id=None):
    '''
    Get the twitter data from twitter search api and store it to prosgres sql.
    It will return after getting more than 3000 tweets per cast to accelerate the speed to get data from every cast.

    parameters:
        cur: psycopg2 cursor
        conn: psycopg2 connection
        engine: sqlalchemy engine
        cast_id: get the cast data from this cast id. If cast id is None, will get the tweets for those cast have no data.
    return:
        twitter_list[0][0]: the cast_id that get the tweets
    '''
    if cast_id is None:
        query = "SELECT t1.id, t1.cast_name, t1.tweet_end_id\
            FROM twitter_cast_score t1\
            where t1.tweet_end_id = 0\
            limit 1"
    else:
        query = f"SELECT t1.id, t1.cast_name, t1.tweet_end_id\
            FROM twitter_cast_score t1\
            where t1.id >= {cast_id}\
            order by t1.id ASC\
            limit 1"

    cur.execute(query)
    twitter_list = cur.fetchall()

    api = get_twitter_data()
    max_id = 2650506631677968385
    current_id = api.get_current_id()
    time.sleep(3)
    
    count = 0

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
                if current_id is not None:
                    query = f"update twitter_cast_score\
                        set tweet_end_id = {current_id}\
                        where twitter_cast_score.id = {twitter_list[0][0]}"
                else:
                    print("default twitter end id")
                    query = f"update twitter_cast_score\
                        set tweet_end_id = 1654959876240441345\
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
        if count > 3000:
            break
    return twitter_list[0][0]

def get_twitter_director_raw_data(cur, conn, engine, director_id=None):
    '''
    Get the twitter data from twitter search api and store it to prosgres sql.
    It will return after getting more than 3000 tweets per director to accelerate the speed to get data from every movie.

    parameters:
        cur: psycopg2 cursor
        conn: psycopg2 connection
        engine: sqlalchemy engine
        director_id: get the director data from this director id. If director id is None, will get the tweets for those director have no data.
    return:
        twitter_list[0][0]: the director_id get the tweets
    '''

    if director_id is None:
        query = "SELECT t1.id, t1.director_name, t1.tweet_end_id\
            FROM twitter_director_score t1\
            where t1.tweet_end_id = 0\
            limit 1"
    else:
        query = f"SELECT t1.id, t1.director_name, t1.tweet_end_id\
            FROM twitter_director_score t1\
            where t1.id >= {director_id}\
            order by t1.id ASC\
            limit 1"
    cur.execute(query)
    twitter_list = cur.fetchall()
    
    api = get_twitter_data()
    max_id = 2650506631677968385
    current_id = api.get_current_id()
    time.sleep(3)
    
    count = 0

    # print(current_id)
    while True:
        print(twitter_list)
        try:
            director_tweets = api.get_search_data(twitter_list[0][1], max_id = max_id)
        except tweepy.errors.TwitterServerError as e:
            # Handle the error
            print("Twitter server error occurred: ", e)
            continue

        if director_tweets is None:
            if count == 0:
                if current_id is not None:
                    query = f"update twitter_director_score\
                        set tweet_end_id = {current_id}\
                        where twitter_director_score.id = {twitter_list[0][0]}"
                else:
                    print("default twitter end id")
                    query = f"update twitter_director_score\
                        set tweet_end_id = 1654959876240441345\
                        where twitter_director_score.id = {twitter_list[0][0]}"
                print(query)
                cur.execute(query)
                conn.commit()
            break

        max_id = director_tweets.iloc[-1].tweet_id - 1
        director_tweets['director_id'] = twitter_list[0][0]
        director_tweets.to_sql("twitter_director_raw_data", engine, if_exists="append", index=False)

        if count == 0 and director_tweets.iloc[0].tweet_id + 1 > twitter_list[0][2]:
            query = f"update twitter_director_score\
                set tweet_end_id = {director_tweets.iloc[0].tweet_id + 1}\
                where twitter_director_score.id = {twitter_list[0][0]}"
            print(query)
            cur.execute(query)
            conn.commit()

        count = count + len(director_tweets.index)
        query = f"update twitter_director_score\
            set tweet_count = tweet_count + {len(director_tweets.index)}\
            where twitter_director_score.id = {twitter_list[0][0]}"
        print(query)
        cur.execute(query)
        conn.commit()
        print(count)
        if count > 3000:
            break
    return twitter_list[0][0]

def get_cast_raw_sentiment(cur, conn, trainer):
    '''
    get tweets from twitter_cast_raw_data and do the sentiment prediction of the tweets.
    update the result of the prediction to twitter_cast_raw_data
    '''
    query = "SELECT t1.id, t1.text\
            FROM twitter_cast_raw_data t1\
            Where score is NULL\
            limit 1024"
    cur.execute(query)
    data = cur.fetchall()
    df = pd.DataFrame(data, columns = ['id', 'text'])
    ddf = dd.from_pandas(df, npartitions=10)

    # Apply the cleaner function to the 'tweets' column of the Dask dataframe
    ddf['text'] = ddf['text'].apply(cleaner, meta=('text', 'object'))

    # Convert the Dask dataframe back to a Pandas dataframe
    df = ddf.compute()
    print(data[0][0])

    df["score"] = trainer.predict(df)
    df.to_sql('temp_cast_table', engine, if_exists='replace')
    query = f"update twitter_cast_raw_data\
        set score = temp_cast_table.score\
        from temp_cast_table\
        where twitter_cast_raw_data.id = temp_cast_table.id"
    # print(query)
    cur.execute(query)
    conn.commit()
    return len(data)

def get_movie_raw_sentiment(cur, conn, trainer):
    '''
    get tweets from twitter_movie_raw_data and do the sentiment prediction of the tweets.
    update the result of the prediction to twitter_movie_raw_data
    '''
    query = "SELECT t1.id, t1.text\
            FROM twitter_movie_raw_data t1\
            Where score is NULL\
            limit 1024"
    cur.execute(query)
    data = cur.fetchall()
    df = pd.DataFrame(data, columns = ['id', 'text'])
    ddf = dd.from_pandas(df, npartitions=10)

    # Apply the cleaner function to the 'tweets' column of the Dask dataframe
    ddf['text'] = ddf['text'].apply(cleaner, meta=('text', 'object'))

    # Convert the Dask dataframe back to a Pandas dataframe
    df = ddf.compute()
    print(data[0][0])

    df["score"] = trainer.predict(df)
    df.to_sql('temp_movie_table', engine, if_exists='replace')
    query = f"update twitter_movie_raw_data\
        set score = temp_movie_table.score\
        from temp_movie_table\
        where twitter_movie_raw_data.id = temp_movie_table.id"
    # print(query)
    cur.execute(query)
    conn.commit()
    return len(data)

def get_director_raw_sentiment(cur, conn, model):
    '''
    get tweets from twitter_director_raw_data and do the sentiment prediction of the tweets.
    update the result of the prediction to twitter_director_raw_data
    '''
    query = "SELECT t1.id, t1.text\
            FROM twitter_director_raw_data t1\
            Where score is NULL\
            ORDER BY id ASC\
            limit 1024"
    cur.execute(query)
    data = cur.fetchall()
    df = pd.DataFrame(data, columns = ['id', 'text'])
    ddf = dd.from_pandas(df, npartitions=10)

    # Apply the cleaner function to the 'tweets' column of the Dask dataframe
    ddf['text'] = ddf['text'].apply(cleaner, meta=('text', 'object'))

    # Convert the Dask dataframe back to a Pandas dataframe
    df = ddf.compute()
    print(data[0][0])

    df["score"] = model.predict(df)
    df.to_sql('temp_director_table', engine, if_exists='replace')
    # for i in range(1024):
    query = f"update twitter_director_raw_data\
        set score = temp_director_table.score\
        from temp_director_table\
        where twitter_director_raw_data.id = temp_director_table.id"
    # print(query)
    cur.execute(query)
    conn.commit()

    return len(data)

def cleaner(tmp_tweet):
    tmp_tweet = tmp_tweet.lower()
    tmp_tweet = re.sub("@[A-Za-z0-9]+","",tmp_tweet) #Remove @ sign
    tmp_tweet = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", tmp_tweet) #Remove http links
    tmp_tweet = " ".join(tmp_tweet.split())
    tmp_tweet = tmp_tweet.replace("#", "").replace("_", " ") #Remove hashtag sign but keep the text
    return tmp_tweet

def get_sentiment_model(path = "model/bert_v2"):
    '''
    initialization of the sentiment model
    '''
    return twitter_sentiment(path)

def calculate_cast_score(cur, conn):
    query = "with cast_average_score as (\
                select t1.cast_id, avg(t1.score) - 1 as avg_score\
                from twitter_cast_raw_data as t1\
                where t1.score is not null\
                group by t1.cast_id\
            )\
            \
            update twitter_cast_score\
            set score = cast_average_score.avg_score\
            from cast_average_score\
            where id = cast_average_score.cast_id"
    cur.execute(query)
    conn.commit()

def calculate_director_score(cur, conn):
    query = "with director_average_score as (\
                select t1.director_id, avg(t1.score) - 1 as avg_score\
                from twitter_director_raw_data as t1\
                where t1.score is not null\
                group by t1.director_id\
            )\
            \
            update twitter_director_score\
            set score = director_average_score.avg_score\
            from director_average_score\
            where id = director_average_score.director_id"
    cur.execute(query)
    conn.commit()

def calculate_movie_score(cur, conn):
    query = "with movie_average_score as (\
                select t1.movie_id, avg(t1.score) - 1 as avg_score\
                from twitter_movie_raw_data as t1\
                where t1.score is not null\
                group by t1.movie_id\
            )\
            \
            update twitter_movie_score\
            set score = movie_average_score.avg_score\
            from movie_average_score\
            where id = movie_average_score.movie_id"
    cur.execute(query)
    conn.commit()

    
if __name__ == "__main__":
    cur, conn, engine = connect_sql()

    # initial director twitter with twitter api
    # for i in range(10):
    #     get_twitter_director_raw_data(cur, conn, engine)
    
    # update director twitter with twitter api
    current_id = 0
    for i in range(10):
        current_id = get_twitter_director_raw_data(cur, conn, engine, current_id)
        current_id = current_id + 1

    # sentiment prediction for each tweet about directors
    model = get_sentiment_model()
    for i in range(10):
        length = get_director_raw_sentiment(cur, conn, model)
        if(length < 1024):
            break

    # calculate the score of directors
    calculate_director_score(cur, conn)

    # # initial cast twitter with twitter api
    # for i in range(10):
    #     get_twitter_cast_raw_data(cur, conn, engine)
    
    # # update cast twitter with twitter api
    # current_id = 0
    # for i in range(10):
    #     current_id = get_twitter_cast_raw_data(cur, conn, engine, current_id)
    #     current_id = current_id + 1

    # # sentiment prediction for each tweet about casts
    # model = get_sentiment_model()
    # for i in range(10):
    #     length = get_cast_raw_sentiment(cur, conn, model)
    #     if(length < 1024):
    #         break

    # # calculate the score of movies
    # calculate_movie_score(cur, conn)

    # # initial movie twitter with twitter api
    # for i in range(10):
    #     get_twitter_movie_raw_data(cur, conn, engine)
    
    # # update movie twitter with twitter api
    # current_id = 0
    # for i in range(10):
    #     current_id = get_twitter_movie_raw_data(cur, conn, engine, current_id)
    #     current_id = current_id + 1

    # # sentiment prediction for each tweet about movies
    # model = get_sentiment_model()
    # for i in range(10):
    #     length = get_movie_raw_sentiment(cur, conn, model)
    #     if(length < 1024):
    #         break

    # # calculate the score of movies
    # calculate_movie_score(cur, conn)