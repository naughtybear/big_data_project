from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
import sys
import os

from airflow.providers.papermill.operators.papermill import PapermillOperator

project_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_directory)

from plugins.twitter.twitter_to_sql import func_get_twitter_director_raw_data, \
    func_get_director_raw_sentiment, func_calculate_director_score, preprocess_data

# Modular programming
def twitter():
    print("twitter is ready")

def reddit():
    print("reddit is ready")

default_args = {
    'owner': 'teamrocket',
    'retry': 5,
    'retry_delay': timedelta(minutes=5)
}


# Scheduling the DAG to run daily
with DAG(
    default_args=default_args,
    dag_id="twitter_dag",
    start_date=datetime(2023, 5, 11),
    schedule_interval='@daily'
) as dag:
    twitter_ready = PythonOperator(
        task_id='Twitter',
        python_callable=twitter
    )
    get_twitter_director_raw_data = PythonOperator(
        task_id='get_twitter_director_raw_data',
        python_callable=func_get_twitter_director_raw_data
    )
    
    get_twitter_preprocess_data = PythonOperator(
        task_id='text_preprocessing_and_feature_extraction_twitter',
        python_callable=preprocess_data
    )

    get_director_raw_sentiment = PythonOperator(
        task_id='get_director_raw_sentiment',
        python_callable=func_get_director_raw_sentiment
    )
    
    calculate_director_score = PythonOperator(
        task_id='calculate_director_score',
        python_callable=func_calculate_director_score
    )

    reddit_ready = PythonOperator(
        task_id='Reddit',
        python_callable=reddit
    )

    get_reddit_preprocess_data = PythonOperator(
        task_id='text_preprocessing_and_feature_extraction_reddit',
        python_callable=preprocess_data
    )

    reddit_director_sentiment = PapermillOperator(
        task_id="reddit_director_sentiment",
        input_nb="plugins/reddit/reddit_director_sentiment.ipynb",
        output_nb="plugins/reddit/output.ipynb"
    )

    reddit_casts_sentiment = PapermillOperator(
        task_id="reddit_casts_sentiment",
        input_nb="plugins/reddit/reddit_casts_sentiment.ipynb",
        output_nb="plugins/reddit/output.ipynb"
    )

    reddit_movie_title_sentiment = PapermillOperator(
        task_id="reddit_movie_title_sentiment",
        input_nb="plugins/reddit/reddit_movie_title_sentiment.ipynb",
        output_nb="plugins/reddit/output.ipynb"
    )

    tmdb_api = PythonOperator(
        task_id='TMBD_API',
        python_callable=preprocess_data
    )

    imdb_api = PythonOperator(
        task_id='IMDB_API',
        python_callable=preprocess_data
    )

    kaggle_dataset = PythonOperator(
    task_id='Kaggle_dataset',
    python_callable=preprocess_data
    )

    handle_missing_data = PythonOperator(
        task_id='handle_missing_data',
        python_callable=preprocess_data
    ) 

    remove_duplicate_data = PythonOperator(
        task_id='remove_duplicate_data',
        python_callable=preprocess_data
    ) 

    standarization_and_normalization = PythonOperator(
        task_id='standarization_and_normalization',
        python_callable=preprocess_data
    )

    store_api_data_in_database = PythonOperator(
        task_id='store_api_data_in_database',
        python_callable=preprocess_data
    )

    combine_data = PapermillOperator(
        task_id="combine_data",
        input_nb="plugins/combine_data.ipynb",
        output_nb="plugins/reddit/output.ipynb"
    )

    data_collection_cleaning = PapermillOperator(
        task_id="data_collection_cleaning",
        input_nb="plugins/data_collection_cleaning.ipynb",
        output_nb="plugins/reddit/tmp/out-{{ execution_date }}.ipynb",    
    )

    model = PapermillOperator(
        task_id="model",
        input_nb="plugins/model.ipynb",
        output_nb="plugins/reddit/output.ipynb"
    )

    [tmdb_api, imdb_api, kaggle_dataset] >> handle_missing_data >> remove_duplicate_data >> standarization_and_normalization >> store_api_data_in_database >> combine_data
    reddit_ready >> get_reddit_preprocess_data >> reddit_director_sentiment >> reddit_casts_sentiment >> reddit_movie_title_sentiment >> combine_data >> data_collection_cleaning >> model
    twitter_ready >> get_twitter_director_raw_data >> get_twitter_preprocess_data >> get_director_raw_sentiment >> calculate_director_score >> combine_data >> data_collection_cleaning >> model
