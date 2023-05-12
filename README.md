# Box Office Predictor: Leveraging Twitter/Reddit and Movie Data

- This project is aimed at building a machine learning model that can predict the box office performance of movies based on movies features and their social media buzz. 

- The project utilizes data from Twitter and Reddit to gather sentiment analysis, and various movie attributes such as cast, director, title,and runtime to build a predictive model.

- The webisite will be created locally to deploy the model and to display the visualizations for more insights.
## Project Group Members
Hongjie Huang, Po Yen Chen, Yizheng Wang, Rahul Raj, Pulkit Khandelwal

## ETL Diagram
![image](/photo/box_office_etl.png)

## Parts of the Project

The project consists of the following parts:

### 1. Data Collection and Cleaning

- This part of the project involves the collection and cleanning of data from kaggle movies datasets and TMDB,IMDB APIs. 
- The collected data will be cleaned, transformed, and standardized and be saved into the table in database for further processing.
- Gather essential movie details, such as casts, director, genres, budgets, ratings, etc.
- Retrieve new movie data from the TMDB dataset API in a period of time, and fill out missing data using IMDB API.

### 2. Reddit
- This part of the project involves collecting the latest comments related to a movie title, director or cast from Reddit.
  * Reddit Casts Sentiment: This part of the project analyzes the sentiment of the comments on Reddit related to the individual cast.
  * Reddit Director Sentiment: This part of the project analyzes the sentiment of the comments on Reddit related to the director name.
  * Reddit Movie Title Sentiment: This part of the project analyzes the sentiment of the comments on Reddit related to the title of a particular movie.

### 3. Twitter
  * Twitter Function: This part of the project involves the creation of a function that gathers data from Twitter using its API.
  * Twitter Sentiment: This is the class that handling the tweets sentiment prediction. Need to download the bert_v2 folder from [here](https://drive.google.com/drive/folders/1C_b4g5G_eSZzA97pGpyvhM0x6AxLExI2?usp=sharing), and put it in the model folder.
  * Twitter to SQL: This part of the project involves storing the collected Twitter data into a SQL database.

### 4. Cloud SQL Connection

This part of the project involves the example of connection to the SQL database in the cloud.

### 5. Combine Data

This part of the project involves the combination of the data collected from Reddit and Twitter and movies datasets.

### 6. Model

- This part of the project involves the analysis of features, creation of visualizations and several machine learning models that can predict the box office performance of a movie based on the combined data.
- The best performing model will be choosed and used in the website.

### 7. Website
- The website is built using Streamlit because it facilitates easy and rapid web development enabling developers to focus on the data and machine learning aspects.
- Required Model Inputs: Movie Name, Director, Casts, Budget, Released Date, Genres, Runtime
- Output: Estimated Revenue
- Data Exploration: Exploration page presents dynamic data visualizations intended for data analytics.
<img width="1456" alt="Screenshot 2023-05-11 at 10 11 28 PM" src="https://github.com/naughtybear/big_data_project/assets/30201131/415563f9-7a25-4238-a9a7-7731d14d41db">
<img width="1470" alt="Screenshot 2023-05-11 at 10 12 29 PM" src="https://github.com/naughtybear/big_data_project/assets/30201131/ea720045-89a7-4e2e-a351-16dbc729717e">

### 8. Airflow
Using Airflow to monitor pipelines. Here is the ETL pipeline

![image](https://github.com/naughtybear/big_data_project/assets/25876670/e5aa5ec8-59da-4d70-aea0-b0322ac1c940)

## Usage

To use this project, please follow the instructions below:

1. Clone or download the repository to your local machine.
2. Install the required dependencies.
3. Follow the instructions in each file to run the scripts in the correct order.
4. To run the data_collection_cleaning file, make sure you have download the files from movie_data folder.
5. Once the data is collected, cleaned, and combined, run the model script to build and train the predictive model.
5. For running inference and checking out visualizations, you can run the website using the following commands in the terminal:
```
pip install streamlit
cd website
streamlit run home.py
```
Head over to http://localhost:8501 if not automatically redirected.
## Conclusion

The Box Office Predictor using Twitter and Reddit and Movie Data is a useful tool for predicting the box office performance of movies. By leveraging sentiment analysis on social media and various movie attributes, this project provides some valuable insights into a movie's potential success.
