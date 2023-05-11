# Box Office Predictor: Leveraging Twitter/Reddit and Movie Data

- This project is aimed at building a machine learning model that can predict the box office performance of movies based on movies features and their social media buzz. 

- The project utilizes data from Twitter and Reddit to gather sentiment analysis, and various movie attributes such as cast, director, title,and runtime to build a predictive model.

- The webisite will be created locally to deploy the model and to display the visualizations for more insights.

## Parts of the Project

The project consists of the following parts:

### 1. Data Collection and Cleaning

- This part of the project involves the collection of data from movies datasets,TMDB,IMDB, using their respective APIs. The collected data is then cleaned to remove irrelevant information and prepare it for further processing.
- Gather essential movie details, such as casts, director, genres, budgets, ratings, etc.
- Retrieve new movie data from the TMDB dataset API in a period of time.

### 2. Reddit Casts Sentiment

This part of the project analyzes the sentiment of the comments on Reddit related to the cast of a particular movie.

### 3. Reddit Director Sentiment

This part of the project analyzes the sentiment of the comments on Reddit related to the director of a particular movie.

### 4. Reddit Movie Title Sentiment

This part of the project analyzes the sentiment of the comments on Reddit related to the title of a particular movie.

### 5. Twitter Function

This part of the project involves the creation of a function that gathers data from Twitter using its API.

### 6. Twitter Sentiment

This part of the project analyzes the sentiment of the tweets related to a particular movie,cast and director.

### 7. Twitter to SQL

This part of the project involves storing the collected Twitter data into a SQL database.

### 8. Cloud SQL Connection

This part of the project involves the example of connection to the SQL database in the cloud.

### 9. Combine Data

This part of the project involves the combination of the data collected from Reddit and Twitter and movies datasets.

### 10. Model

- This part of the project involves the analysis of features, creation of visualizations and several machine learning models that can predict the box office performance of a movie based on the combined data.
- The best performing model will be choosed and used in the website.

### 11. Website
- The website is built using Streamlit because it facilitates easy and rapid web development enabling developers to focus on the data and machine learning aspects.
- Required Model Inputs: Movie Name, Director, Casts, Budget, Released Date, Genres, Runtime
- Output: Estimated Revenue
- Data Exploration: Exploration page presents dynamic data visualizations intended for data analytics.

### 12. Airflow
- Todo: 

## Usage

To use this project, please follow the instructions below:

1. Clone or download the repository to your local machine.
2. Install the required dependencies.
3. Follow the instructions in each file to run the scripts in the correct order.
4. To run the data_collection_cleaning file, make sure you have download the files from movie_data folder.
5. Once the data is collected, cleaned, and combined, run the model script to build and train the predictive model.
5. 

## Conclusion

The Box Office Predictor using Twitter and Reddit and Movie Data is a useful tool for predicting the box office performance of movies. By leveraging sentiment analysis on social media and various movie attributes, this project provides some valuable insights into a movie's potential success.
