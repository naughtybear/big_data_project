from social_media_utils import SentimentScore


def get_sentiment_scores(movie_name, casts, directors):
    scores = SentimentScore()
    twitter_movie_score, reddit_movie_score = scores.find_movie_scores(movie_name)

    twitter_cast_score = 0.0
    reddit_cast_score = 0.0
    casts = casts.split(",")
    for cast in casts:
        cast = cast.strip()
        t, r = scores.find_cast_scores(cast)
        twitter_cast_score += t
        reddit_cast_score += r
    twitter_cast_score /= len(casts)
    reddit_cast_score /= len(casts)

    twitter_director_score = 0.0
    reddit_director_score = 0.0
    directors = directors.split(",")
    for director in directors:
        director = director.strip()
        t, r = scores.find_director_scores(director)
        twitter_director_score += t
        reddit_director_score += r
    twitter_director_score /= len(directors)
    reddit_director_score /= len(directors)

    return [twitter_movie_score, twitter_cast_score, twitter_director_score], [reddit_movie_score, reddit_cast_score, reddit_director_score]

# if __name__ == "__main__":
#     twitter_scores, reddit_scores = get_sentiment_scores("The Dark Knight", "Christian Bale, Heath Ledger, Aaron Eckhart, Michael Caine, Maggie Gyllenhaal, Gary Oldman, Morgan Freeman", "Christopher Nolan")
