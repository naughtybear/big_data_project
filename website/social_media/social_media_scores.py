
from social_media.reddit_utils import get_reddit_sentiment_ratio


def get_sentiment_scores(movie_name, casts, directors):

    twitter_movie_name_score = get_twitter_sentiment_ratio(movie_name)
    reddit_movie_name_score = get_reddit_sentiment_ratio(movie_name)

    twitter_cast_score = 0
    reddit_cast_score = 0
    for cast in casts:
        twitter_cast_score += get_twitter_sentiment_ratio(cast)
        reddit_cast_score += get_reddit_sentiment_ratio(cast)
    twitter_cast_score /= len(casts)
    reddit_cast_score /= len(casts)

    twitter_director_score = 0
    reddit_director_score = 0
    for director in directors:
        twitter_director_score += get_twitter_sentiment_ratio(director)
        reddit_director_score += get_reddit_sentiment_ratio(director)
    twitter_director_score /= len(directors)
    reddit_director_score /= len(directors)

    twitter_setiment_score = [twitter_movie_name_score, twitter_cast_score, twitter_director_score]
    reddit_setiment_score = [reddit_movie_name_score, reddit_cast_score, reddit_director_score]

    return twitter_setiment_score, reddit_setiment_score

