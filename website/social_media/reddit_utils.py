import praw
import re
from textblob import TextBlob


def authenticate_reddit():
    user_agent = "testscript by u/azerw1"
    reddit = praw.Reddit(
        client_id="XqjIOXMlWjaBJDv3TKWU1Q",
        client_secret="qV-nr6S76j4mSl0mEuQFz-08wktYtg",
        username='azerw1',
        password='$#2hz35EdD=g5xP',
        user_agent=user_agent
    )
    return reddit


def clean_txt(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'RT[\s]+', '', text)
    text = re.sub(r'https?:\/\/\S+', '', text)
    text = re.sub(r':', '', text)
    return text


def remove_emoji(string):
    emoji_pattern = re.compile("[" u"\U0001F600-\U0001F64F" u"\U0001F300-\U0001F5FF" u"\U0001F680-\U0001F6FF" u"\U0001F1E0-\U0001F1FF" u"\U00002500-\U00002BEF" u"\U00002702-\U000027B0" u"\U00002702-\U000027B0" u"\U000024C2-\U0001F251" u"\U0001f926-\U0001f937" u"\U00010000-\U0010ffff" u"\u2640-\u2642" u"\u2600-\u2B55" u"\u200d" u"\u23cf" u"\u23e9" u"\u231a" u"\ufe0f" u"\u3030" "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)


def get_polarity(text):
    return TextBlob(text).sentiment.polarity


def get_insight(score):
    if score < 0:
        return "Negative"
    elif score == 0:
        return "Neutral"
    else:
        return "Positive"


def get_reddit_sentiment_ratio(movie_title):
    reddit = authenticate_reddit()
    subreddit = reddit.subreddit("movies")
    sentiments = []

    for submission in subreddit.search(movie_title, syntax='lucene', sort='relevance', limit=10):
        sentiments.append(get_insight(get_polarity(
            clean_txt(remove_emoji(submission.title)))))
        submission.comments.replace_more(limit=0)
        for top_level_comment in submission.comments:
            sentiments.append(get_insight(get_polarity(
                clean_txt(remove_emoji(str(top_level_comment.body))))))

    likes = sentiments.count("Positive")
    dislikes = sentiments.count("Negative")

    if (likes + dislikes) != 0:
        return (likes - dislikes) / (likes + dislikes)
    else:
        return 0
