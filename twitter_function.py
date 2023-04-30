import tweepy
import json
import pandas as pd


class get_twitter_data():
    def __init__(self, credential_file="twitter_api_key.json"):
        with open(credential_file, "r") as file_content:
            cred = json.load(file_content)
            self.access_token = cred["Access Token"]
            self.access_secret = cred["Access Token Secret"]
            self.bearer_token = cred["Bearer Token"]
            self.consumer_key = cred["API Key"]
            self.consumer_secret= cred["API Key Secret"]

        auth = tweepy.OAuthHandler(self.consumer_key, self.consumer_secret)
        auth.set_access_token(self.access_token, self.access_secret)
        self.api = tweepy.API(auth)

    def get_follower(self, usernames):
        # get followers of usernames(multiple users)
        user_info = self.api.get_user(screen_name=usernames)
        
        return user_info.followers_count
    
    def get_search_data(self, query, since_id=None):
        # get search data within seven days
        tweet_list = []
        max_id = 2650506631677968385

        while True:
            # Use the user_timeline() method to retrieve the user's tweets
            if since_id:
                replies = self.api.search_tweets(q=query, count=200, max_id=max_id, since_id = since_id)
            else:
                replies = self.api.search_tweets(q=query, count=200, max_id=max_id)
            if len(replies) == 0:
                break

            # Iterate through the replies and print the text of each reply
            for tweet in replies:
                tweet_list.append([tweet.id, tweet.created_at, tweet.text, tweet.favorite_count, tweet.retweet_count])

            max_id = int(tweet_list[-1][0]) - 1

        return tweet_list

    def get_account_tweets(self, username):
        # get all tweet post by specific account
        tweet_list = []
        max_id = 2650506631677968385

        while True:
            # Use the user_timeline() method to retrieve the user's tweets
            # year before 2014 contain replies and tags post
            tweets = self.api.user_timeline(screen_name=username, count=200, max_id=max_id)
            if len(tweets) == 0:
                break

            for tweet in tweets:
                if tweet.in_reply_to_status_id:
                    tweet_type = "Reply"
                elif tweet.entities['user_mentions']:
                    tweet_type = "Tagged"
                else:
                    tweet_type = "Post"
                tweet_list.append([tweet.id, tweet.created_at, tweet.text, tweet.favorite_count, tweet.retweet_count, tweet_type])
            
            max_id = int(tweet_list[-1][0]) - 1

        return pd.DataFrame(tweet_list, columns=["Id", "CreateDate", "Text", "Likes", "Retweet", "Type"])
    

if __name__ == "__main__":
    api = get_twitter_data()
    print(api.get_follower("TomCruise"))
    print(api.get_account_tweets("TomCruise"))