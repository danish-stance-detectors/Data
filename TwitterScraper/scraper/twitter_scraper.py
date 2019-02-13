import json
import tweepy
import time

# Loads twitter credentials
twitter_cred = dict()
with open('twitter_credentials.json') as f:
    secret_info = json.load(f)
    twitter_cred["CONSUMER_KEY"] = secret_info["CONSUMER_KEY"]
    twitter_cred["CONSUMER_SECRET"] = secret_info["CONSUMER_SECRET"]
    twitter_cred["ACCESS_KEY"] = secret_info["ACCESS_KEY"]
    twitter_cred["ACCESS_SECRET"] = secret_info["ACCESS_SECRET"]

# sets up tweepy api
auth = tweepy.OAuthHandler(twitter_cred["CONSUMER_KEY"], twitter_cred["CONSUMER_SECRET"])
auth.set_access_token(twitter_cred["ACCESS_KEY"], twitter_cred["ACCESS_SECRET"])
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())

search = "#ulvesagen"

#wait_on_rate_limit=True to get a lot
# applies search term
submarine_tweets = api.search(q=search, count=100, lang="da")

# dump data into file
with open('tweets_' + search + '.json', 'w') as f:
    json.dump(submarine_tweets, f)

# for tweet in submarine_tweets:
#     print(tweet.text)
# count = 0
# while True and count < 10:
#     try:
#         tweet = c.next()
#         count += 1
#         # Insert into db
#         print(tweet)
#     except tweepy.TweepError:
#         time.sleep(60 * 15)
#         continue
#     except StopIteration:
#         break