import json
import tweepy
import time

# Loads twitter credentials
twitter_cred = dict()
with open('twitter_archive_credentials.json') as f:
    secret_info = json.load(f)
    twitter_cred["CONSUMER_KEY"] = secret_info["CONSUMER_KEY"]
    twitter_cred["CONSUMER_SECRET"] = secret_info["CONSUMER_SECRET"]
    twitter_cred["ACCESS_KEY"] = secret_info["ACCESS_KEY"]
    twitter_cred["ACCESS_SECRET"] = secret_info["ACCESS_SECRET"]

# sets up tweepy api
auth = tweepy.OAuthHandler(twitter_cred["CONSUMER_KEY"], twitter_cred["CONSUMER_SECRET"])
auth.set_access_token(twitter_cred["ACCESS_KEY"], twitter_cred["ACCESS_SECRET"])
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser(), wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

search = "#kimwall"

# look into this: https://stackoverflow.com/questions/22469713/managing-tweepy-api-search

# #wait_on_rate_limit=True to get a lot
# # applies search term
submarine_tweets = api.search(q=search, rpp=100, lang="da")

print(submarine_tweets)

# # dump data into file
# with open('./data/tweets_' + search + '2' + '.json', 'w') as f:
#     json.dump(submarine_tweets, f)

# replies=[] 
# non_bmp_map = dict()

# tweet_id = 990676636071874565
# user_name = 'BjarkeCharlie'

# for full_tweets in tweepy.Cursor(api.user_timeline,screen_name=user_name,timeout=999999).items(10):
#   for tweet in tweepy.Cursor(api.search,q='to:'+user_name, since_id=tweet_id,timeout=999999).items(1000):
#     if hasattr(tweet, 'in_reply_to_status_id_str'):
#       if (tweet.in_reply_to_status_id_str==full_tweets.id_str):
#         replies.append(tweet.text)
#   print("Tweet :",full_tweets.text.translate(non_bmp_map))
#   for elements in replies:
#        print("Replies :",elements)

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