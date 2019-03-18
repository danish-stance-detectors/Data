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
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
#, parser=tweepy.parsers.JSONParser()
search = "valg"

# look into this: https://stackoverflow.com/questions/22469713/managing-tweepy-api-search

# #wait_on_rate_limit=True to get a lot
# # applies search term
# max_tweets = 100
# for item in api.search(q=search).items():
#   print(item)

def getRepliesTo(tweet_id, user_name):
  # for result in api.search(q='@' + user_name, count=1000):
  for result in api.user_timeline(screen_name=user_name, count=1000):
    #if result._json['in_reply_to_status_id'] == tweet_id:
    # print(type(result)) 
    print(result._json['created_at'].encode('utf-8')) 
    # for status in result[1]:
    #   # print(status[0])
    #   # print(str(status).encode('utf-8'))
    #   # tweet_json = json.loads(str(status).encode('utf-8'))
    #   tweet_str = str(status).encode('utf-8')
    #   print(tweet_str)
      # if status['in_reply_to_status_id'] == tweet_id:
      #   print(status)

getRepliesTo(916579969064632320, 'christianevejlo')
# searched_tweets = [status for status in tweepy.Cursor(api.search, q=search).items(max_tweets)]
# print(searched_tweets)

# submarine_tweets = api.search(q=search, rpp=100, lang="da")

# print(submarine_tweets)

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