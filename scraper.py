import praw

reddit = praw.Reddit('aedl')
subreddit = reddit.subreddit('Denmark')

for submission in reddit.subreddit('Denmark').hot(limit=10):
    print(submission.title)