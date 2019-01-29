import praw 
import json
import logging
from datetime import datetime

handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
logger = logging.getLogger('prawcore')
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

reddit = praw.Reddit('aedl')

def getredditsubmission(subid):
    submission = reddit.submission(id=subid) 

    submission_json = submissioninfo(submission)
    submission_json['user'] = userinfo(submission.author)
    submission_json['subreddit'] = subredditinfo(submission.subreddit, submission.subreddit_id)
    submission.comment_sort = 'old'
    submission_json['comments'] = commentsinfo(submission.comments)
    return submission_json


def submissioninfo(submission):
    submission_json = {}
    submission_json['title'] = submission.title
    submission_json['text'] = submission.selftext
    submission_json['submission_id'] = submission.id
    submission_json['created'] = convtime(submission.created_utc)
    submission_json['num_comments'] = submission.num_comments
    submission_json['url'] = submission.permalink
    submission_json['upvotes'] = submission.score
    submission_json['is_video'] = submission.is_video
    return submission_json

def userinfo(user):
    user_data = {}
    user_data['id'] = user.id
    user_data['username'] = user.name
    user_data['karma'] = user.comment_karma
    user_data['created'] = convtime(user.created_utc)
    user_data['gold_status'] = user.is_gold
    user_data['is_employee'] = user.is_employee
    user_data['has_verified_email'] = user.has_verified_email
    return user_data

def subredditinfo(subreddit, subreddit_id):
    subreddit_data = {}
    subreddit_data['name'] = subreddit.display_name
    subreddit_data['subreddit_id'] = subreddit_id
    subreddit_data['created'] = convtime(subreddit.created_utc)
    subreddit_data['subscribers'] = subreddit.subscribers
    return subreddit_data

def commentsinfo(comments):
    comments_data = []
    comments.replace_more(limit=None) #all MoreComments objects will be replaced
    for comment in comments.list(): #flatten all nested comments
        comment.refresh()
        data = {}
        data['comment_id'] = comment.id
        data['text'] = comment.body
        data['created'] = convtime(comment.created_utc)
        data['is_submitter'] = comment.is_submitter
        data['submission_id'] = comment.link_id
        data['parent_id'] = comment.parent_id
        data['comment_url'] = comment.permalink
        data['upvotes'] = comment.score
        data['replies'] = comment.replies.__len__()
        data['user'] = userinfo(comment.author)

        comments_data.append(data)
    return comments_data

def convtime(utctime):
    return datetime.utcfromtimestamp(utctime).strftime("%Y-%m-%d %H:%M:%S")

subid = '8cx0da' #'Mener I at der skal være ulve i Dk? Hvorfor/hvorfor ikke?'
submission_json = getredditsubmission(subid)
with open("{0}.json".format(subid), 'w') as outfile:
    json.dump(submission_json, outfile)