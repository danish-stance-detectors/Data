import praw 
import prawcore.exceptions
import json
import logging
from datetime import datetime
import time
import os
import getopt
import sys

def enablelogging():
    """Enable logging. Will print HTTP calls to terminal."""
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    logger = logging.getLogger('prawcore')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

def getredditsubmission(reddit, subid):
    """Retrieve information in JSON about a Submission, inlcuding its
    author, Subreddit, and all comments with corresponding user unfo"""
    submission = reddit.submission(id=subid) 

    sub_data = submissioninfo(submission)
    sub_data['user'] = userinfo(submission.author)
    sub_data['subreddit'] = subredditinfo(submission.subreddit, submission.subreddit_id)
    submission.comment_sort = 'old'
    sub_data['comments'] = commentsinfo(submission.comments)
    return sub_data


def submissioninfo(submission):
    """Retrieve essential data for a Submission"""
    sub_data = {}
    sub_data['title'] = submission.title
    sub_data['text'] = submission.selftext
    sub_data['submission_id'] = submission.id
    sub_data['created'] = convtime(submission.created_utc)
    sub_data['num_comments'] = submission.num_comments
    sub_data['url'] = submission.permalink
    sub_data['upvotes'] = submission.score
    sub_data['is_video'] = submission.is_video
    return sub_data

def userinfo(user):
    """Retrieve relevant information for a Redditor"""
    if ( user is None ):
        return {}
    user_data = {}
    try:
        user_data['id'] = user.id
        user_data['username'] = user.name
        user_data['karma'] = user.comment_karma
        user_data['created'] = convtime(user.created_utc)
        user_data['gold_status'] = user.is_gold
        user_data['is_employee'] = user.is_employee
        user_data['has_verified_email'] = user.has_verified_email
    except prawcore.exceptions.NotFound:
        return {}
    return user_data

def subredditinfo(subreddit, subreddit_id):
    """Retrieve essential data for a Subreddit"""
    subreddit_data = {}
    subreddit_data['name'] = subreddit.display_name
    subreddit_data['subreddit_id'] = subreddit_id
    subreddit_data['created'] = convtime(subreddit.created_utc)
    subreddit_data['subscribers'] = subreddit.subscribers
    return subreddit_data

def commentsinfo(comments):
    comments_data = []
    while True:
        try:
            #all MoreComments objects will be replaced
            #May cause many API calls, and thus exceptions
            #Keep trying until all are replaced
            comments.replace_more(limit=None)
            break
        except Exception:
            print('Handling replace_more exception')
            time.sleep(1)
    for comment in comments.list(): #flatten all nested comments
        data = {}
        data['comment_id'] = comment.id
        data['text'] = comment.body
        is_deleted = False
        if ( data['text'] == '[deleted]'):
            is_deleted = True
        data['is_deleted'] = is_deleted
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
    """Convert POSIX time to YYYY-MM-DD HH:MM:SS"""
    return datetime.utcfromtimestamp(utctime).strftime("%Y-%m-%d %H:%M:%S")

def process_submissions(reddit, datafolder, submission_ids):
    """Go through submission_ids and process information
    from reddit for each Submission and output in JSON into
    respective folders in the datafolder, depending on their topic"""
    with open(submission_ids, 'r') as subids:
        for line in subids.readlines()[1:]: #skip header
            vals = line.split(',')
            subid = vals[0].strip()
            topic = vals[1].strip()
            topicfolder = os.path.join(datafolder, topic)
            if not os.path.exists(topicfolder):
                os.makedirs(topicfolder)
            else:
                existing_submissions = os.listdir(topicfolder)
                if "{0}.json".format(subid) in existing_submissions:
                    print("Skip", subid)
                    continue
            print("Process", subid)
            sub_data = getredditsubmission(reddit, subid)
            path = os.path.join(topicfolder, "{0}.json".format(subid))
            with open(path, 'w', encoding="utf-8") as outfile:
                json.dump(sub_data, outfile, ensure_ascii=False)


def main(argv):
    reddit = None
    try:
        opts, _ = getopt.getopt(argv, "u:h:l", ["user=","help","log"])
    except getopt.GetoptError:
        print("see: scraper.py -help")
        sys.exit(2)
    for opt, val in opts:
        if opt in ("-l", "-log"):
            enablelogging()
        elif opt in ("-u", "-user"):
            reddit = praw.Reddit(val)
        elif opt in ("-h", "-help"):
            print("run 'scraper.py -u' or 'scraper.py -user' with valid praw agent from praw.ini")
            print("run with -l or -log to enable logging of API calls")
            sys.exit()

    datafolder = "submissions/"
    submission_ids = "submission_ids.csv"

    if reddit:
        process_submissions(reddit, datafolder, submission_ids)    

if __name__ == "__main__":
    main(sys.argv[1:])