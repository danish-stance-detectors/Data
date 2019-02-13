import praw 
import prawcore.exceptions
from psaw import PushshiftAPI
import json
import logging
from datetime import datetime as dt
import time
import os
import getopt
import sys
import csv

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
    sub_data['created'] = utctodate(submission.created_utc)
    sub_data['num_comments'] = submission.num_comments
    sub_data['url'] = submission.permalink
    sub_data['text_url'] = submission.url
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
        user_data['created'] = utctodate(user.created_utc)
        user_data['gold_status'] = user.is_gold
        user_data['is_employee'] = user.is_employee
        user_data['has_verified_email'] = user.has_verified_email if user.has_verified_email is not None else False
    except Exception:
        return user_data
    return user_data

def subredditinfo(subreddit, subreddit_id):
    """Retrieve essential data for a Subreddit"""
    subreddit_data = {}
    subreddit_data['name'] = subreddit.display_name
    subreddit_data['subreddit_id'] = subreddit_id
    subreddit_data['created'] = utctodate(subreddit.created_utc)
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
        try:
            data['comment_id'] = comment.id
            data['text'] = comment.body
            is_deleted = False
            if ( data['text'] == '[deleted]'):
                is_deleted = True
            data['is_deleted'] = is_deleted
            data['created'] = utctodate(comment.created_utc)
            data['is_submitter'] = comment.is_submitter
            data['submission_id'] = comment.link_id
            data['parent_id'] = comment.parent_id
            data['comment_url'] = comment.permalink
            data['upvotes'] = comment.score
            data['replies'] = comment.replies.__len__()
            data['user'] = userinfo(comment.author)
        except Exception:
            print("Comment fail")

        comments_data.append(data)
    return comments_data

def utctodate(utctime):
    """Convert POSIX time to YYYY-MM-DD HH:MM:SS"""
    return dt.utcfromtimestamp(utctime).strftime("%Y-%m-%d %H:%M:%S")

def datetoutc(date):
    """Convert date in  list format [YYYY, M, D] to POSIX time"""
    year = int(date[0])
    month = int(date[1])
    day = int(date[2])
    return int(dt(year, month, day).timestamp())


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

def fetch_all_for_topic(csvfile, topic, submissions):
    write_header = not os.path.exists(csvfile)
    with open(csvfile, 'a+', encoding="utf-8", newline='') as out:
        csvwriter = csv.writer(out)
        if write_header:
            csvwriter.writerow(['sub_id', 'topic', 'title', 'url'])
        for sub in submissions: #reddit.subreddit('Denmark').search(query):
            csvwriter.writerow([sub.id, topic, sub.title, "https://www.reddit.com{0}".format(sub.permalink)])

def process_queries(csv_queryfile, pushAPI, outfolder):
    with open(csv_queryfile, 'r', encoding="utf-8") as queries:
        for line in queries.readlines()[1:]: #skip csv header
            vals = line.split(',')
            filename = vals[0].strip()
            topic = vals[1].strip()
            query = vals[2].strip()
            after_date = datetoutc(vals[3].strip().split('-')) #YYYY-M-D
            before_date = datetoutc(vals[4].strip().split('-')) #YYYY-M-D
            score = vals[5].strip() #lower limit score
            subs = list(pushAPI.search_submissions(
                        after=after_date,
                        before=before_date,
                        subreddit='Denmark',
                        q=query,
                        limit=50,
                        score='>{0}'.format(score)))
            outfilepath = os.path.join(outfolder, filename)
            fetch_all_for_topic(outfilepath, topic, subs)


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
            help_msg = """ 
            run with '-u'/'-user' and valid praw agent argument from praw.ini\n
            run with '-l'/'-log' to enable logging of API calls
            """
            sys.exit(help_msg)

    if not reddit:
        sys.exit("Reddit instance could not be obtained!\nSee '-help' for more information")

    pushAPI = PushshiftAPI(reddit)

    datafolder = "submissions/"
    submission_ids = "submission_ids/"

    process_queries('queries.csv', pushAPI, submission_ids)

    for info_file in os.listdir(submission_ids):
        process_submissions(reddit, datafolder, os.path.join(submission_ids, info_file))   


if __name__ == "__main__":
    main(sys.argv[1:])