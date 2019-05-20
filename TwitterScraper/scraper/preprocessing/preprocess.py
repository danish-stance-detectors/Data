import argparse
import sys
import json

# Converts files in the twint json format to branch json format

def main(argv):
    parser = argparse.ArgumentParser(description='Query twitter data')
    parser.add_argument('-f', '--file', help='File to preprocess')
    parser.add_argument('-o', '--out', help='Output folder name')
    parser.add_argument('-r', '--reddit', action='store_true', default=True, help='Output in reddit converted format')
    args = parser.parse_args(argv)

    if args.file:
        tweets = read_twint_file(args.file)
        conversations = arrange_conversations(tweets)

        conversation_branches = dict()
        for id, tweets in conversations.items():
            source = [c for c in tweets if int(c['has_parent_tweet']) == 0]
            if len(tweets) > 1 and len(source) > 0:
                branches = sort_conversation(id, tweets)
                branches.sort(key=lambda x: x[0]['created_at'])
                # convert to reddit format
                if args.reddit:
                    source = tweet_to_submission(source[0])
                    for i in range(len(branches)):
                        for j in range(len(branches[i])):
                            branches[i][j] = tweet_to_reddit_format(branches[i][j])
                        
                conv_dict = {'redditSubmission': source, 'branches':branches}

                with open(args.out + id + '.json', 'w+', encoding='utf8') as conv_file:
                    json.dump(conv_dict, conv_file, ensure_ascii=False)

def sort_conversation(conv_id, conversation):
    bottom_tweets = find_bottom_tweets(conversation)
    branches = [find_branch_from_bottom(bt, conversation) for bt in bottom_tweets]
    for branch in branches:
        branch.sort(key=lambda x: x['created_at'])

    return branches

def find_branch_from_bottom(bottom_tweet, tweets):
    current_tweet = bottom_tweet
    branch = []

    while current_tweet is not None:
        if current_tweet['id'] == int(current_tweet['conversation_id']):
            current_tweet = None
        else:
            branch.append(current_tweet)
            current_tweet = find_parent_tweet(current_tweet['id'], tweets)
    
    return branch
        
def find_parent_tweet(tweet_id, tweets):

    for tweet in tweets:
        for reply_info in tweet['replies']:
            if reply_info['id'] == tweet_id:
                return tweet
    
    return None

def find_bottom_tweets(tweets):
    return [t for t in tweets if t['has_parent_tweet'] == 1 and t['replies_count'] == 0]

# Groups tweet list into conversations
def arrange_conversations(tweet_list):
    conversations = dict()

    for tweet in tweet_list:
        conv_id = tweet['conversation_id']
        if conv_id not in conversations:
            conversations[conv_id] = []
        conversations[conv_id].append(tweet)
    
    return conversations

def read_twint_file(file_name):
    """
    Reads a twint file and returns tweets with replies in json format
    """
    tweets = []
    with open(file_name, "r", encoding='utf8') as file:
        for line in file.readlines():
            tweet = json.loads(line)
            if int(tweet['replies_count']) > 0 or int(tweet['has_parent_tweet']) > 0:
                tweets.append(tweet)

    return tweets

# converts a tweet to reddit comment format
def tweet_to_reddit_format(tweet):
    format_tweet = {
        'annotator' : '',
        'comment' : {
            'comment_id' : tweet['id'],
            'text' : tweet['tweet'],
            'parent_id' : '',
            'comment_url' : tweet['link'],
            'created' : tweet['date'] + 'T' + tweet['time'],
            'upvotes' : tweet['likes_count'],
            'is_submitter' : False,
            'is_deleted' : False,
            'text_url' : '',
            'replies' : tweet['replies_count'],
            'user' : [],
            'submission_id' : tweet['conversation_id'],
            'SDQC_Submission' : 'Supporting',
            'SDQC_Parent' : 'Supporting',
            'Certainty' : 'Certain',
            'Evidentiality' : 'No evidence,',
            'AnnotatedAt': '2019-03-21T08:33:08'
        }
    }

    return format_tweet

# converts a source tweet to reddit submission format
def tweet_to_submission(tweet):
    format_source = {
        'submission_id' : tweet['id'],
        'title' : tweet['tweet'],
        'text' : tweet['tweet'],
        'created' : tweet['date'] + 'T' + tweet['time'],
        'num_comments' : tweet['replies_count'],
        'url' : tweet['link'],
        'text_url' : '',
        'is_video' : False,
        'upvotes' : tweet['likes_count'],
        'user' : [], # Default values
        'subreddit' : '',
        'comments' : '',
        'IsIrrelevant' : False,
        'IsRumour' : True,
        'TruthStatus': 'Unverified',
        'RumourDescription': 'Default fill value',
        'SourceSDQC': 'Supporting',
        'UserName': 'OfUnknownOrigin'
    }
    

    return format_source

if __name__ == "__main__":
    main(sys.argv[1:])