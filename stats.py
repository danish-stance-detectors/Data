import os
import json

datafolder = "submissions/"

events = sorted(os.listdir(datafolder))

print("{0}\t{1} {2}\n".format('ID', 'comments', 'event'))
total_comments = 0
total_submissions = 0
total_events = 0
for event in events:
    if not os.path.isdir(os.path.join(datafolder, event)):
            continue
    submissions = os.listdir(os.path.join(datafolder, event))
    for submission in submissions:
        with open(os.path.join(datafolder, event, submission), 'r', encoding='utf-8') as datafile:
            sub = json.load(datafile)
            sub_id = sub['submission_id']
            comments = sub['num_comments']
            print("{0}\t{1}\t {2}".format(sub_id, comments, event))
            total_comments += comments
        total_submissions += 1
    total_events += 1
print()
print("Events:", total_events)
print("Submissions:", total_submissions)
print("Comments:", total_comments)