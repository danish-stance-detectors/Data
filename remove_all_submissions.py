import os

datafolder = 'submissions/'

events = os.listdir(datafolder)
for event in events:
    path = os.path.join(datafolder, event)
    if not os.path.isdir(path):
        continue
    os.rmdir(path)
    print("Removed", path)