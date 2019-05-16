import twint
import argparse
import sys

# See links below for setup and use
# https://github.com/twintproject/twint
# https://github.com/twintproject/twint/wiki/Module

def main(argv):
    parser = argparse.ArgumentParser(description='Query twitter data')
    parser.add_argument('-q', '--query', help='Query string', n_args='*')
    parser.add_argument('-o', '--out', help='Output folder name')
    parser.add_argument('-tid', '--tweet_id', help='Tweet id')
    parser.add_argument('-lang', '--language', help='Set language filter.')
    parser.add_argument('-since', '--since', help='take tweets since some data of format "yyyy-mm-dd"')
    parser.add_argument('-until', '--until', help='take tweets until some data of format "yyyy-mm-dd"')
    args = parser.parse_args(argv)

    if args.query:
        c = twint.Config()

        c.Search = args.query
        if args.since:
            c.Since = args.since
        if args.until:
            c.Until = args.until
        if args.language:
            c.Lang = args.language

        c.Get_replies = True
        c.Retweets = True
        c.Count = True
        c.Hide_output = True

        c.Custom["tweets"] = ["id"]
        c.Custom["tweets"] = ["conversation_id"]
        c.Custom["tweets"] = ["created_at"]
        c.Custom["tweets"] = ["time"]
        c.Custom["tweets"] = ["tweet"]
        c.Custom["tweets"] = ["replies_count"]
        c.Custom["tweets"] = ["mentions"]
        
        if args.out:
            c.Store_json = True
            c.Output = 'output/' + args.out

        twint.run.Search(c)

        

if __name__ == "__main__":
    main(sys.argv[1:])