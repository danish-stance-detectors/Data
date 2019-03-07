import twint

# See links below for setup and use
# https://github.com/twintproject/twint
# https://github.com/twintproject/twint/wiki/Module

c = twint.Config()

c.Search = "#ubådssagen"
c.Since = "2017-01-01"
c.Until = "2018-01-01"
c.Get_replies = True
c.Count = True

twint.run.Search(c)