import re, sys

wiki_corpus = sys.stdin
out_file = 'wiki_sentences.txt'

with open(out_file, 'w+', encoding='utf8') as outfile:
    sections = wiki_corpus.read().split('\n\n\n\n')
    for section in sections:
        for line in section.split('\n'):
            if(re.fullmatch(r'^\[\[\d+\]\]$', line) or len(line)==1):
                continue
            outfile.write(line + '\n')

