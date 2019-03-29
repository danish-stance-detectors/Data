import re, sys

wiki_corpus = 'da_wiki_text.txt'
out_file = 'wiki_sentences.txt'

with open(out_file, 'w+', encoding='utf8') as outfile, open(wiki_corpus, 'r', encoding='utf8') as corpus:
    sections = corpus.read().split('\n\n\n\n')
    line_count = 0
    for section in sections:
        for line in section.split('\n'):
            if(re.fullmatch(r'^\[\[\d+\]\]$', line) or len(line)==1):
                continue
            line = re.sub("[^a-zA-ZæøåÆØÅ0-9]", " ", line)
            outfile.write(line + '\n')
            line_count += 1
            if line_count % 100000 == 0:
                print("Parsed %d lines" % line_count)

