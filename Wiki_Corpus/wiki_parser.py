import re, sys
from nltk import word_tokenize, sent_tokenize

wiki_corpus = 'da_wiki_text.txt'
out_file = 'wiki_sentences.txt'
sec_num = re.compile(r'(^\[\[\d+\]\]\n)')
punctuation = re.compile('[^a-zA-ZæøåÆØÅ0-9]')

with open(out_file, 'w+', encoding='utf8') as outfile, open(wiki_corpus, 'r', encoding='utf8') as corpus:
    sections = corpus.read().split('\n\n\n\n')
    line_count = 0
    for section in sections:
        sentences = sent_tokenize(section, language='danish')
        # sentences_merged = []
        # for sentence in sentences:
        #     sentence = sec_num.sub('', sentence)
        #     if not sentence[0].isupper():
        #         last = sentences_merged[-1]
        #         del sentences_merged[-1]
        #         sentences_merged.append(' '.join([last, sentence]))
        #     else:
        #         sentences_merged.append(sentence)

        sentence_tokens = [word_tokenize(sec_num.sub('', t), language='danish') for t in sentences]
        for tokens in sentence_tokens:
            line_count += 1
            if line_count % 100000 == 0:
                print("Parsed %d lines" % line_count)
            tokens_clean = []
            for token in tokens:
                if token and not punctuation.match(token):
                    tokens_clean.append(token)
            if not tokens_clean or len(tokens_clean) == 1:
                continue
            for t in tokens_clean:
                outfile.write(t + ' ')
            outfile.write('\n')

