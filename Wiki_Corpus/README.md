# Extracting data from the Danish Wiki dump
Parsing the Wiki dump is done by running wiki_parser.py with redirected stdin to a file containing the dump in plain-text like this: 'python wiki_parser.py < da_wiki_text.txt'.

The data is extracted into a text file in this folder called 'wiki_sentences.txt', which consists of a sentence on each new line.
The output file is overwritten if run multiple times.
The sentence is the original state of the word from the dataset.


To get the data, download it from https://drive.google.com/file/d/0B5lWReQPSvmGSVRRQmlOMTFKRzQ/edit, originating from the Polyglot project at https://sites.google.com/site/rmyeid/projects/polyglot. 