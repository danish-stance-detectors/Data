import os, fnmatch, re
import argparse
from nltk import word_tokenize

punctuation = re.compile('[^a-zA-ZæøåÆØÅ0-9]')


dsl_data_folder = "./dsl_corpus_data/"

parser = argparse.ArgumentParser(description='Parse DSL data')
parser.add_argument('-p', '--pos', dest='pos', action='store_true', default=False, help='Include POS tags')
args = parser.parse_args()

def find_files(path, filter):
    file_paths = []
    for root, _, files in os.walk(path):
        for file in fnmatch.filter(files, filter):
            file_paths.append(os.path.join(root, file))
    
    return file_paths

def parse_file(file_path, pos_tags=False):
    start_tag = "<s"
    end_tag = "</s>"
    current_sentence = ""
    # current_pos_tags = ""
    sentences = []
    with open(file_path, "r", encoding="utf8") as txt_file:
        for line in txt_file.readlines():
            if line.startswith(end_tag):
                clean_text = punctuation.sub(' ', current_sentence.strip())
                tokens = word_tokenize(clean_text, language='danish')
                # if pos_tags:
                    # s += '\t' + current_pos_tags
                    # current_pos_tags = ""
                sentences.append(tokens)
                current_sentence = ""
            elif not line.startswith(start_tag):
                instance = line.split("\t")
                first_word = instance[0]
                current_sentence += (first_word + " ")
                # if pos_tags:
                    # epos_tag = instance[5]
                    # pos_class = epos_tag.split(':')[0]
                    # current_pos_tags += (pos_class + " ")

    return sentences

# parses files one by one and saves them to output path
def save_all_files(file_path_list, output_path):
    if os.path.exists(output_path):
        os.remove(output_path)
    
    with open(output_path, "a+", encoding="utf8") as output_file:
        for file_path in file_path_list:
            sentences = parse_file(file_path, args.pos)

            for sentence in sentences:
                for token in sentence:
                    output_file.write(token + ' ')
                output_file.write("\n")

files = find_files(dsl_data_folder, "*.txt")

if args.pos:
    save_all_files(files, "./dsl_sentences_pos")
else:
    save_all_files(files, "./dsl_sentences.txt")