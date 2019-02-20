import os, fnmatch

dsl_data_folder = "./dsl_corpus_data/"

def find_files(path, filter):
    file_paths = []
    for root, _, files in os.walk(path):
        for file in fnmatch.filter(files, filter):
            file_paths.append(os.path.join(root, file))
    
    return file_paths

def parse_file(file_path):
    start_tag = "<s"
    end_tag = "</s>"
    current_sentence = ""
    sentences = []
    with open(file_path, "r", encoding="utf8") as txt_file:
        for line in txt_file.readlines():
            if line.startswith(end_tag):
                sentences.append(current_sentence.strip())
                current_sentence = ""
            elif not line.startswith(start_tag):
                first_word = line.split("\t")[0]
                current_sentence += (first_word + " ")

    return sentences

# parses files one by one and saves them to output path
def save_all_files(file_path_list, output_path):
    if os.path.exists(output_path):
        os.remove(output_path)
    
    with open(output_path, "a+", encoding="utf8") as output_file:
        for file_path in file_path_list:
            sentences = parse_file(file_path)

            for sentence in sentences:
                output_file.write("%s\n" % sentence)

files = find_files(dsl_data_folder, "*.txt")

save_all_files(files, "./dsl_sentences.txt")