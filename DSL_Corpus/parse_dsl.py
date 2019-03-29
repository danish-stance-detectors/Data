import os, fnmatch

dsl_data_folder = "./dsl_corpus_data/"

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
    current_pos_tags = ""
    sentences = []
    with open(file_path, "r", encoding="utf8") as txt_file:
        for line in txt_file.readlines():
            if line.startswith(end_tag):
                s = current_sentence.strip() + '\t'
                if pos_tags:
                    s += current_pos_tags
                    current_pos_tags = ""
                sentences.append(s)
                current_sentence = ""
            elif not line.startswith(start_tag):
                instance = line.split("\t")
                first_word = instance[0]
                current_sentence += (first_word + " ")
                if pos_tags:
                    epos_tag = instance[5]
                    pos_class = epos_tag.split(':')[0]
                    current_pos_tags += (pos_class + " ")

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