# Source: https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import getopt, sys

dsl_file = 'dsl_sentences.txt'

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs)
    
def save_obj(obj, filename):
    with open(filename, 'wb', encoding='utf8') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def load_data(data_file):
    training_data = []
    word_to_ix = {}
    tag_to_ix = {}
    with open(data_file, 'r', encoding='utf8') as data:
        for line in data.readlines():
            instances = line.rstrip('\n').split('\t')
            sentence = instances[0].split()
            pos_tags = instances[1].split()
            for word in sentence:
                if word not in word_to_ix:
                    word_to_ix[word] = len(word_to_ix)
            for pos_tag in pos_tags:
                if pos_tag not in tag_to_ix:
                    tag_to_ix[pos_tag] = len(tag_to_ix)
            training_data.append((sentence, pos_tags))
    return training_data, word_to_ix, tag_to_ix

class LSTMPOSTagger(nn.Module):
    
    def __init__(self, embedding_dim, hidden_dim, data_file):
        super(LSTMPOSTagger, self).__init__()
        self.hidden_dim = hidden_dim
        training_data, word_dict, tag_dict = load_data(data_file)
        self.training_data = training_data
        self.word_dict = word_dict
        self.tag_dict = tag_dict
        self.word_embeddings = nn.Embedding(len(word_dict), embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, len(tag_dict))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

def train_and_save(emb_dim, hidden_dim, epochs, data_file):
    if not data_file:
        exit(2)
    model = LSTMPOSTagger(emb_dim, hidden_dim, data_file)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    training_data = model.training_data

    # Train
    n = len(training_data)
    for epoch in range(epochs):
        avg_loss = 0.0
        for sentence, tags in training_data:
            model.zero_grad()

            sentence_in = prepare_sequence(sentence, model.word_dict)
            targets = prepare_sequence(tags, model.tag_dict)

            tag_scores = model(sentence_in)

            loss = loss_function(tag_scores, targets)
            avg_loss += loss.item()
            loss.backward()
            optimizer.step()
        avg_loss /= n
        print("Epoch: {0}\tavg_loss: {1}".format(epoch, avg_loss))

    torch.save(model.state_dict(), 
        'model_state_dict_emb{0}_hidden{1}_epoch{2}.pt'.format(emb_dim, hidden_dim, epochs))

    return model

def load_model(model_file_path, emb_dim, hidden_dim, data_file):
    model = LSTMPOSTagger(emb_dim, hidden_dim, data_file)
    model.load_state_dict(torch.load(model_file_path))
    return model

def predict_pos_tags(model, sentence_tokens):
    model.eval()

    with torch.no_grad():
        inputs = prepare_sequence(sentence_tokens, model.word_dict)
        tag_scores = model(inputs)
        predicted = torch.argmax(tag_scores.data, dim=1)
        return predicted


def main(argv):
    emb_dim = 50
    hidden_dim = 50
    epochs = 10
    data_file = ""
    try:
        opts, _ = getopt.getopt(argv, "e:h:f:", ["emb_dim=","hidden_dim=","filename=","help"])
    except getopt.GetoptError:
        print("see: pos_tagger.py -help")
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-e', '-emb_dim'):
            emb_dim = arg
        elif opt in ('-h', '-hidden_dim'):
            hidden_dim = arg
        elif opt in ('-f', '-filename'):
            data_file = arg
        elif opt in '-help':
            print('Run: pos_tagger -emb_dim=<EMBEDDING SIZE> -hidden_dim=<HIDDEN DIMINSIONS> -filename=<TRAIN DATA FILE>')

    train_and_save(emb_dim, hidden_dim, epochs, data_file)

if __name__ == "__main__":
    main(sys.argv[1:])