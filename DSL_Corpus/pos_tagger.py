# Source: https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import getopt 
import sys
import random
from gensim.models.keyedvectors import KeyedVectors
from sklearn.model_selection import train_test_split

dsl_file = 'dsl_sentences_pos.txt'
rand = random.Random(42)

def prepare_sequence(seq, to_ix, device='cpu'):
    idxs = [to_ix[w] if w in to_ix else 0 for w in seq]
    return torch.tensor(idxs, device=device)
    
def save_obj(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def load_data(data_file, cap, shuffle_lines=True):
    X = []
    y = []
    word_to_ix = {}
    word_to_ix = {'<OOV>': 0}
    tag_to_ix = {}
    tag_to_ix = {'<OOV>': 0}
    with open(data_file, 'r', encoding='utf8') as data:
        print('Readling lines...')
        lines = data.readlines()
        if shuffle_lines:
            rand.shuffle(lines)
        print('Done')
        for line in lines[:cap]:
            instances = line.rstrip('\n').split('\t')
            sentence = instances[0].split()
            pos_tags = instances[1].split()
            for word in sentence:
                if word not in word_to_ix:
                    word_to_ix[word] = len(word_to_ix)
            for pos_tag in pos_tags:
                if pos_tag not in tag_to_ix:
                    tag_to_ix[pos_tag] = len(tag_to_ix)
            X.append(sentence)
            y.append(pos_tags)
        print('Done')
    return X, y, word_to_ix, tag_to_ix

class LSTMPOSTagger(nn.Module):
    
    def __init__(self, embedding_dim, hidden_dim, word_dict, tag_dict, epochs, pre_trained_weights=None):
        super(LSTMPOSTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_dict = word_dict
        self.tag_dict = tag_dict
        self.ix_to_tag = {v: k for k, v in tag_dict.items()}
        self.epochs = epochs
        if pre_trained_weights:
            self.word_embeddings = nn.Embedding.from_pretrained(pre_trained_weights)
        else:
            self.word_embeddings = nn.Embedding(len(word_dict), embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, len(tag_dict))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

def train_and_save(args, emb_dim, hidden_dim, epochs, data_file, lines=100000, word_embeddings=None, cuda=False):
    if not data_file:
        exit(2)
    args.device = None
    if cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
    X, y, word_to_ix, tag_to_ix = load_data(data_file, cap=lines)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.33 
    )
    data = {
        'train': list(zip(X_train, y_train)),
        'val': list(zip(X_val, y_val))
    }
    dataset_sizes = {
        'train': len(X_train),
        'val': len(X_val)
    }
    for split, size in dataset_sizes.items():
        print(split, size)

    if word_embeddings:
        word_embeddings = torch.FloatTensor(KeyedVectors.load(word_embeddings))

    model = LSTMPOSTagger(emb_dim, hidden_dim, word_to_ix, tag_to_ix, epochs, word_embeddings).to(args.device)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # Train
    print('Training...')
    with open('pos_tagger_output.csv', 'w+') as outfile:
        for epoch in range(epochs):
            print("*****Epoch {}*****".format(epoch))
            for phase, dataset in data.items():
                if phase == 'train':
                    model.train()
                else:
                    model.eval()
                running_loss = 0.0
                running_corrects = 0

                for sentence, tags in dataset:
                    model.zero_grad()

                    sentence_in = prepare_sequence(sentence, word_to_ix, args.device)
                    targets = prepare_sequence(tags, tag_to_ix, args.device)

                    with torch.set_grad_enabled(phase == 'train'):
                        tag_scores = model(sentence_in)
                        _, preds = torch.max(tag_scores, 1)
                        loss = loss_function(tag_scores, targets)
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item()
                    running_corrects += torch.sum(preds == targets.data)
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = float(running_corrects) / dataset_sizes[phase]
                stats = '{:10} loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc)
                print(stats)
                outfile.write(stats + '\n')
    print('Done')
    print('Saving parameters...')
    checkpoint = {
        'model': LSTMPOSTagger(emb_dim, hidden_dim, word_to_ix, tag_to_ix, epochs),
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, 
        'model_checkpoint_emb{0}_hidden{1}_epoch{2}.pt'.format(emb_dim, hidden_dim, epochs))
    print('Done')
    return model

def load_checkpoint_for_eval(filepath):
    checkpoint = torch.load(filepath, map_location='cpu')
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model

def predict_pos_tags(model, sentence_tokens):
    model.eval()
    with torch.no_grad():
        inputs = prepare_sequence(sentence_tokens, model.word_dict)
        tag_scores = model(inputs)
        predicted = torch.argmax(tag_scores.data, dim=1)
        tags = []
        for pos_tag_id in predicted.data():
            tags.append(model.ix_to_tag[pos_tag_id])
        return predicted


def main(argv):
    parser = argparse.ArgumentParser(description='Train and save POS tagging model')
    parser.add_argument('-ed', '--emb_dim', dest='emb_dim', nargs='?', default='50', type=int, help='Embedding dimensions')
    parser.add_argument('-hd', '--hidden_dim', dest='hidden_dim', nargs='?', default='50', type=int, help='Hidden dimensions')
    parser.add_argument('-e', '--epochs', dest='epochs', nargs='?', default='10', type=int, help='Number of training epochs')
    parser.add_argument('-l', '--lines', dest='lines', nargs='?', default='100000', type=int, help='Number of training lines')
    parser.add_argument('-f', '--file', dest='data_file', default=dsl_file, help='Filename of data file')
    parser.add_argument('-w-', '--word_embeddings', dest='word_embeddings', nargs='?', help='Filename of pretrained word embeddings')
    parser.add_argument('-c', '--cuda', dest='cuda', action='store_true', help='Enable CUDA')
    args = parser.parse_args(argv)
    train_and_save(args, args.emb_dim, args.hidden_dim, args.epochs, args.data_file, args.lines, args.word_embeddings, args.cuda)
    # model = load_checkpoint_for_eval('model_checkpoint_emb100_hidden100_epoch100.pt')
    # tokens = "der var simpelthen ikke forsyninger nok som nåede frem til fronten til at et ordentligt forsvar kunne organiseres for slet ikke at tale om offensive operationer".split()
    # print(predict_pos_tags(model, tokens))


if __name__ == "__main__":
    main(sys.argv[1:])