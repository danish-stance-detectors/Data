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
from sklearn.model_selection import train_test_split

dsl_file = 'dsl_sentences_pos.txt'
rand = random.Random(42)

def prepare_sequence(seq, to_ix, device='cpu'):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, device=device)
    
def save_obj(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def load_data(data_file, cap=10000, shuffle_lines=True):
    X = []
    y = []
    word_to_ix = {}
    tag_to_ix = {}
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
    
    def __init__(self, embedding_dim, hidden_dim, word_dict, tag_dict, epochs):
        super(LSTMPOSTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_dict = word_dict
        self.tag_dict = tag_dict
        self.epochs = epochs
        self.word_embeddings = nn.Embedding(len(word_dict), embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, len(tag_dict))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

def train_and_save(args, emb_dim, hidden_dim, epochs, data_file, cuda=False):
    if not data_file:
        exit(2)
    args.device = None
    if cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
    X, y, word_to_ix, tag_to_ix = load_data(data_file)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.33, random_state=rand, stratify=y 
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
        
    model = LSTMPOSTagger(emb_dim, hidden_dim, word_to_ix, tag_to_ix, epochs).to(args.device)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # Train
    print('Training...')
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
            print('{:10} loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
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

# def load_model(model_file_path, emb_dim, hidden_dim, data_file):
#     model = LSTMPOSTagger(emb_dim, hidden_dim, data_file)
#     model.load_state_dict(torch.load(model_file_path))
#     return model

def load_checkpoint_for_eval(filepath):
    checkpoint = torch.load(filepath)
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
        return predicted


def main(argv):
    parser = argparse.ArgumentParser(description='Train and save POS tagging model')
    parser.add_argument('-ed', '--emb_dim', dest='emb_dim', nargs='?', default='50', type=int, help='Embedding dimensions')
    parser.add_argument('-hd', '--hidden_dim', dest='hidden_dim', nargs='?', default='50', type=int, help='Hidden dimensions')
    parser.add_argument('-e', '--epochs', dest='epochs', nargs='?', default='10', type=int, help='Number of training epochs')
    # parser.add_argument('-l', '--lines', dest='lines', nargs='?', default='2000', type=int, help='Number of training lines')
    parser.add_argument('-f', '--file', dest='data_file', default=dsl_file, help='Filename of data file')
    parser.add_argument('-c', '--cuda', dest='cuda', action='store_true', help='Enable CUDA')
    args = parser.parse_args(argv)
    train_and_save(args, args.emb_dim, args.hidden_dim, args.epochs, args.data_file, args.cuda)

if __name__ == "__main__":
    main(sys.argv[1:])