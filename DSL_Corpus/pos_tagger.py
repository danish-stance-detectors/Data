# Source: https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs)

tag_to_ix = {}
with open('POS_CLASSES.txt', 'r', encoding='utf8') as pos_tags:
    for pos_tag in pos_tags.readlines():
        tag_to_ix[pos_tag.rstrip('\n')] = len(tag_to_ix)
# print(tag_to_ix)

training_data = []
word_to_ix = {}
with open('dsl_sentences.txt', 'r', encoding='utf8') as data:
    for line in data.readlines()[:20]:
        instances = line.rstrip('\n').split('\t')
        sentence = instances[0].split()
        pos_tags = instances[1].split()
        training_data.append((sentence, pos_tags))
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
# print(training_data)
# print(word_to_ix)

EMBEDDING_DIM = 6
HIDDEN_DIM = 6

class LSTMPOSTagger(nn.Module):
    
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMPOSTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

model = LSTMPOSTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Train
for epoch in range(100):
    for sentence, tags in training_data:
        model.zero_grad()

        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)

        tag_scores = model(sentence_in)

        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

# See what the scores are after training
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)
    predicted = torch.argmax(tag_scores.data, dim=1)

    print(tag_scores)
    print(predicted)