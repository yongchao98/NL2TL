from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import os
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from argparse import ArgumentParser
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 40

parser = ArgumentParser()
parser.add_argument('-seed', '--seed', type=int, default=1203) # input random seed
parser.add_argument('-data_size', '--data_size', default='0.01-0.09') # The dataset size range '0.1-0.9' or '0.01-0.09'
args = parser.parse_args()
int_seed = args.seed
data_size = args.data_size

SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def readLangs(lang1, lang2, reverse=False, train_ratio=0.1):
    print("Reading lines...")

    # Read the file and split into lines
    home_path = 'Data_transfer_domain/Seq2Seq_baseline/Seq2seq_lifted_dataset_all_txt/'

    my_file = open(home_path + 'src.txt', "r")
    data_test_sentence = my_file.read()
    test_sentence = data_test_sentence.split("\n\n\n")
    my_file.close()

    my_file = open(home_path + 'tar.txt', "r")
    data_test_LTL = my_file.read()
    test_LTL = data_test_LTL.split("\n\n\n")
    my_file.close()
    print('test sentence length is: ', test_sentence)
    print('test LTL length is: ', test_LTL)

    lines_train = []; lines_test = []
    for i in range(len(test_sentence)):
        if i < int(len(test_sentence)*train_ratio):
          lines_train.append(test_sentence[i]+'\t'+test_LTL[i])
        else:
          lines_test.append(test_sentence[i]+'\t'+test_LTL[i])
    print('The training ratio is: ', train_ratio)
    print('The num of training dataset is: ', len(lines_train))
    print('The num of testing dataset is: ', len(lines_test))
    print('/n'*2)

    # Split every line into pairs and normalize
    pairs_train = [[s for s in l.split('\t')] for l in lines_train]
    pairs_test = [[s for s in l.split('\t')] for l in lines_test]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs_train = [list(reversed(p)) for p in pairs_train]
        input_lang_total = Lang(lang2)
        output_lang_total = Lang(lang1)
        pairs_test = [list(reversed(p)) for p in pairs_test]
    else:
        input_lang_total = Lang(lang1)
        output_lang_total = Lang(lang2)
    return input_lang_total, output_lang_total, pairs_train, pairs_test

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(lang1, lang2, reverse=False ,train_ratio=0.1):
    input_lang_total, output_lang_total, pairs_train, pairs_test = readLangs(lang1, lang2, reverse, train_ratio)
    print("Read %s train sentence pairs" % len(pairs_train))
    pairs_train = filterPairs(pairs_train)
    print("Trimmed to %s sentence pairs" % len(pairs_train))

    print("Read %s test sentence pairs" % len(pairs_test))
    pairs_test = filterPairs(pairs_test)
    print("Trimmed to %s sentence pairs" % len(pairs_test))
    print("Counting words...")
    for pair in pairs_train:
        input_lang_total.addSentence(pair[0])
        output_lang_total.addSentence(pair[1])
    for pair in pairs_test:
        input_lang_total.addSentence(pair[0])
        output_lang_total.addSentence(pair[1])

    print("Counted words:")
    print(input_lang_total.name, input_lang_total.n_words)
    print(output_lang_total.name, output_lang_total.n_words)
    return input_lang_total, output_lang_total, pairs_train, pairs_test

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(input_lang, output_lang, pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length = MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

import time
import math
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def trainIters(encoder, decoder, n_iters, pairs_train, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(input_lang_total, output_lang_total,random.choice(pairs_train))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang_total, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang_total.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

dataset_name = 'dataset_all_seq2seq_baseline'
output_dir_total = 'trained_models/' + dataset_name + '/'
if not os.path.exists(output_dir_total):
  os.mkdir(output_dir_total)

if data_size == '0.1-0.9':
    output_dir = output_dir_total+dataset_name+'_seed'+str(int_seed)+'_one'+'/'
elif data_size == '0.01-0.09':
    output_dir = output_dir_total + dataset_name + '_seed' + str(int_seed) + '_pointone' + '/'
if not os.path.exists(output_dir):
  os.mkdir(output_dir)

with open(output_dir +'result.txt', 'w') as f_result:
    for i in range(9):
        if data_size == '0.1-0.9':
            ratio = 0.1*(i+1)
        elif data_size == '0.01-0.09':
            ratio = 0.01*(i+1)
        input_lang_total, output_lang_total, pairs_train, pairs_test = prepareData('sentence', 'stl', False, train_ratio=ratio)
        hidden_size = 256
        encoder1 = EncoderRNN(input_lang_total.n_words, hidden_size).to(device)
        attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang_total.n_words, dropout_p=0.1).to(device)
        trainIters(encoder1, attn_decoder1, 75000, pairs_train, print_every=5000)

        count = 0
        for j in range(min(len(pairs_test),1000)):
            pair = pairs_test[j]
            output_words, attentions = evaluate(encoder1, attn_decoder1, pair[0])
            output_sentence = ' '.join(output_words)
            if pair[1].split(' ') == output_sentence.split(' ')[:-1]:
                count += 1
            else:
                print('>', pair[0])
                print('=', pair[1])
                print('<', output_sentence)
                print('\n')
        print('Training dataset ratio is: ', ratio)
        print('Accuracy is: ', count /len(pairs_test))
        print('\n'*2)
        f_result.write(str(ratio) + '  ' + str(count / (j + 1)) + '\n')
f_result.close()
