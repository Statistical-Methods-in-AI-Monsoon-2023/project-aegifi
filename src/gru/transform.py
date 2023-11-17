import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torchtext.transforms as T 
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
import multiprocessing
from tqdm import tqdm

df = pd.read_csv('./data/preprocessed_data.csv')

X = df['plot'].values
tokenizer = get_tokenizer('basic_english')

def yield_tokens(data):
    for plot in data:
        tokens = tokenizer(plot)
        yield tokens
        
vocab = build_vocab_from_iterator(yield_tokens(X),min_freq=5, specials=['<UNK>', '<SOS>', '<EOS>', '<PAD>'], special_first=True)
vocab.set_default_index(vocab['<UNK>'])
# print(vocab.get_itos())
print("Vocab size: ", len(vocab))

def getTransform(vocab):
    text_tranform = T.Sequential(
        # converts the sentences to indices based on given vocabulary
        T.VocabTransform(vocab=vocab),
        # Add <sos> at beginning of each sentence. 1 because the index for <sos> in vocabulary is 1 as seen in previous section
        T.AddToken(1, begin=True),
        # Add <eos> at beginning of each sentence. 2 because the index for <eos> in vocabulary is 2 as seen in previous section
        T.AddToken(2, begin=False)
    )
    return text_tranform

def apply_transform(vocab, tokenizer, text):
    return torch.tensor(getTransform(vocab=vocab)(tokenizer(text)), dtype=torch.int)

# X = transfrom(X, vocab, tokenizer)
X = [apply_transform(vocab, tokenizer, text) for text in tqdm(X)]

X = pad_sequence(X, padding_value=vocab.get_stoi()['<PAD>'])
X = X.T
print(X.shape)

# save X
torch.save(X, './data/X.pt')
# save vocab
torch.save(vocab, './data/vocab.pt')