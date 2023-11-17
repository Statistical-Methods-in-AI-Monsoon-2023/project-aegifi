import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, jaccard_score, classification_report
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torchtext.transforms as T 
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
import multiprocessing
from tqdm import tqdm

import sys
sys.path[0] += '/../../utils/'
from utils import load_data, hit_rate, preprocess_data

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, vocab_size, num_layers, learning_rate, dropout, bidirectional, device):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.bidirectional = bidirectional
        self.learning_rate = learning_rate
        
        self.embedding = nn.Embedding(vocab_size, input_size)
        self.gru_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gru_layers.append(nn.GRU(input_size=input_size, hidden_size=hidden_size, device=device, bidirectional=bidirectional, batch_first=True))
            self.norm_layers.append(nn.LayerNorm(hidden_size * 2 if bidirectional else hidden_size))
            input_size = hidden_size * 2 if bidirectional else hidden_size
        
        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Set initial hidden and cell states
        # h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_size).to(self.device)
        # c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_size).to(self.device)
        
        # Forward propagate LSTM pass in a mast to get the EOS token
        
            # eos_mask = (x == torch.tensor([2]).to(self.device).unsqueeze(0).unsqueeze(2)).any(dim=2)
        # eos = (x == 2).nonzero()[:, 1]
        # eos[eos == 0] = x.size(1)
        x = self.embedding(x)
        for i in range(self.num_layers):
            x, _ = self.gru_layers[i](x)
            x = self.norm_layers[i](x)
            x = self.dropout(x)
            
        # x = torch.stack([x[i, eos[i] - 1, :] for i in range(x.size(0))])
        
        
        x = self.fc(x[:, -1, :])
        x = self.sigmoid(x)
        
        return x
        
    def get_name(self):
        return 'gru'
            

def train_loop(model, train_loader, val_loader, criterion, device, epochs, save_model=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            # print(data.shape, target.shape)
            # Forward pass
            outputs = model(data)
            # print(outputs.shape, target.shape)
            loss = criterion(outputs, target)
            train_loss += loss.item()
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if (batch_idx + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch + 1, epochs, batch_idx + 1, len(train_loader), loss.item()), end='\r')
        print('\nEpoch [{}/{}], Train Loss: {:.4f}'.format(epoch + 1, epochs, train_loss / len(train_loader)))
        # model.eval()
        # with torch.no_grad():
        #     correct = 0
        #     total = 0
        #     val_loss = 0
        #     for data, target in val_loader:
        #         data = data.to(device)
        #         target = target.to(device)
        #         outputs = model(data)
        #         loss = criterion(outputs, target)
        #         val_loss += loss.item()
        #         predicted = torch.round(outputs.data)
        #         total += target.size(0) * target.size(1)
        #         correct += (predicted == target).sum().item()
        #     print('Epoch [{}/{}], Val Loss: {:.4f}, Val Accuracy: {:.2f}%'.format(epoch + 1, epochs, val_loss / len(val_loader), 100 * correct / total))
    if save_model:
        torch.save(model.state)

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
print('Using device:', device)

# print(model)
X = torch.load('./data/X.pt')
y = np.load('./vectorised_data/y.npy')

vocab = torch.load('./data/vocab.pt')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GRU(input_size=300, hidden_size=128, output_size=20, vocab_size=len(vocab), num_layers=3, learning_rate=5e-4, dropout=0.5, bidirectional=False, device=device)
model.to(device)

# create a dataloader for train and val
train_loader = DataLoader(TensorDataset(X_train, torch.from_numpy(y_train).float()), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_test, torch.from_numpy(y_test).float()), batch_size=32, shuffle=False)

criterion = nn.BCELoss()
train_loop(model, train_loader, val_loader, criterion, device, epochs=2, save_model=False)

model.eval()

with torch.no_grad():
    correct = 0
    total = 0
    test_loss = 0
    total = 0
    predicted_labels = []
    true_labels = []
    yes = True
    for data, target in val_loader:
        data = data.to(device)
        target = target.to(device)
        outputs = model(data)
        loss = criterion(outputs, target)
        test_loss += loss.item()
        if yes:
            print(outputs.data[:10])
            yes = False
        predicted = torch.round(outputs.data)
        predicted_labels.extend(predicted.cpu().numpy())
        true_labels.extend(target.cpu().numpy())
        total += target.size(0) * target.size(1)
        
    predicted_labels = np.array(predicted_labels)
    true_labels = np.array(true_labels)
    print(predicted_labels[:10])
    print(true_labels[:10])
    print('Test Loss: {:.4f}'.format(test_loss / len(val_loader)))
    print('Jaccard Score: {:.4f}'.format(jaccard_score(true_labels, predicted_labels, average='samples')))
    print('Hit Rate: {:.4f}'.format(hit_rate(predicted_labels, true_labels)))
    print(classification_report(true_labels, predicted_labels, zero_division=True))