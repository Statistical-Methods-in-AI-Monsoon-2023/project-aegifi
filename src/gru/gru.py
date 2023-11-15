import torch
import torch.nn as nn

import sys
sys.path[0] += '/../utils/'
from utils import load_data, hit_rate

class GRU(nn.Module):
    # add number of memory cells as parameter
    def __init__(self, input_size, hidden_size, output_size, num_layers, learning_rate, dropout, bidirectional, device):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.bidirectional = bidirectional
        self.learning_rate = learning_rate
        
        self.gru_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gru_layers.append(nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=1, bidirectional=bidirectional, batch_first=True))
            self.norm_layers.append(nn.LayerNorm(hidden_size * 2 if bidirectional else hidden_size))
            input_size = hidden_size * 2 if bidirectional else hidden_size
        
        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_size).to(self.device)
        # c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_size).to(self.device)
        
        # Forward propagate LSTM
        for i in range(self.num_layers):
            x, _ = self.gru_layers[i](x, h0[i:i+1])
            x = self.norm_layers[i](x)
            x = self.dropout(x)
        
        # Decode the hidden state of the last time step
        x = self.fc(x[:, -1, :])
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
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, target)
            train_loss += loss.item()
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if (batch_idx + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch + 1, epochs, batch_idx + 1, len(train_loader), loss.item()))
        print('Epoch [{}/{}], Train Loss: {:.4f}'.format(epoch + 1, epochs, train_loss / len(train_loader)))
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            val_loss = 0
            for data, target in val_loader:
                data = data.to(device)
                target = target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
            print('Epoch [{}/{}], Val Loss: {:.4f}, Val Accuracy: {:.2f}%'.format(epoch + 1, epochs, val_loss / len(val_loader), 100 * correct / total))
    if save_model:
        torch.save(model.state)

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')

model = GRU(input_size=300, hidden_size=128, output_size=20, num_layers=2, learning_rate=0.001, dropout=0.5, bidirectional=False, device=device)
X_train, X_test, y_train, y_test = load_data()

