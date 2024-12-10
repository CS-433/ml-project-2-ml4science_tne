import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

class Trainer():
    def __init__(self, model, lr, epochs, batch_size, weight_decay=1e-2, save_path='model.pth', device='cpu'):
        self.model = model.to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.save_path = save_path

    def train(self, train_loader, val_loader):
        valid_loss_min = np.Inf

        for epoch in range(self.epochs):
            train_loss = 0.0
            valid_loss = 0.0
            train_acc = 0.0
            valid_acc = 0.0

            self.model.train()
            for input, target in tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.epochs} - Training'):
                input, target = input.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(input)
                loss = self.criterion(logits, target)
                loss.backward()
                self.optimizer.step()

                _, preds = torch.max(logits, 1)
                train_loss += loss.item() * input.size(0)
                train_acc += torch.sum(preds == target).item()

            train_loss /= len(train_loader.dataset)
            train_acc /= len(train_loader.dataset)

            self.model.eval()
            with torch.no_grad():
                for input, target in tqdm(val_loader, desc=f'Epoch {epoch+1}/{self.epochs} - Validation'):
                    input, target = input.to(self.device), target.to(self.device)
                    logits = self.model(input)
                    _, preds = torch.max(logits, 1)
                    loss = self.criterion(logits, target)
                    valid_loss += loss.item() * input.size(0)
                    valid_acc += torch.sum(preds == target).item()

            valid_loss /= len(val_loader.dataset)
            valid_acc /= len(val_loader.dataset)
                
            print(f'Epoch {epoch+1}/{self.epochs}, Train Loss: {loss.item()}, Train Accuracy: {train_acc:.4f},\
                Valid Loss: {valid_loss.item()}, Valid Accuracy: {valid_acc:.4f}')
            
            if valid_loss <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
                torch.save(self.model.state_dict(), self.save_path)
                valid_loss_min = valid_loss