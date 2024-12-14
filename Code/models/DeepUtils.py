import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

class Trainer():
    def __init__(self, model, lr, epochs, weight_decay=1e-3, save_path='model.pth', device='cpu'):
        self.model = model.to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        self.lr = lr
        self.epochs = epochs
        self.save_path = save_path
        self.device = device

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

            print(f'Epoch: {epoch+1} \tTraining Loss: {train_loss:.6f} \tTraining Acc: {train_acc:.6f}')

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
                
            print(f'Epoch: {epoch+1} \tValidation Loss: {valid_loss:.6f} \tValidation Acc: {valid_acc:.6f}')
            
            if valid_loss <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}). Saving model ...'.format(valid_loss_min, valid_loss))
                torch.save(self.model.state_dict(), self.save_path)
                valid_loss_min = valid_loss

class DfDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = torch.from_numpy(features).float()
        self.labels = labels.to_numpy()
        classes = sorted(list(set(self.labels)))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        features = self.features[idx]
        label = self.class_to_idx[self.labels[idx]]
        return features, label