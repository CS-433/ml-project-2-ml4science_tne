from tabnanny import verbose
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

class Trainer():
    def __init__(self, model, lr, epochs, weight_decay=1e-3, save_path='model.pth', device='cpu'):
        """Create a generic trainer for any Deep Learning model.

        Args:
            model (nn.Module): PyTorch model to train.
            lr (float): learning rate.
            epochs (int): number of epochs.
            weight_decay (float, optional): weight decay for AdamW. Defaults to 1e-3.
            save_path (str, optional): path where to save the final model. Defaults to 'model.pth'.
            device (str, optional): device on which to run the model. Defaults to 'cpu'.
        """
        self.model = model.to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        self.lr = lr
        self.epochs = epochs
        self.save_path = save_path
        self.device = device

    def train(self, train_loader, val_loader, verbose=True):
        """Train the model.

        Args:
            train_loader (Dataloader): PyTorch dataloader for training.
            val_loader (DataLoader): PyTorch dataloader for validation.
            verbose (bool, optional): whether or not to print intermediate results. Defaults to True.

        Returns:
            float: last validation accuracy.
        """
        for epoch in range(self.epochs):
            train_loss = 0.0
            valid_loss = 0.0
            train_acc = 0.0
            valid_acc = 0.0

            self.model.train()
            for input, target in (tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.epochs} - Training') if verbose else train_loader):
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

            if verbose:
                print(f'Epoch: {epoch+1} \tTraining Loss: {train_loss:.6f} \tTraining Acc: {train_acc:.6f}')

            self.model.eval()
            with torch.no_grad():
                for input, target in (tqdm(val_loader, desc=f'Epoch {epoch+1}/{self.epochs} - Validation') if verbose else val_loader):
                    input, target = input.to(self.device), target.to(self.device)
                    logits = self.model(input)
                    _, preds = torch.max(logits, 1)
                    loss = self.criterion(logits, target)
                    valid_loss += loss.item() * input.size(0)
                    valid_acc += torch.sum(preds == target).item()

            valid_loss /= len(val_loader.dataset)
            valid_acc /= len(val_loader.dataset)

            if verbose:    
                print(f'Epoch: {epoch+1} \tValidation Loss: {valid_loss:.6f} \tValidation Acc: {valid_acc:.6f}')
            
        torch.save(self.model.state_dict(), self.save_path)
        return valid_loss

class DfDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        """Create a PyTorch dataset using the given features and labels.

        Args:
            features (ndarray): array containing the values of all features.
            labels (ndarray): array containing the labels.
        """
        self.features = torch.from_numpy(features).float()
        self.labels = labels.to_numpy()
        classes = sorted(list(set(self.labels)))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    def __len__(self):
        """Compute the length of the dataset.

        Returns:
            int: length of the dataset.
        """
        return len(self.labels)
    
    def __getitem__(self, idx):
        """Returns a single element from the dataset.

        Args:
            idx (int): index of the element.

        Returns:
            tuple(Tensor, ndarray): the features and the label of the element.
        """
        features = self.features[idx]
        label = self.class_to_idx[self.labels[idx]]
        return features, label