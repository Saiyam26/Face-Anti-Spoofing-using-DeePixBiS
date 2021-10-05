import torch
import torch.nn as nn
from Metrics import test_accuracy, test_loss


class Trainer():
    def __init__(self, train_dl, val_dl, model, epochs, opt, loss_fn, device='cpu'):
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.model = model.to(device)
        self.epochs = epochs
        self.opt = opt
        self.loss_fn = loss_fn
        self.device = device

    def train_one_epoch(self, num):
        print(f'\nEpoch ({num+1}/{self.epochs})')
        print('----------------------------------')
        # self.model.train()
        for batch, (img, mask, label) in enumerate(self.train_dl):
            img, mask, label = img.to(self.device), mask.to(self.device), label.to(self.device)
            net_mask, net_label = self.model(img)
            loss = self.loss_fn(net_mask, net_label, mask, label)

            # Train
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            if batch % 9 == 0:
                print(f'Loss : {loss}')

        # self.model.eval()
        test_acc = test_accuracy(self.model, self.val_dl)
        test_los = test_loss(self.model, self.val_dl, self.loss_fn)

        print(f'Test Accuracy : {test_acc}  Test Loss : {test_los}')
        return test_acc, test_los

    def fit(self):
        training_loss = []
        training_acc = []
        self.model.train()
        for epoch in range(self.epochs):
            train_acc, train_loss = self.train_one_epoch(epoch)
            training_acc.append(train_acc)
            training_loss.append(train_loss)

        return training_acc, training_loss
