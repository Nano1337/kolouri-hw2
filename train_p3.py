import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class Model(nn.Module): 
    def __init__(self, num_classes): 
        super(Model, self).__init__()
        ksize = 25
        self.conv1d = nn.Conv1d(1, 16, kernel_size=ksize, padding_mode='circular', padding=ksize//2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * 1000, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1d(x)
        x = self.relu(x)
        x = self.flatten(x)  # Flatten the output for the linear layer
        x = self.fc(x)
        return x

def save_plot_line_graphs(list1, list2, filename="loss.png", plot_type='loss'):

    plt.figure(figsize=(10, 6))  # Set the figure size for better readability
    
    # Plotting both lists
    plt.plot(list1, label='Train', marker='.')  
    plt.plot(list2, label='Test', marker='.') 
    
    # Adding title and labels
    if plot_type == 'loss': 
        plt.title('Loss vs Epochs')
    else: 
        plt.title('Accuracy vs Epochs')
    plt.xlabel('Epochs')
    if plot_type == 'loss': 
        plt.ylabel('Loss')
    else: 
        plt.ylabel('Accuracy')
    
    plt.legend()
    plt.savefig(filename)
    plt.close()

def apply_roll(tensor,  dims, p=0.5, max_shift=None): 
    rand_n = random.random()

    if max_shift is None: 
        max_shift = tensor.size(dims)

    shifts = random.randint(-max_shift, max_shift)

    if rand_n < p:
        return torch.roll(tensor, shifts=shifts, dims=dims)
    else: 
        return tensor

if __name__ == "__main__": 

    # configs
    num_epochs = 250
    lr = 0.005
    num_classes = 2
    hidden_dim = 10

    # load data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = np.load('hw2_p3.pkl', allow_pickle=True)
    X_train, y_train, X_test, y_test = data

    # create dataset
    train_set = TensorDataset(X_train, y_train)
    test_set = TensorDataset(X_test, y_test)

    # create dataloaders
    train_loader = DataLoader(
        train_set, 
        batch_size=16
    )
    test_loader = DataLoader(
        test_set, 
        batch_size=50
    )

    # create model, loss objective, and optimizer

    
    model = Model(num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    epoch_train_acc, epoch_test_acc, epoch_train_loss, epoch_test_loss = [], [], [], []
    for epoch in tqdm(range(num_epochs)): 

        # train loop
        train_losses, train_acc = [], []
        for i, (x, y) in enumerate(train_loader): 
            x, y = x.to(device), y.to(device)
            
            # apply shift data aug (usually done in dataset but I'm lazy)
            x = apply_roll(x, dims=1)

            out = model(x)
            loss = loss_fn(out, y)  
            train_losses.append(loss.item())

            acc = (torch.argmax(out, dim=1) == y).float().mean()
            train_acc.append(acc.item())

            opt.zero_grad()
            loss.backward()
            opt.step()

        epoch_train_acc.append(np.mean(np.array(train_acc)))
        epoch_train_loss.append(np.mean(np.array(train_losses)))

        # test loop
        test_losses, test_acc = [], []
        for i, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            with torch.no_grad(): 
                out = model(x)
                loss = loss_fn(out, y)
                test_losses.append(loss.item())

                acc = (torch.argmax(out, dim=1) == y).float().mean()
                test_acc.append(acc.item())

        epoch_test_acc.append(np.mean(np.array(test_acc)))
        epoch_test_loss.append(np.mean(np.array(test_losses)))


    save_plot_line_graphs(epoch_train_acc, epoch_test_acc, filename='figs/p3_acc.png', plot_type='acc')
    save_plot_line_graphs(epoch_train_loss, epoch_test_loss, filename='figs/p3_loss.png', plot_type='loss')


        


