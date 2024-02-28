import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class Model(nn.Module): 
    def __init__(self, hidden_dim, num_classes): 
        super(Model, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(1000, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )       

    def forward(self, x): 
        return self.model(x)


if __name__ == "__main__": 

    # configs
    num_epochs = 100
    lr = 0.01
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
    model = Model(hidden_dim, num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    epoch_train_acc, epoch_test_acc, epoch_train_loss, epoch_test_loss = [], [], [], []
    for epoch in tqdm(range(num_epochs)): 

        # train loop
        train_losses, train_acc = [], []
        for i, (x, y) in enumerate(train_loader): 
            x, y = x.to(device), y.to(device)
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

    
        


        


