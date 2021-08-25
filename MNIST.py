import seaborn as sns
sns.color_palette("bright")
from Neural_ODE import *
import torch
from torch import nn
import torchvision
import pandas as pd

use_cuda = torch.cuda.is_available()
# In this file, Neural ODE will be used for supervised learning
# MNIST dataset will be downloaded online and the process of
# loss decline will be visualiazed


def add_time(in_tensor, t):
    bs, c, w, h = in_tensor.shape
    return torch.cat((in_tensor, t.expand(bs, 1, w, h)), dim=1)


class ConvODEF(ODEF):
    def __init__(self, dim):
        super(ConvODEF, self).__init__()
        self.conv1 = conv3x3(dim + 1, dim)
        self.norm1 = norm(dim)
        self.conv2 = conv3x3(dim + 1, dim)
        self.norm2 = norm(dim)

    def forward(self, x, t):
        xt = add_time(x, t)
        h = self.norm1(torch.relu(self.conv1(xt)))
        ht = add_time(h, t)
        dxdt = self.norm2(torch.relu(self.conv2(ht)))
        return dxdt


class ContinuousNeuralMNISTClassifier(nn.Module):
    # defining MNIST classifier at here
    # which combines convolution layers with ODE
    def __init__(self, ode):
        super(ContinuousNeuralMNISTClassifier, self).__init__()
        self.downsampling = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
        )
        self.feature = ode
        self.norm = norm(64)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.downsampling(x)
        x = self.feature(x)
        x = self.norm(x)
        x = self.avg_pool(x)
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        x = x.view(-1, shape)
        out = self.fc(x)
        return out


def data_loader_train():
    # load training data
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST("data/mnist", train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize((img_mean,), (img_std,))
                                   ])
                                   ),
        batch_size=batch_size, shuffle=True
    )
    return train_loader


def data_loader_test():
    # load testing data
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST("data/mnist", train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize((img_mean,), (img_std,))
                                   ])
                                   ),
        batch_size=128, shuffle=True
    )
    print(test_loader)
    return test_loader


def test():
    accuracy = 0.0
    num_items = 0
    test_loader = data_loader_test()
    model.eval()
    criterion = nn.CrossEntropyLoss()
    print(f"Testing...")
    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(enumerate(test_loader), total=len(test_loader)):
            if use_cuda:
                data = data.cuda()
                target = target.cuda()
            output = model(data)
            accuracy += torch.sum(torch.argmax(output, dim=1) == target).item()
            num_items += data.shape[0]
    accuracy = accuracy * 100 / num_items
    print("Test Accuracy: {:.3f}%".format(accuracy))


def train(epoch):
    num_items = 0
    train_losses = []
    train_loader = data_loader_train()
    model.train()
    criterion = nn.CrossEntropyLoss()
    print(f"Training Epoch {epoch}...")
    for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        if use_cuda:
            data = data.cuda()
            target = target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_losses += [loss.item()]
        num_items += data.shape[0]
    print('Train loss: {:.5f}'.format(np.mean(train_losses)))
    return train_losses


if __name__ == '__main__':
    # defining model at here for future use
    func = ConvODEF(64)
    ode = NeuralODE(func)
    model = ContinuousNeuralMNISTClassifier(ode)
    if use_cuda:
        model = model.cuda()
    img_std = 0.3081
    img_mean = 0.1307
    # the size for per batch is 32
    batch_size = 32
    optimizer = torch.optim.Adam(model.parameters())
    # the total epochs could be changed at here
    n_epochs = 5
    test()
    train_losses = []
    for epoch in range(1, n_epochs + 1):
        train_losses += train(epoch)
        test()
    plt.figure(figsize=(9, 5))
    history = pd.DataFrame({"loss": train_losses})
    history["cum_data"] = history.index * batch_size
    history["smooth_loss"] = history.loss.ewm(halflife=10).mean()
    history.plot(x="cum_data", y="smooth_loss", figsize=(12, 5), title="train error")
