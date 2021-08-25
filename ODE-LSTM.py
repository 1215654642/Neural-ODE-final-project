import sys
from torchdyn.models import *
from torchdyn import *
from torchdiffeq import odeint
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.metrics.functional import accuracy

import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import statistics
import matplotlib.pyplot as plt


class TempDataModule(pl.LightningDataModule):
    # pre-process training and testing data
    def __init__(self, window=10, batch_size=1, with_time=False):
        super().__init__()
        self.window = window
        self.batch_size = batch_size
        self.with_time = with_time

    def setup(self, stage=None):
        df = pd.read_csv(
            r"data/climate/DailyDelhiClimateTrain.csv")
        df1 = pd.read_csv(
            r"data/climate/DailyDelhiClimateTest.csv")
        train = self.normalise1D(torch.FloatTensor(df['meantemp']))
        test = self.normalise1D(torch.FloatTensor(df1['meantemp']))

        if (self.with_time):
            train_with_time = []
            test_with_time = []
            for i in range(len(train)):
                train_with_time.append(torch.Tensor([torch.Tensor([i]), train[i]]))
            for i in range(len(test)):
                test_with_time.append(torch.Tensor([torch.Tensor([i]), test[i]]))
            train_with_time = torch.stack(train_with_time)
            test_with_time = torch.stack(test_with_time)

            self.train_data = self.sequence_data(train_with_time, self.window)
            self.test_data = self.sequence_data(test_with_time, self.window)
        else:
            self.train_data = self.sequence_data(train, self.window)
            self.test_data = self.sequence_data(test, self.window)

    def train_dataloader(self):
        if self.with_time:
            train_dataloader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=False)
        else:
            train_dataloader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

        return train_dataloader

    def test_dataloader(self):
        test_dataloader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)

        return test_dataloader

    def normalise1D(self, tensor1D):

        return tensor1D

    def sequence_data(self, data, window):
        sequence = []
        for i in range(0, len(data) - window):
            inputs_seq = data[i:i + window]
            label = data[i + window:i + window + 1]
            sequence.append((inputs_seq, label))

        return sequence


class ODELSTMCell(nn.Module):
    #combining LSTM with ODE as a cell
    def __init__(self, input_size, hidden_size):
        super(ODELSTMCell, self).__init__()
        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.input_size = input_size
        self.hidden_size = hidden_size
        # choosing suitble solver and the used solver here is 'dopri5'
        # and adjoint method has been applied to decline memory cost
        self.ode = NeuralDE(self.fc, solver='dopri5', sensitivity='adjoint')
    def forward(self, inputs, hx, ts):
        batch_size = ts.size(0)
        trajectory = []
        for i, t in enumerate(ts):
            trajectory.append(self.ode.trajectory(hx[0], t))
        trajectory = torch.stack(trajectory)

        new_h = trajectory[torch.arange(batch_size), 1, torch.arange(batch_size), :]
        new_h, new_c = self.lstm(inputs, (new_h, hx[1]))
        new_h, new_c = self.lstm(inputs, (new_h, new_c))

        return (new_h, new_c)


class ODELSTM(pl.LightningModule):
    # input_size : input size of initial model
    # hidden_size : the size of hidden state
    # cell: the cell used here is ODELSTMcell state above
    def __init__(self, input_size, hidden_size):
        super(ODELSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.cell = ODELSTMCell(input_size, hidden_size)
        self.fc = nn.Linear(self.hidden_size, 1)

    def forward(self, x, timespans):
        batch_size = x.size(0)
        seq_len = x.size(1)
        hidden_state = (
            torch.zeros((batch_size, self.hidden_size)),
            torch.zeros((batch_size, self.hidden_size)),
        )

        outputs = []
        last_output = torch.zeros((batch_size, 1))
        for t in range(1, seq_len):
            inputs = torch.unsqueeze(x[:, t], 1)
            ts = timespans[:, t - 1:t + 1]
            hidden_state = self.cell.forward(inputs, hidden_state, ts)
            current_output = self.fc(hidden_state[0])
            outputs.append(current_output)
            last_output = current_output

        outputs = torch.stack(outputs, dim=1)

        return last_output

    def training_step(self, batch, batch_idx):
        # compute MSEloss
        inputs, labels = batch
        logits = self.forward(inputs[:, :, 1], inputs[:, :, 0])
        criterion = nn.MSELoss()
        loss = criterion(logits, labels[:, :, 1])

        return loss

    def test_step(self, batch, batch_idx):
        # return predictions
        inputs, labels = batch
        logits = self.forward(inputs[:, :, 1], inputs[:, :, 0])
        items = [x.item() for x in logits]

        return {'predictions': items}

    def test_epoch_end(self, outputs):
        temp_data_module = TempDataModule()
        df = pd.read_csv(
            r"data/climate/DailyDelhiClimateTrain.csv")
        df1 = pd.read_csv(
            r"data/climate/DailyDelhiClimateTest.csv")
        data = temp_data_module.normalise1D(torch.FloatTensor(df['meantemp']))
        old = temp_data_module.normalise1D(torch.FloatTensor(df1['meantemp']))
        # collect all predictions
        preds = [x['predictions'] for x in outputs]
        old_items = [x.item() for x in old]
        results = []
        for i in range(len(preds)):
            for pred in preds[i]:
                preds1.append(pred)
        # draw a graph of test predictions
        plt.title('meantemp')
        plt.ylabel('meantemp')
        plt.xlabel('Day')
        plt.autoscale(axis='x', tight=True)
        plt.plot(preds1, label='prediction')
        plt.plot(old_items[10:], label='actual')
        plt.legend()
        plt.show()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)

def drawing():
    # translate date data and draw corresponding graph
    date_init = 201301
    date = []
    for i in range(4):
        for j in range(1,13):
            date.append(j)
    date.append(1)
    df = pd.read_csv("data/climate/DailyDelhiClimateTrain.csv")
    x = 2013
    d1 = df['meantemp']
    d2 = df['humidity']
    d3 = df['meanpressure']
    df['date'] = pd.to_datetime(df['date'])
    a = df.groupby(df['date'].dt.month & df['date'].dt.year == 2013).mean()
    n = pd.DataFrame
    a1 = np.where(df['date'].dt.year == x)[0]
    b1 = df.groupby(df['date'][a1].dt.month).mean()
    a2 = np.where(df['date'].dt.year == 2014)[0]
    a3 = np.where(df['date'].dt.year == 2015)[0]
    a4 = np.where(df['date'].dt.year == 2016)[0]
    a5 = np.where(df['date'].dt.year == 2017)[0]
    b2 = df.groupby(df['date'][a2].dt.month).mean()
    b3 = df.groupby(df['date'][a3].dt.month).mean()
    b4 = df.groupby(df['date'][a4].dt.month).mean()
    b5 = df.groupby(df['date'][a5].dt.month).mean()
    result = b1.append([b2,b3,b4,b5])
    plt.subplot(311)
    plt.xlabel('Date')
    plt.ylabel('Mean Temperature')
    plt.plot(df['date'],df['meantemp'],'b')
    plt.subplot(312)
    plt.ylabel('Humidity')
    plt.xlabel('Date')
    plt.plot(df['date'], df['humidity'],'y')
    plt.subplot(313)
    plt.xlabel('Date')
    plt.ylabel('Wind Speed')
    plt.plot(df['date'], df['wind_speed'],'r')
    plt.show()


# Dataset describes minimum daily temperaturres over several years
# in Delhi
if __name__ == '__main__':
    preds1 = []
    temp_data_module = TempDataModule(10, 16, with_time=True)
    # hidden size of ODELSTM model is 100
    model = ODELSTM(input_size=1, hidden_size=100)
    # the maximum epochs set in here is 30 for all attributes
    trainer = pl.Trainer(max_epochs=30)
    trainer.fit(model, temp_data_module)
    trainer.test()
    df = pd.read_csv("data/climate/DailyDelhiClimateTest.csv")
    df1 = pd.read_csv("data/climate/DailyDelhiClimateTrain.csv")
    df2 = pd.concat([df1, df])
    df['date'] = pd.to_datetime(df['date'])
    df1['date'] = pd.to_datetime(df1['date'])
    df2['date'] = pd.to_datetime(df2['date'])
    fig1 = plt.figure(figsize=(12, 6))
    plt.title('Extrapolating')
    plt.xlabel('date')
    plt.ylabel('meantemp')
    plt.plot(df2['date'], df2['meantemp'], color='black', linewidth=2, linestyle='dotted', label='sample data')
    # plt.plot(df['date'],df['meantemp'],linestyle = '-',color = 'red',linewidth = 1 )
    plt.plot(df['date'][10:], preds1, "red", linewidth=1, linestyle='-', label='prediction')
    plt.legend()
    plt.show()
    fig1.savefig('/Users/haoyaozhang/Desktop/NODE/meantemp.jpg')


