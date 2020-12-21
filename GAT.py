# %%
# import packages
from dgl.nn.pytorch import GATConv
import time
import pickle
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import random
import os
import dgl
device = 'cpu'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %%
# define dataset


class elliptic_dataset(dgl.data.DGLDataset):
    def __init__(self, path, raw_dir=None, force_reload=False, verbose=False):
        super(elliptic_dataset, self).__init__(name='elliptic',
                                               raw_dir=raw_dir, force_reload=force_reload, verbose=verbose)
        # read features
        feature_path = os.path.join(path, "elliptic_txs_features.csv")
        df = pd.read_csv(feature_path, header=None)
        fdf = df.to_numpy()
        num_features = 93
        self.features = torch.zeros(
            (len(fdf), num_features), dtype=torch.float32)
        self.timestepidx = []
        current_time = 0
        for i, feature in enumerate(fdf):
            if current_time < feature[1]:
                current_time += 1
                self.timestepidx.append(i)

            self.features[i] = torch.tensor(
                feature[2:num_features+2], dtype=torch.float32)
        print("features read!")
        # read classes
        class_path = os.path.join(path, "elliptic_txs_classes.csv")
        df = pd.read_csv(class_path)
        self.IdToidx = {}
        for idx, Id in enumerate(df["txId"].to_numpy()):
            self.IdToidx[Id] = idx
        label = df["class"].str.replace("unknown", '3')
        label = label.str.replace("2", '0')
        label = label.astype(np.float).to_numpy()
        self.totalnode = len(label)
        # 1 bad 2 good 3 unknown
        self.label = []
        for idx in range(len(self.timestepidx)):
            if(idx < len(self.timestepidx)-1):
                self.label.append(torch.tensor(
                    label[self.timestepidx[idx]:self.timestepidx[idx+1]], dtype=torch.float32))

            else:
                self.label.append(torch.tensor(
                    label[self.timestepidx[idx]:], dtype=torch.float32))
        print("class read!")
        # read edge
        edge_path = os.path.join(path, "elliptic_txs_edgelist.csv")
        df = pd.read_csv(edge_path)
        adjlist = [[[], []] for _ in range(len(self.timestepidx))]
        froms = df["txId1"].astype(np.int).to_numpy()
        tos = df["txId2"].astype(np.int).to_numpy()
        for (id1, id2) in zip(froms, tos):
            current_time = int(fdf[self.IdToidx[id1]][1])
            adjlist[current_time -
                    1][0].append(self.IdToidx[id1] - self.timestepidx[current_time-1])
            adjlist[current_time -
                    1][0].append(self.IdToidx[id2] - self.timestepidx[current_time-1])
            adjlist[current_time -
                    1][1].append(self.IdToidx[id2] - self.timestepidx[current_time-1])
            adjlist[current_time -
                    1][1].append(self.IdToidx[id1] - self.timestepidx[current_time-1])
        self.graphlist = []
        for i in range(len(adjlist)):
            self.graphlist.append(dgl.DGLGraph((adjlist[i][0], adjlist[i][1])))
            try:
                self.graphlist[i].ndata['feat'] = self.features[self.timestepidx[i]
                    :self.timestepidx[i+1]]
            except IndexError:
                self.graphlist[i].ndata['feat'] = self.features[self.timestepidx[i]:]

        print("edge read!")

    def process(self):
        pass

    def __getitem__(self, idx):
        return (self.graphlist[idx], self.label[idx])

    def __len__(self):
        return len(self.label)


# %%
dataset = elliptic_dataset("dataset/elliptic_bitcoin_dataset")
# %%
# create collate_fn


# create dataloaders
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle = True)
# %%
# define model


# %%


class GAT(nn.Module):
    def __init__(self, f_in, f_out, num_heads, num_layers, dropout):
        super(GAT, self).__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.num_heads = num_heads
        self.GATlayers = nn.ModuleList()
        self.sigmoid = nn.Sigmoid()
        for i in range(num_layers):
            self.GATlayers.append(GATConv(
                f_in, f_out, num_heads, feat_drop=dropout, attn_drop=dropout))
            f_in = f_out*num_heads

    def forward(self, graph, features):
        hidden_features = features
        for layer in self.GATlayers:
            hidden_features = layer(graph, hidden_features)
            hidden_features = hidden_features.reshape(
                -1, self.f_out*self.num_heads)

        output = hidden_features.mean(axis=1)
        return output


# %%


def eval_model(datalist):
    with torch.no_grad():
        positive = 0
        true_positive = 0
        report_positive = 0
        negative = 0
        acc = 0
        for timestep in datalist:
            start = dataset.timestepidx[timestep]
            try:
                end = dataset.timestepidx[timestep+1]
            except:
                end = len(dataset.features)
            output = model(dataset.graphlist[timestep].to(device),
                           dataset.features[start:end].to(device))
            output = torch.sigmoid(output).detach().cpu()
            labeled_idx = torch.where(dataset.label[timestep] != 3)
            positive_idx = torch.where(dataset.label[timestep] == 1)
            positive += float(positive_idx[0].shape[0])
            true_positive += torch.sum(torch.round(
                output[positive_idx[0]]) == dataset.label[timestep][positive_idx], dtype=torch.float32)
            report_positive += torch.sum(
                torch.round(output[labeled_idx]), dtype=torch.float32)
            # negative acc
            negative += torch.sum(dataset.label[timestep] == 0)
            acc += torch.sum(torch.round(output[labeled_idx]) ==
                             dataset.label[timestep][labeled_idx], dtype=torch.float32)
        recall = true_positive/positive
        try:
            precision = true_positive/report_positive
            f1 = 2/(1/precision+1/recall)
        except:
            precision = 0
            f1 = 0
        print(
            f"acc={acc/(positive+negative):.4f} precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}")
        return f1


# %%
model = GAT(93, 50, 8, 6, 0.2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# %%
# training
traininglist = range(30)
validationlist = range(30, 40)
testlist = range(40, 49)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.])).to(device)
plot_train = []
plot_val = []
plot_test = []
EPOCH = 1000
for epoch in range(EPOCH):
    total_positive = 0
    total_negative = 0
    total_true_positive = 0
    total_false_positive = 0
    total_acc = 0
    for timestep in random.sample(traininglist, len(traininglist)):
        starttime = time.time()
        positive = 0
        negative = 0
        true_positive = 0
        false_positive = 0
        loss = 0
        acc = 0
        start = dataset.timestepidx[timestep]
        try:
            end = dataset.timestepidx[timestep+1]
        except:
            end = len(dataset.features)
        output = model(dataset.graphlist[timestep].to(device),
                       dataset.features[start:end].to(device))
        labeled_idx = torch.where(dataset.label[timestep] != 3)
        loss = criterion(output[labeled_idx[0]],
                         dataset.label[timestep][labeled_idx].to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # eval
    if ((epoch+1) % 50 == 0):
        print(epoch+1)
        print("train")
        plot_train.append(eval_model(traininglist))
        print("val")
        plot_val.append(eval_model(validationlist))
        print("test")
        plot_test.append(eval_model(testlist))
# %%
plt.plot(plot_train)
plt.plot(plot_val)
plt.plot(plot_test)
# %%
torch.save(model, "./models/test_model.bin")

# %%
model = torch.load("./models/test_model.bin")
# %%
