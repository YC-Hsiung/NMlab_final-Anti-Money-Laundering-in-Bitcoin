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
from sklearn.metrics import roc_auc_score
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
                self.graphlist[i].ndata['feat'] = self.features[self.timestepidx[i]                                                                :self.timestepidx[i+1]]
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
# %%
# define model


class GAT(nn.Module):
    def __init__(self, f_in):
        super(GAT, self).__init__()
        self.f_in = f_in
        self.GATlayers = nn.ModuleList()
        self.GATlayers.extend([
            GATConv(f_in, 20, 4),
            GATConv(80, 25, 2),
            GATConv(50, 10, 1)
        ]
        )

    def forward(self, graph, features):
        hidden_features = features
        for layer in self.GATlayers:
            hidden_features = layer(graph, hidden_features)
            hidden_features = hidden_features.reshape(
                hidden_features.shape[0], -1)

        return hidden_features


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.GATlayers = nn.ModuleList()
        self.tanh = nn.Tanh()
        self.GATlayers.extend([
            GATConv(10, 10, 4),
            GATConv(40, 20, 4),
            GATConv(80, 93, 1),
        ]
        )

    def forward(self, graph, features):
        hidden_features = features
        for layer in self.GATlayers:
            hidden_features = layer(graph, hidden_features)
            hidden_features = hidden_features.reshape(
                hidden_features.shape[0], -1)

        return hidden_features

# %%


def eval_model(evallist):
    total_acc = 0
    total_positive = 0
    total_negative = 0
    total_false_positive = 0
    total_true_positive = 0
    auc = 0
    for timestep in evallist:
        start = dataset.timestepidx[timestep]
        try:
            end = dataset.timestepidx[timestep+1]
        except:
            end = len(dataset.features)
        graph = dataset.graphlist[timestep].to(device)
        features = dataset.features[start:end].to(device)
        labeled_idx = torch.where(dataset.label[timestep] != 3)
        threshold = 1
        with torch.no_grad():
            output = (D(graph, E(graph, features)) -
                      features).pow(2).mean(dim=1).detach().cpu()
        positive_idx = torch.where(dataset.label[timestep] == 1)
        positive = float(positive_idx[0].shape[0])
        true_positive = torch.sum(
            (output[positive_idx[0]] > threshold).float() == dataset.label[timestep][positive_idx], dtype=torch.float32)
        false_positive = torch.sum(
            output[labeled_idx] > threshold, dtype=torch.float32)-true_positive
        # negative acc
        negative = torch.sum(dataset.label[timestep] == 0)
        acc = torch.sum((output > threshold)[labeled_idx].float() ==
                        dataset.label[timestep][labeled_idx], dtype=torch.float32)

        total_acc += acc
        total_positive += positive
        total_negative += negative
        total_true_positive += true_positive
        total_false_positive += false_positive
        auc += roc_auc_score(dataset.label[timestep][labeled_idx].int(),
                             output[labeled_idx])
    recall = total_true_positive/total_positive
    try:
        precision = total_true_positive / \
            (total_true_positive+total_false_positive)
        f1 = 2/(1/precision+1/recall)
    except:
        precision = 0
        f1 = 0
    print(f"[{epoch+1}/{EPOCH}] acc={total_acc/(total_negative+total_positive):.4f} loss={loss:.4f} auc={auc/len(evallist):.4f} f1={f1:.4f}")


# %%
E = GAT(93).to(device)
D = Decoder().to(device)
optimizer_E = torch.optim.Adam(E.parameters(), lr=0.001)
optimizer_D = torch.optim.Adam(D.parameters(), lr=0.001)


# %%
# training
traininglist = range(30)
validationlist = range(30, 40)
testlist = range(40, 49)
# criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.])).to(device)
criterion = nn.BCELoss()
criterion_mse = nn.MSELoss()
EPOCH = 5000
for epoch in range(EPOCH):
    total_positive = 0
    total_negative = 0
    total_true_positive = 0
    total_false_positive = 0
    total_acc = 0
    auc = 0
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
        graph = dataset.graphlist[timestep].to(device)
        features = dataset.features[start:end].to(device)
        noise = torch.randn_like(features).to(device)*0.2
        output = D(graph, E(graph, features+noise))
        loss = criterion_mse(output, features)
        optimizer_D.zero_grad()
        optimizer_E.zero_grad()
        loss.backward()
        optimizer_D.step()
        optimizer_E.step()
    # eval
    if (epoch % 50 == 0):
        eval_model(traininglist)
        eval_model(validationlist)
        eval_model(testlist)

# %%
#torch.save(model, "./models/OCGAN_model.bin")

# %%
#model = torch.load("./models/test_model.bin")
# %%
