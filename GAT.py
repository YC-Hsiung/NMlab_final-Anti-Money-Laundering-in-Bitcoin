# %%
# import packages
import pickle
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import os
from torchsummary import summary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %%
# define dataset


class elliptic_dataset(torch.utils.data.Dataset):
    def __init__(self, path):
        # read features
        feature_path = os.path.join(path, "elliptic_txs_features.csv")
        df = pd.read_csv(feature_path, header=None)
        fdf = df.to_numpy()
        num_features = 165
        self.features = torch.zeros(
            (len(fdf), num_features), dtype=torch.float32)
        self.timestepidx = []
        current_time = 0
        for i, feature in enumerate(fdf):
            if current_time < feature[1]:
                current_time += 1
                self.timestepidx.append(i)

            self.features[i] = torch.tensor(feature[2:], dtype=torch.float32)
        print("features read!")
        # read classes
        class_path = os.path.join(path, "elliptic_txs_classes.csv")
        df = pd.read_csv(class_path)
        self.IdToidx = {}
        for idx, Id in enumerate(df["txId"].to_numpy()):
            self.IdToidx[Id] = idx
        label = df["class"].str.replace("unknown", '3')
        label = label.astype(np.float).to_numpy()
        self.totalnode = len(label)
        # 1 bad 2 good 3 unknown
        self.label = []
        for idx in range(len(self.timestepidx)):
            try:
                self.label.append(
                    label[self.timestepidx[idx]:self.timestepidx[idx+1]])
            except:
                self.label.append(label[self.timestepidx[idx]:])
        print("class read!")
        # read edge
        edge_path = os.path.join(path, "elliptic_txs_edgelist.csv")
        df = pd.read_csv(edge_path)
        self.adjlist = []
        for i in range(len(self.timestepidx)):
            self.adjlist.append([])
            try:
                for _ in range(self.timestepidx[i+1]-self.timestepidx[i]):
                    self.adjlist[i].append([])
            except:
                for _ in range(self.totalnode-self.timestepidx[i]):
                    self.adjlist[i].append([])
        froms = df["txId1"]
        tos = df["txId2"]
        for (id1, id2) in zip(froms, tos):
            current_time = int(fdf[self.IdToidx[id1]][1])
            self.adjlist[current_time-1][self.IdToidx[id1] -
                                         self.timestepidx[current_time-1]].append(self.IdToidx[id2] -
                                                                                  self.timestepidx[current_time-1])
        print("edge read!")

    def __getitem__(self, idx):
        return (self.adjlist[idx], self.features[idx], self.label[idx])

    def __len__(self):
        return len(self.label)


# %%
dataset = elliptic_dataset("dataset/elliptic_bitcoin_dataset")
# split dataset by timestep
# %%
# define model


class MultiGATLayer(nn.Module):
    def __init__(self, f_in, f_out, num_heads):
        super(MultiGATLayer, self).__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.num_heads = num_heads
        self.W = [nn.Linear(f_in, f_out).to(device)
                  for _ in range(self.num_heads)]
        self.a = [nn.Linear(f_in*2, 1).to(device)
                  for _ in range(self.num_heads)]
        self.softmax = nn.Softmax().to(device)
        self.leakyrelu = nn.LeakyReLU(0.2).to(device)

    def forward(self, adjlist, features):
        output_features = torch.zeros(len(features), self.f_out*self.num_heads)
        for node in range(len(features)):
            neighbors = adjlist[node].copy()
            neighbors.append(node)
            node_features = [features[neighbor] for neighbor in neighbors]
            node_features = torch.tensor(
                torch.stack(node_features), dtype=torch.float32)
            attentionkey = [self.W[i](node_features)
                            for i in range(self.num_heads)]
            transformed_features = torch.zeros(
                self.num_heads, len(neighbors), self.f_in*2)
            for k in range(self.num_heads):
                for i in range(len(neighbors)):
                    row = torch.cat(
                        [attentionkey[k][-1], attentionkey[k][i]], dim=0)
                    transformed_features[k, i, :] = row
            att_weights = [self.leakyrelu(self.a[k](transformed_features[k]))
                           for k in range(self.num_heads)]
            att_weights = [self.softmax(torch.transpose(att_weights[k], 0, 1))
                           for k in range(self.num_heads)]
            output = [torch.matmul(att_weights[k], attentionkey[k])
                      for k in range(self.num_heads)]
            output = torch.cat(output, dim=1)
            output_features[node] = output
        return output_features


# %%


class MultiGAT(nn.Module):
    def __init__(self, f_in, f_out, num_heads, num_layers):
        super(MultiGAT, self).__init__()
        self.num_heads = num_heads
        self.f_in = f_in
        self.f_out = f_out
        self.num_layers = num_layers
        self.GATlayers = [MultiGATLayer(f_in, f_out, num_heads)
                          for _ in range(num_layers)]
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, adjlist, features):
        hidden_features = features
        for i in range(self.num_layers):
            hidden_features = self.GATlayers[i](adjlist, hidden_features)
            if(i < self.num_layers):
                hidden_features = self.relu(hidden_features)
        output = self.sigmoid(hidden_features.mean(axis=1))
        return output


# %%
model = MultiGAT(165, 165, 1, 2).to(device)
# %%

# %%
# training
EPOCH = 3
criterion = nn.BCELoss()
paralist = []
for layer in model.GATlayers:
    for p in layer.a:
        paralist.append({'params': p.parameters()})
    for p in layer.W:
        paralist.append({'params': p.parameters()})
optimizer = torch.optim.Adam(paralist, lr=0.001)
for epoch in range(EPOCH):
    total_acc = 0
    total_num = 0

    for timestep in range(len(dataset.timestepidx)):
        num = 0
        loss = 0
        acc = 0
        start = dataset.timestepidx[timestep]
        try:
            end = dataset.timestepidx[timestep+1]
        except:
            end = len(dataset.timestepidx)
        output = model(dataset.adjlist[timestep],
                       dataset.features[start:end].to(device))
        for idx, label in enumerate(dataset.label[timestep]):
            if (label == 1):
                loss += criterion(output[idx],
                                  torch.tensor((0), dtype=torch.float32))
                num += 1
                total_num += 1
                if(output[idx] < 0.5):
                    acc += 1
                    total_acc += 1
            elif (label == 2):
                loss += criterion(output[idx],
                                  torch.tensor((1), dtype=torch.float32))
                num += 1
                total_num += 1
                if(output[idx] > 0.5):
                    acc += 1
                    total_acc += 1
        loss /= num
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"[{timestep+1}/49] loss={loss} acc={acc}/{num}={acc/num}")
    print(f"[{epoch+1}/{EPOCH}] acc={total_acc/total_num}")

# %%
pickle.dump(model, "./models/test_model.pkl")

# %%
