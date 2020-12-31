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
# define model


# %%


class Discriminator(nn.Module):
    def __init__(self, f_in, f_out, num_heads, num_layers, feat_drop, attn_drop):
        super(Discriminator, self).__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.num_heads = num_heads
        self.GATlayers = nn.ModuleList()
        # self.sigmoid = nn.Sigmoid()
        self.fc = nn.Sequential(
            nn.Linear(self.f_out*self.num_heads, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            # nn.Sigmoid(),
        )
        for _ in range(num_layers):
            self.GATlayers.append(GATConv(
                f_in, f_out, num_heads, feat_drop=feat_drop, attn_drop=attn_drop, activation=nn.Sigmoid()))
            f_in = f_out*num_heads

    def forward(self, graph, features):
        hidden_features = features
        for layer in self.GATlayers:
            hidden_features = layer(graph, hidden_features)
            hidden_features = hidden_features.reshape(
                -1, self.f_out*self.num_heads)
        output = self.fc(hidden_features).squeeze()
        return output, hidden_features


class Generator(nn.Module):
    def __init__(self, f_in, f_out, num_heads, num_layers, feat_drop, attn_drop):
        super(Generator, self).__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.num_heads = num_heads
        self.GATlayers = nn.ModuleList()
        # self.sigmoid = nn.Sigmoid()
        self.fc = nn.Sequential(
            nn.Linear(self.f_out*self.num_heads, 93),
            nn.ReLU(),
            nn.Linear(93, 93),
            # nn.Sigmoid(),
        )
        for _ in range(num_layers):
            self.GATlayers.append(GATConv(
                f_in, f_out, num_heads, feat_drop=feat_drop, attn_drop=attn_drop, activation=nn.Sigmoid()))
            f_in = f_out*num_heads

    def forward(self, graph, features):
        hidden_features = features
        for layer in self.GATlayers:
            hidden_features = layer(graph, hidden_features)
            hidden_features = hidden_features.reshape(
                -1, self.f_out*self.num_heads)
        output = self.fc(hidden_features).squeeze()
        return output


# %%


def eval_model(datalist):
    with torch.no_grad():
        D.eval()
        G.eval()
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
            output = D(dataset.graphlist[timestep].to(device),
                       dataset.features[start:end].to(device))[0].detach().cpu()
            # less_confidence_idx = torch.where(abs(output-0.5) <= 0.2)
            # output[less_confidence_idx] = 1
            labeled_idx = torch.where(dataset.label[timestep] != 3)
            positive_idx = torch.where(dataset.label[timestep] == 1)
            positive += float(positive_idx[0].shape[0])
            true_positive += torch.sum((
                output[positive_idx[0]] < 0).float() == dataset.label[timestep][positive_idx], dtype=torch.float32)
            report_positive += torch.sum(
                (output[labeled_idx] < 0).float(), dtype=torch.float32)
            # negative acc
            negative += torch.sum(dataset.label[timestep] == 0)
            acc += torch.sum((output[labeled_idx] < 0).float() ==
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
        D.train()
        G.train()
        return f1


# %%
D = Discriminator(93, 50, 8, 6, 0, 0).to(device)
g_dim = 25
G = Generator(g_dim, 25, 4, 3, 0, 0).to(device)
optimizer_D = torch.optim.Adam(D.parameters(), lr=0.001)
optimizer_G = torch.optim.Adam(G.parameters(), lr=0.001)

# %%
# training
traininglist = range(30)
validationlist = range(30, 40)
testlist = range(40, 49)
criterion = nn.MSELoss()
plot_train = []
plot_val = []
plot_test = []
EPOCH = 1000
for epoch in range(EPOCH):
    for timestep in random.sample(traininglist, len(traininglist)):
        starttime = time.time()
        start = dataset.timestepidx[timestep]
        try:
            end = dataset.timestepidx[timestep+1]
        except:
            end = len(dataset.features)
        graph = dataset.graphlist[timestep].to(device)
        features = dataset.features[start:end].to(device)
        # train D
        for p in D.parameters():
            p.require_grad = True
        for p in G.parameters():
            p.require_grad = False
        for _ in range(5):
            true_output = D(graph, features)
            uniform_vector = torch.rand((len(features), g_dim), device=device)
            fake_features = G(graph, uniform_vector)
            fake_output = D(graph, fake_features)
            loss_d = -fake_output.mean()+true_output.mean()
            optimizer_D.zero_grad()
            loss_d.backward()
            optimizer_D.step()
        # train G
        for p in D.parameters():
            p.require_grad = False
        for p in G.parameters():
            p.require_grad = True
        uniform_vector = torch.rand((len(features), g_dim), device=device)
        fake_features = G(graph, uniform_vector)
        fake_output = D(graph, fake_features)
        loss_g = fake_output.mean()
        optimizer_G.zero_grad()
        loss_g.backward()
        optimizer_G.step()
        print(f"\r loss_d={loss_d:.4f} loss_g={loss_g:.4f}", end="")
        # eval
    if ((epoch+1) % 10 == 0):
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
# %%
