# %%
# import packages
from dgl.nn.pytorch import GATConv
from sklearn.metrics import roc_auc_score
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
        label = df["class"]
        label = label.str.replace("2", '0')
        label = label.str.replace("unknown", '2')
        label = label.astype(np.float).to_numpy()
        self.totalnode = len(label)
        # 1 bad 0 good 3 unknown
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
                self.graphlist[i].ndata['feat'] = self.features[self.timestepidx[i]:self.timestepidx[i+1]]
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


class GAT(nn.Module):
    def __init__(self, f_in, f_out, num_heads, num_layers, feat_drop, attn_drop):
        super(GAT, self).__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.num_heads = num_heads
        self.GATlayers = nn.ModuleList()
        self.fc_class = nn.Sequential(
            nn.Linear(self.f_out*self.num_heads, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            # nn.Sigmoid(),
        )
        self.fc_confidence = nn.Sequential(
            nn.Linear(self.f_out*self.num_heads, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid(),
        )
        for i in range(num_layers):
            self.GATlayers.append(GATConv(
                f_in, f_out, num_heads, feat_drop=feat_drop, attn_drop=attn_drop, activation=nn.ReLU()))
            f_in = f_out*num_heads

    def forward(self, graph, features):
        hidden_features = features
        for layer in self.GATlayers:
            hidden_features = layer(graph, hidden_features)
            hidden_features = hidden_features.reshape(
                -1, self.f_out*self.num_heads)
        predict = self.fc_class(hidden_features)
        confidence = self.fc_confidence(hidden_features)

        return predict, confidence


# %%


def eval_model(datalist):
    with torch.no_grad():
        model.eval()
        positive = 0
        true_positive = 0
        report_positive = 0
        confidence_true_positive = 0
        confidence_report_positive = 0
        negative = 0
        acc = 0
        auc = 0
        for timestep in datalist:
            start = dataset.timestepidx[timestep]
            try:
                end = dataset.timestepidx[timestep+1]
            except:
                end = len(dataset.features)
            output, confidence = model(dataset.graphlist[timestep].to(device),
                                       dataset.features[start:end].to(device))
            confidence = confidence.detach().cpu()
            output = output.detach().cpu()
            output = torch.sigmoid(output).detach().cpu()
            labels = dataset.label[timestep]
            labeled_idx = torch.where(labels != 2)

            positive_idx = torch.where(labels == 1)
            positive += float(positive_idx[0].shape[0])
            true_positive += torch.sum(torch.round(
                output[positive_idx[0]]).squeeze() == labels[positive_idx].squeeze(), dtype=torch.float32)
            # negative acc
            report_positive += torch.sum(torch.round(
                output[labeled_idx]), dtype=torch.float32)
            negative += torch.sum(labels == 0)
            acc += torch.sum(torch.round(output[labeled_idx]).squeeze() ==
                             labels[labeled_idx].squeeze(), dtype=torch.float32)
            auc += roc_auc_score(labels[labeled_idx].squeeze().int(),
                                 1-confidence[labeled_idx].squeeze())
            # by confidence score
            threshold = 0.8
            less_confidence_idx = torch.where(confidence < threshold)
            confidence_prediction = torch.zeros_like(confidence)
            confidence_prediction[less_confidence_idx] = 1
            confidence_true_positive += torch.sum(
                confidence_prediction[positive_idx].squeeze() == labels[positive_idx].squeeze(), dtype=torch.float32)
            confidence_report_positive += torch.sum(
                confidence_prediction, dtype=torch.float32)
        recall = true_positive/positive
        confidence_recall = confidence_true_positive/positive
        try:
            precision = true_positive/report_positive
            f1 = 2/(1/precision+1/recall)
            confidence_precision = confidence_true_positive/confidence_report_positive
            confidence_f1 = 2/(1/confidence_precision+1/confidence_recall)
        except:
            precision = 0
            f1 = 0
            confidence_precision = 0
            confidence_f1 = 0
        print(
            f"acc={acc/(positive+negative):.4f} precision={precision:.4f} recall={recall:.4f} f1={f1:.4f} auc={auc/len(datalist):.4f} c_pre={confidence_precision:.4f} c_rec={confidence_recall:.4f} c_f1={confidence_f1:.4f}")
        model.train()


# %%
model = GAT(93, 40, 16, 4, 0.3, 0.3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# %%
# training
traininglist = range(30)
validationlist = range(30, 40)
testlist = range(40, 49)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.])).to(device)
# criterion = nn.BCELoss()
plot_train = []
plot_val = []
plot_test = []
EPOCH = 500
for epoch in range(EPOCH):
    for timestep in random.sample(traininglist, len(traininglist)):
        starttime = time.time()
        start = dataset.timestepidx[timestep]
        try:
            end = dataset.timestepidx[timestep+1]
        except:
            end = len(dataset.features)
        output, confidence = model(dataset.graphlist[timestep].to(device),
                                   dataset.features[start:end].to(device))
        labeled_idx = torch.where(dataset.label[timestep] != 2)
        # labels = dataset.label[timestep][labeled_idx].to(device)
        labels = torch.tensor(
            dataset.label[timestep][labeled_idx], device=device)
        #labels[labeled_idx] = 0
        output_ = confidence[labeled_idx].squeeze()*output[labeled_idx].squeeze() + \
            (1-confidence[labeled_idx].squeeze())*labels.squeeze()
        # output_ = confidence.squeeze()*output.squeeze() + \
        #    (1-confidence.squeeze())*labels.squeeze()
        loss_t = criterion(output_, labels)
        loss_c = -torch.log(confidence[labeled_idx]).mean()/3
        #loss_c = -torch.log(confidence+1e-6).mean()/3
        loss = loss_t+loss_c
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'\r loss_t={loss_t:.4f} loss_c={loss_c:.4f}', end='')
        # eval
    if ((epoch+1) % 20 == 0):
        print('')
        print(epoch+1)
        print("train")
        eval_model(traininglist)
        print("val")
        eval_model(validationlist)
        print("test")
        eval_model(testlist)
# %%
print("train")
eval_model(traininglist)
print("val")
eval_model(validationlist)
print("test")
eval_model(testlist)
# %%
torch.save(model, "./models/confidence_model.bin")

# %%
model = torch.load("./models/test_model.bin")
# %%
