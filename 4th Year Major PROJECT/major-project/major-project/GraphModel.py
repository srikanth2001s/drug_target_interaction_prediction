import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import*
from sklearn.model_selection import train_test_split
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, global_max_pool
import warnings


dataset = torch.load('preprocessed_dataset.pt')
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.2, random_state=42)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, shuffle=True, drop_last=True)

warnings.filterwarnings("ignore")

def Model():
    class GCNNet(torch.nn.Module): #nf labels #nf filters #dimensionality of embed  #                                      #dropping fraction of neurons to prevent overfitting
        def __init__(self, n_output=2, n_filters=32, embed_dim=128, num_features_xd=81, num_features_xt=25, output_dim=128, dropout=0.2):

            super(GCNNet, self).__init__()

            self.n_output = n_output
            self.conv1 = GCNConv(num_features_xd, num_features_xd)
            self.conv2 = GCNConv(num_features_xd, num_features_xd*2)
            self.conv3 = GCNConv(num_features_xd*2, num_features_xd * 4)
            self.fc_g1 = torch.nn.Linear(num_features_xd*4, 1024)
            self.fc_g2 = torch.nn.Linear(1024, output_dim)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)

            self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
            self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
            self.fc1_xt = nn.Linear(32*121, output_dim)

            self.fc1 = nn.Linear(2*output_dim, 1024)
            self.fc2 = nn.Linear(1024, 512)
            self.out = nn.Linear(512, self.n_output)

        def forward(self, data):
            x, edge_index, batch = data.x, data.edge_index, data.batch
            target = data.target

            x = self.conv1(x, edge_index)
            x = self.relu(x)
            x = self.conv2(x, edge_index)
            x = self.relu(x)

            x = self.conv3(x, edge_index)
            x = self.relu(x)
            x = global_max_pool(x, batch)       
            x = self.relu(self.fc_g1(x))
            x = self.dropout(x)
            x = self.fc_g2(x)
            x = self.dropout(x)

            embedded_xt = self.embedding_xt(target)
            conv_xt = self.conv_xt_1(embedded_xt)
            xt = conv_xt.view(-1, 32 * 121)
            xt = self.fc1_xt(xt)

            xc = torch.cat((x, xt), 1)
            xc = self.fc1(xc)
            xc = self.relu(xc)
            xc = self.dropout(xc)
            xc = self.fc2(xc)
            xc = self.relu(xc)
            xc = self.dropout(xc)
            out = self.out(xc)
            return out
    model = GCNNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()
    return model, optimizer, loss_function


def TrainTheModel(num_epochs=2):
    model, optimizer, loss_function = Model()
    train_loss = torch.zeros(num_epochs)
    valid_loss = torch.zeros(num_epochs)
    train_acc = torch.zeros(num_epochs)
    valid_acc = torch.zeros(num_epochs)
    best_acc = 0
    best_model = None
    for epoch in range(num_epochs):
        model.train()
        for _, data in enumerate(train_loader):
            optimizer.zero_grad()
            out = model(data)
            loss = loss_function(out, data.y)
            loss.backward()
            optimizer.step()
            train_loss[epoch] += loss.item()
            pred = out.max(1)[1]
            train_acc[epoch] += pred.eq(data.y).sum().item()
        train_loss[epoch] /= len(train_loader.dataset)
        train_acc[epoch] /= len(train_loader.dataset)

        model.eval()
        for data in val_loader:
            out = model(data)
            loss = loss_function(out, data.y)
            valid_loss[epoch] += loss.item()
            pred = out.max(1)[1]
            valid_acc[epoch] += pred.eq(data.y).sum().item()
        valid_loss[epoch] /= len(val_loader.dataset)
        valid_acc[epoch] /= len(val_loader.dataset)
        if (valid_acc[epoch] > best_acc):
            best_model = model
        print('Epoch: {:03d}, Train Loss: {:.5f}, Train Acc: {:.5f}, Val Loss: {:.5f}, Val Acc: {:.5f}'.format(
            epoch, train_loss[epoch], train_acc[epoch], valid_loss[epoch], valid_acc[epoch]))
    return best_model, train_loss, valid_loss, train_acc, valid_acc


model, train_loss, valid_loss, train_acc, valid_acc = TrainTheModel()

torch.save(model.state_dict(), 'dataset.pt')

model.load_state_dict(torch.load('dataset.pt'))
model.eval()
test_acc = 0
precision = 0
recall = 0
for data in test_loader:
    out = model(data)
    pred = out.max(1)[1]
    test_acc += pred.eq(data.y).sum().item()
    precision += precision_score(data.y, pred)
    recall += recall_score(data.y, pred)
test_acc /= len(test_loader.dataset)
precision /= len(test_loader.dataset)
recall /= len(test_loader.dataset)

print('Precision: {:.5f}'.format(precision))
print('Recall: {:.5f}'.format(recall))
print('Test Accuracy: {:.5f}'.format(test_acc))

f1_score = 2*(precision*recall)/(precision+recall)
print('F1 Score: {:.5f}'.format(f1_score))

