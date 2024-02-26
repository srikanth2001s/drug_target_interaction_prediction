import torch
import torch.nn as nn
from sklearn.metrics import *
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


def Model(): #defines and returns neural network model
    class GCNNet(torch.nn.Module): #This class inherits from torch.nn.Module which has several functions of neural network architecture
        def __init__(self, n_output=2, n_filters=32, embed_dim=128, drug_feat=81, prot_feat=20, output_dim=128,
                     dropout=0.2): #dropping some fraction of neurons to prevent overfitting

            super(GCNNet, self).__init__()

            self.n_output = n_output #assigns the no.of output classes
            self.conv1 = GCNConv(drug_feat, drug_feat)
            self.conv2 = GCNConv(drug_feat, drug_feat * 2)
            self.conv3 = GCNConv(drug_feat * 2, drug_feat * 4)
            self.fc_g1 = torch.nn.Linear(drug_feat * 4, 1024) #These lines define fully connected layers that follow the graph convolutional layers.
            self.fc_g2 = torch.nn.Linear(1024, output_dim)
            self.relu = nn.ReLU() #This line defines the activation function (Rectified Linear Unit, ReLU) that is used after each layer's output to introduce non-linearity.
            self.dropout = nn.Dropout(dropout)

            self.embedding_xt = nn.Embedding(prot_feat + 1, embed_dim) #embedding layers which convert discrete inputs into continuous embeddings.
            self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
            self.fc1_xt = nn.Linear(32 * 121, output_dim)

            self.fc1 = nn.Linear(2 * output_dim, 1024)
            self.fc2 = nn.Linear(1024, 512)
            self.out = nn.Linear(512, self.n_output)

        def forward(self, data): #here, input data is unpacked and passed
            x, edge_index, batch = data.x, data.edge_index, data.batch #batch contains information about which nodes belong to which graphs or batches
            target = data.target

            x = self.conv1(x, edge_index) #These lines perform graph convolutional operations
            x = self.relu(x)
            x = self.conv2(x, edge_index)
            x = self.relu(x)
            x = self.conv3(x, edge_index)
            x = self.relu(x)            #This sequence of operations extracts hierarchical graph features from the input data.

            x = global_max_pool(x, batch) #Global max pooling computes the maximum value for each feature across all nodes within each graph, effectively summarizing the most important features for each graph.
            x = self.relu(self.fc_g1(x))
            x = self.dropout(x)
            x = self.fc_g2(x)
            x = self.dropout(x)

            embedded_xt = self.embedding_xt(target) #an embedding layer that converts the discrete text data into continuous embeddings
            conv_xt = self.conv_xt_1(embedded_xt) 
            xt = conv_xt.view(-1, 32 * 121)
            xt = self.fc1_xt(xt) #1D convolutional layer (self.conv_xt_1) is applied, followed by reshaping (view) and linear transformation (self.fc1_xt)

            xc = torch.cat((x, xt), 1)
            xc = self.fc1(xc) #. The concatenated features are then passed through fully connected layers (self.fc1 and self.fc2) with ReLU activations and dropout
            xc = self.relu(xc)
            xc = self.dropout(xc)
            xc = self.fc2(xc)
            xc = self.relu(xc)
            xc = self.dropout(xc)
            out = self.out(xc) # the output is passed through the last fully connected layer (self.out), which produces the model's predictions.
            return out

    model = GCNNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()
    return model, optimizer, loss_function #The model, optimizer, and loss_function are then returned as a tuple from the Model function.

def TrainTheModel(num_itr=2):
    model, optimizer, loss_function = Model() #It calls the Model() function to initialize the neural network model (model), optimizer (optimizer), and loss function (loss_function)
    train_loss = torch.zeros(num_itr)
    valid_loss = torch.zeros(num_itr)
    train_acc = torch.zeros(num_itr)
    valid_acc = torch.zeros(num_itr) #initialising 4 tensors
    Final_acc = 0
    Final_model = None
    for itr in range(num_itr):
        model.train() #sets the model in training mode
        for z, data in enumerate(train_loader): #z is just an enumeration variable, and data represents batch of training data.
            optimizer.zero_grad() #clears gradients of prev. parameters
            out = model(data) #computes model prediction 
            loss = loss_function(out, data.y) #It calculates the loss between the model's predictions (out) and the true labels (data.y)
            loss.backward() #It computes the gradients of the loss with respect to the model's parameters
            optimizer.step() #Updates the model's parameters using the optimizer based on the obtained gradients.
            train_loss[itr] += loss.item() #training loss
            pred = out.max(1)[1] #It calculates the predicted class labels by taking class with highest probability
            train_acc[itr] += pred.eq(data.y).sum().item() #It computes the training accuracy by comparing the predicted labels (pred) with the true labels (data.y)
        train_loss[itr] /= len(train_loader.dataset) #training loss and accuracy for the current epoch are normalized by dividing by the total number of training samples in the dataset
        train_acc[itr] /= len(train_loader.dataset)

        model.eval() #sets in evaluation mode
        for data in val_loader: #It computes the model's predictions (out) for the current batch of validation data.
            out = model(data)
            loss = loss_function(out, data.y)
            valid_loss[itr] += loss.item()
            pred = out.max(1)[1]
            valid_acc[itr] += pred.eq(data.y).sum().item()
        valid_loss[itr] /= len(val_loader.dataset)
        valid_acc[itr] /= len(val_loader.dataset)
        if (valid_acc[itr] > Final_acc):
            Final_model = model
        print(f'Iteration: {itr:02d}, Training Loss: {train_loss[itr]:.5f}, Training Acc: {train_acc[itr]:.5f}, Validation Acc: {valid_acc[itr]:.5f}')
    return Final_model, train_loss, valid_loss, train_acc, valid_acc

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
    test_acc = test_acc + pred.eq(data.y).sum().item()
    precision = precision + precision_score(data.y, pred)
    recall = recall + recall_score(data.y, pred)

precision = precision/len(test_loader.dataset) #normalising
recall = recall/len(test_loader.dataset)
test_acc = test_acc/len(test_loader.dataset)

f1_score = 2*(precision*recall)/(precision + recall)

print(f'Precision: {precision:.5f}')
print(f'Recall: {recall:.5f}')
print("Accuracy (Testing) : ", test_acc*100,"%")
print(f'F1 Score: {f1_score:.5f}')
