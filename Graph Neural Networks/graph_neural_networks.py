import torch
import matplotlib.pyplot as plt
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.profile import get_model_size
from torch.nn import Softmax, Linear
from torch_geometric.nn import global_max_pool, SAGEConv
from torch_geometric.loader import DataLoader
from torch import nn, optim

def plotting(title, x, y):
    plt.figure(figsize=(10, 5))
    plt.title(title, fontsize=20)
    plt.plot(x, y, color='#88CCEE', linewidth=3)
    plt.xlabel('Epochs', fontsize=18)
    plt.ylabel(title, fontsize=18)
    plt.grid(True)
    # plt.legend(fontsize=18)
    # plt.savefig('Graph_mnist'+ title +'.png')
    plt.show()


def accuracy(prediction, y):
    """Calculate accuracy."""
    acc = ((prediction.argmax(dim=1) == y).sum() / len(y)).item()
    print("pred", prediction.argmax(dim=1))
    print("y", y)
    return acc


class GraphCNN(torch.nn.Module):
    def __init__(self, input_neurons, hidden_channels, output_neurons):
        super(GraphCNN, self).__init__()
        # Graph-convolution section:
        self.initial_conv = SAGEConv(in_channels=input_neurons,  out_channels=hidden_channels)
        self.convolutional_layer = SAGEConv(in_channels=hidden_channels,  out_channels=hidden_channels)
        # Global mean pooling layer:
        self.pooling = global_max_pool

        # TODO: DEFINE HERE THE LAYERS OF THE MLP CLASSIFIER
        # THE LAYERS
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 3)
        # THE OPTIMIZER
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)
        # BCE LOSS
        self.criterion = nn.CrossEntropyLoss()
        # Hint: remember that the input size must match with the output size of the last graph_convolutional layer!



        # TODO: DEFINE HERE THE OPTIMIZER

        # TODO: DEFINE HERE THE CROSS ENTROPY LOSS

    def forward(self, input_data):
        x = input_data.x.float()
        edge_index = input_data.edge_index
        batch_index = input_data.batch
        # "batch_index" is a tensor that specify for each node its corresponding graph of the entire batch.
        # Example of batch index: [0,0,0,0,1,1,1,1,2,2,2,2....] The first 4 elements belong to the first graph,
        # the second 4 belong to the second graph and so on...

        x = self.initial_conv(x, edge_index)
        x = x.tanh()
        x = self.convolutional_layer(x, edge_index)
        x = x.tanh()
        x = self.convolutional_layer(x, edge_index)
        x = x.tanh()
        x = self.convolutional_layer(x, edge_index)
        x = x.tanh()

        x = self.pooling(x, batch_index)    # [n_nodes*n_graph_of_the_batch, n_features] -> [n_graph_of_the_batch, n_features]
        x= self.lin1(x)
        x = x.tanh()
        x= self.lin1(x)
        x = x.tanh()
        x= self.lin1(x)
        x = x.tanh()
        output = self.lin2(x)
        return output

    def train(self, input_data):
        # Reset gradients
        self.optimizer.zero_grad()
        # Passing the node features and the connection info
        prediction = self.forward(input_data)
        # Calculating the loss and gradients
        loss = self.criterion(prediction, input_data.y)
        loss.backward()
        # Update using the gradients
        self.optimizer.step()
        # Computing accuracy based on prediction
        #print(input_data.y.size())
        #print(prediction.size())
        acc = accuracy(prediction, input_data.y)
        #print()
        return loss.item(), acc


def main():

    # Dataset loading:
    dataset = MNISTSuperpixels(root=r"C:\Users\39349\PycharmProjects\pythonProject\Progetto_04_GNN\mnist\mnist")
    """
    print('Dataset:\t\t', dataset)
    print("number of graphs:\t\t", len(dataset))
    print("number of classes:\t\t\t", dataset.num_classes)
    print("number of node features:\t", dataset.num_node_features)
    print("number of edge features:\t", dataset.num_edge_features)

    print('-----------------------------------------------------------------------')

    single_data = dataset[0]

    print('Single data:\t\t', single_data)
    print("class:\t\t\t", single_data.y.item())
    print("node features:\t", single_data.x)
    """
    # For reasons of time, the dataset has been lightened. You can choose to use the entire data set, but it's up to you.
    num_classes = 3
    classes_set = [i for i in range(num_classes)]
    new_dataset = []
    for el in dataset:
        if el.y in classes_set:
            new_dataset.append(el)
    dataset_size = len(new_dataset)

    # Hyperparameters:
    batch_size = 128
    hidden_channels = 32
    num_epochs = 8
    input_neurons = dataset.num_node_features
    output_neurons = num_classes
    # Creating dataloaders for storing and shuffling training data:
    train_loader = DataLoader(dataset=new_dataset[:int(dataset_size * 0.8)], batch_size=batch_size, shuffle=True)

    # Print some useful information:
    print('\nTraining of a GNN on the graph Mnist dataset')
    print(44 * '-')
    print('Dataset:')
    print("Number of graphs:\t\t\t\t\t  ", dataset_size)
    print("Number of node features:\t\t\t\t  ", dataset.num_node_features)
    print("Number of edge features:\t\t\t\t  ", dataset.num_edge_features)

    # TODO: Define your model here:
    model = GraphCNN(input_neurons, hidden_channels, output_neurons)

    # Model size and number of parameters of the model:
    print(44 * '-')
    print("Model:")
    print('Model size (bytes):\t\t\t\t\t  ', get_model_size(model))
    print("Number of parameters:\t\t\t\t   ", sum(p.numel() for p in model.parameters()))
    print(44 * '-')
    # Now it's time to train the GNN:
    epochs = range(1, num_epochs+1)
    train_losses, train_accuracies = [], []
    print('Training ...')
    for epoch in epochs:
        loss_per_epoch = []
        accuracy_per_epoch = []
        for i, mini_batch in enumerate(train_loader):
            train_loss, train_acc = model.train(mini_batch)
            loss_per_epoch.append(train_loss)
            accuracy_per_epoch.append(train_acc)

        train_losses.append(sum(loss_per_epoch)/len(loss_per_epoch))
        train_accuracies.append(100*sum(accuracy_per_epoch)/len(accuracy_per_epoch))

        print(f"Ep {epoch} \t| Loss {round(train_losses[-1], 5)} \t| Accuracy: {round(train_accuracies[-1], 3)} %")

    # Here a test dataloader is created with batch_size = 1.
    test_loader = DataLoader(dataset=new_dataset[int(dataset_size * 0.8):], batch_size=1, shuffle=False)

    # TODO: Evaluate the model on test set by computing the value of the test accuracy.

    # TODO: Plot the train loss and the train accuracy over epochs (a plotting function is already present at the beginning of this code)


if __name__ == main():
    main()
