import math
import torch
import torch.nn as nn
from scipy.special import hermite
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class PINN(nn.Module):
    def __init__(self, energy):
        super(PINN, self).__init__()
        self.hidden_layer1 = nn.Linear(1, 50)
        self.hidden_layer2 = nn.Linear(50, 40)
        self.hidden_layer3 = nn.Linear(40, 30)
        self.hidden_layer4 = nn.Linear(30, 20)
        self.hidden_layer5 = nn.Linear(20, 10)
        self.output_layer = nn.Linear(10, 1)

        ## Diff. Eq. Params ##
        self.E = energy
        self.w = torch.rand(1, requires_grad=True)
        # TODO: instantiate omega and set the constant to be predicted
        params = list(self.parameters())
        params.append(self.w)
        self.optimizer = optim.Adam(params, lr=0.01)
        self.criterion = torch.nn.MSELoss()

    def forward(self, x):
        inputs = torch.cat([x], axis=1)
        layer1_out = torch.sigmoid(self.hidden_layer1(inputs))
        layer2_out = torch.sigmoid(self.hidden_layer2(layer1_out))
        layer3_out = torch.sigmoid(self.hidden_layer3(layer2_out))
        layer4_out = torch.sigmoid(self.hidden_layer4(layer3_out))
        layer5_out = torch.sigmoid(self.hidden_layer5(layer4_out))
        # For regression, no activation is used in output layer
        output = self.output_layer(layer5_out)
        return output

    def diff_eq(self, x):
        ## Prediction ##
        psi = self.forward(x)
        ## TODO: evaluate the first and second order derivatives of psi
        psi_x = torch.autograd.grad(psi.sum(), x, create_graph=True)[0]
        psi_xx = torch.autograd.grad(psi_x.sum(), x, create_graph=True)[0]
        ## TODO: return the implicit function for the differential equation
        return -0.5 * psi_xx + 0.5 * psi * ((self.w * x) ** 2) - self.E * psi


    def train_step(self, diff_eq_x, dataset_x, amplitudes):

        self.optimizer.zero_grad()

        psi = self.forward(dataset_x)
        loss_data = self.criterion(psi, amplitudes)

        # Here we estimate the differential equation loss
        f_out = self.diff_eq(diff_eq_x) # output of f(x,t)
        zeros_diff_eq = torch.zeros(size=(500, 1), requires_grad=False, dtype=torch.float32)
        loss_diff_eq = self.criterion(f_out, zeros_diff_eq)

        # TODO: sum the two losses (loss_total)
        loss_total = loss_data + loss_diff_eq
        # Here we compute the backpropagation
        loss_total.backward()
        self.optimizer.step()
        return loss_total


def wavefunc(x, n, omega):
    H = hermite(int(n))
    return (2 / np.pi)**(1/4) * np.exp(-omega / 2 * x ** 2) / np.sqrt(2**n * math.factorial(n)) * H(x)



##############
## TRAINING ##
##############
def main():
    ## Parameters ##
    energy = 2.75
    x_min, x_max = -6, 6

    ## Dataset in tensor format ##
    dataset_ = np.load("SCHRODINGER_dataset.npz")
    x_data, psi_dataset = dataset_["x_data"], dataset_["psi_data"]

    x_data = torch.tensor(x_data, requires_grad=True, dtype=torch.float32)
    psi_dataset = torch.tensor(psi_dataset, requires_grad=True, dtype=torch.float32)

    model = PINN(energy)

    ## TODO: Train the model, define number of epochs
    n_epochs = 1000
    for epoch in range(n_epochs):
        ## Differential Equation data ##
        x_eq = (x_min - x_max) * torch.rand(size=(500, 1), requires_grad=True, dtype=torch.float32) + x_max
        loss = model.train_step(diff_eq_x=x_eq, dataset_x=x_data, amplitudes=psi_dataset)
        print(epoch+1, "Training Loss:", loss.item())

    with torch.autograd.no_grad():
        omega = model.w.item()
        n_pred = round((energy/omega) - 0.5)
        print("Learned omega: {}".format(omega))
        print("Energy level: {}".format(n_pred))

        ## Data to plot ##
        x_test = np.arange(x_min, x_max, 1/200).reshape((x_max-x_min)*200, 1)
        psi_true = np.array([wavefunc(x, n_pred, omega) for x in x_test])
        ## TODO: predict the psi function from x in order to plot them
        psi_test = model.forward(torch.tensor(x_test, dtype=torch.float32))



    ## Plotting ##
    fig, ax = plt.subplots()
    ax.plot(x_test, psi_test.detach().numpy(), label="Model")
    ax.plot(x_test, psi_true, '--', label="True Values")
    ax.plot(x_data.detach().numpy(), psi_dataset.detach().numpy(), 'o', label="Dataset")

    ax.legend(loc='upper right', shadow=True)
    ax.set_xlabel('Space coords (a.u.)', weight='bold')
    ax.set_ylabel('Probability amplitude', weight='bold')
    ax.set_title('Wavefunction', fontsize=22, weight='bold')
    plt.show()


if __name__=="__main__":
    main()
