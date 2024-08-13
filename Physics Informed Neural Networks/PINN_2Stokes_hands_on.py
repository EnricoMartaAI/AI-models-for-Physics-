import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch import optim

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.hidden_layer1 = nn.Linear(1, 5)
        self.hidden_layer2 = nn.Linear(5, 5)
        self.hidden_layer3 = nn.Linear(5, 5)
        self.hidden_layer4 = nn.Linear(5, 5)
        self.hidden_layer5 = nn.Linear(5, 5)
        self.output_layer = nn.Linear(5, 1)
        self.tau = torch.rand(1, requires_grad=True)
        self.B = torch.rand(1, requires_grad=True)

        params = list(self.parameters())
        params.append(self.tau)
        params.append(self.B)
        self.optimizer = optim.Adam(params, lr=0.01)
        self.criterion = nn.MSELoss()


    def forward(self, t):
        inputs = torch.cat([t], axis=1)
        layer1_out = torch.sigmoid(self.hidden_layer1(inputs))
        layer2_out = torch.sigmoid(self.hidden_layer2(layer1_out))
        layer3_out = torch.sigmoid(self.hidden_layer3(layer2_out))
        layer4_out = torch.sigmoid(self.hidden_layer4(layer3_out))
        layer5_out = torch.sigmoid(self.hidden_layer5(layer4_out))
        # For regression, no activation is used in output layer
        output = self.output_layer(layer5_out)
        return output

    def diff_eq(self, t):
        # Prediction on velocities
        v = self.forward(t)
        # First Order Derivative dv/dt
        v_t = torch.autograd.grad(v.sum(), t, create_graph=True)[0]
        f = v_t + v/self.tau - self.B
        return f

    def train_step(self, diff_eq_times, dataset_times, dataset_vel):
        self.optimizer.zero_grad()
        # Here we estimate the differential equation loss
        f_out = self.diff_eq(diff_eq_times)  # output of f(x,t)
        zeros_diff_eq = torch.zeros(size=(500, 1), requires_grad=False, dtype=torch.float32)
        loss_diff_eq = self.criterion(f_out, zeros_diff_eq)

        loss_data = self.criterion(dataset_vel, self.forward(dataset_times))
        loss_total = loss_data + loss_diff_eq
        # Here we compute the backpropagation
        loss_total.backward()
        self.optimizer.step()
        return loss_total.item()

def main():
    dataset_ = np.load(r"C:\Users\39349\PycharmProjects\pythonProject\Progetto_05_PINN\STOKES_dataset.npz")
    t_dataset, v_dataset = dataset_["times"], dataset_["velocities"]
    tau_ = 2.0
    times_min, times_max = 0.0, 2 * tau_
    # Dataset in tensor format
    times_data = torch.tensor(t_dataset, requires_grad=True, dtype=torch.float32)
    velocities = torch.tensor(v_dataset, requires_grad=True, dtype=torch.float32)
    model = PINN()
    n_epochs = 30000
    batch_size = 500
    for epoch in range(n_epochs):
        # Differential Equation data
        times_eq = times_max + (times_min - times_max) * torch.rand(size=(batch_size, 1),
                                                                    requires_grad=True, dtype=torch.float32)
        loss = model.train_step(diff_eq_times=times_eq, dataset_times=times_data, dataset_vel=velocities)
        print(epoch+1, "Training Loss:", loss)

    # Data to plot
    with torch.autograd.no_grad():
        t_test = np.arange(0, 2*tau_, 1/200).reshape(800, 1)
        v_true = np.array([2 * np.exp(-t / tau_) + 4 for t in t_test])
        # use .detach().numpy() method to convert torch.tensor to numpy
        v_test = model.forward(torch.tensor(t_test, dtype=torch.float32))

    # Plotting
    fig, ax = plt.subplots()
    ax.plot(t_test, v_test, label="Model")
    ax.plot(t_test, v_true, '--', label="True Values")
    ax.plot(t_dataset, v_dataset, 'o', label="Dataset")

    ax.legend(loc='upper right', shadow=True)
    ax.set_xlabel('Time (s)', weight='bold')
    ax.set_ylabel('Velocity (m/s)', weight='bold')
    ax.set_title('Stokes Dynamics', fontsize=22, weight='bold')
    plt.show()

    print("B=",model.B)
    print("Tau = ",model.tau)

if __name__ == "__main__":
    main()
