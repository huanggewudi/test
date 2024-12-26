import torch
import torch.nn as nn


class Gate(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Gate, self).__init__()
        self.linear = nn.Linear(input_size + hidden_size, hidden_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), dim=-1)
        output = self.linear(combined)
        output = self.sigmoid(output)
        return output


class GruCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GruCell, self).__init__()
        self.reset_gate = Gate(input_size, hidden_size)
        self.update_gate = Gate(input_size, hidden_size)
        self.linear = nn.Linear(input_size + hidden_size, hidden_size)
        self.tanh = nn.Tanh()

    def forward(self, input_data, hidden):
        reset = self.reset_gate(input_data, hidden)
        update = self.update_gate(input_data, hidden)
        hidden_hat = self.linear(torch.cat((input_data, hidden * reset), dim=-1))
        hidden_hat = self.tanh(hidden_hat)
        output = hidden * (1 - update) + hidden_hat * update
        return output
