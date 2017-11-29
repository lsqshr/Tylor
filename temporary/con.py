import torch
from torch import nn
import torch.autograd as autograd


class Recog(nn.Module):

    def __init__(self, input_dim, hidden_dim, tagset_size):
        super(Recog, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.hidden2code = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim).cuda()),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim).cuda()))

    def forward(self, x):
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        out = self.hidden2code(lstm_out.view(1, -1))
        return out
