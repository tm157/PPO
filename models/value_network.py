import torch.nn as nn
import torch.nn.functional as F

class Value(nn.Module):
    def __init__(self, num_inputs):
        super().__init__()
        self.affine1 = nn.Linear(num_inputs, 100)
        self.affine2 = nn.Linear(100, 100)
        self.value_head = nn.Linear(100, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))

        state_value = self.value_head(x)

        return state_value
