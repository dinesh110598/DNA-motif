from torch import nn
from torch.nn import functional as F
import torch

class PWM_Net(nn.Module):
    def __init__(self, r, w1) -> None:
        super().__init__()
        self.r = r
        self.w1 = w1
        self.seq = nn.Sequential(
            nn.Linear(1, 32, False),
            nn.Mish(),
            nn.Linear(32, 32),
            nn.Mish(),
            nn.Dropout(0.2),
            nn.Linear(32, 4*r*w1)
        )
        self.sfmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.seq(x)
        x = torch.squeeze(x).view([self.r, self.w1, 4])
        return self.sfmax(x)


class Gated_Block(nn.Module):
    def __init__(self, in_ch, in_len, ker_size) -> None:
        super().__init__()
        # Build a block of gated convolution
        # operations conditioned on phi
        self.conv1 = nn.Conv1d(in_ch, 32, ker_size)
        self.act1 = nn.Sigmoid()
        self.act2 = nn.Tanh()
        self.conv2 = nn.Conv1d(16, 16, 1)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        x = self.conv1(x)
        x1, x2 = torch.split(x, 16, dim=1)
        x1 = self.act1(x1)
        x2 = self.act2(x2)
        x = torch.mul(x1, x2)
        x = self.conv2(x)
        x = F.mish(x)
        return self.pool(x)

# This network outputs must be conditioned
# on the module identity I, that means I can
# make the input as y!
class Z_Net(nn.Module):
    def __init__(self) -> None:
        # We assume l=100 explicitly
        super().__init__()
        self.layers = [
            Gated_Block(1, 100, 5),
            Gated_Block(16, 48, 5),
            Gated_Block(16, 22, 5)
        ]
        self.conv1 = nn.Conv1d(16, 16, 1)
        self.class_pred = nn.Conv1d(16, 1, 1)
        # Predicts fg vs bg
        self.conv2 = nn.Conv1d(16, 16, 1)
        self.Z_pred = nn.Conv1d(16, 1, 1)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        class_logits = F.mish(self.conv1(x))
        class_logits = self.class_pred(class_logits)
        class_probs = self.softmax(class_logits)

        Z_delta = F.mish(self.conv2(x))
        Z_delta = self.Z_pred(Z_delta)
        Z_delta = torch.tanh(Z_delta)
        return (class_probs.squeeze(),
                Z_delta.squeeze())
