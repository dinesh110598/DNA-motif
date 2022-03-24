import torch
from torch import nn
from torch.nn import functional as F
from torch import distributions as dist
import numpy as np

class MultiLinear(nn.Module):
    def __init__(self, in_features, out_features,
                 array_len):
        super().__init__()
        i = in_features
        o = out_features
        l = array_len
        self.weight = nn.parameter.Parameter(
            torch.randn(l, o, i)/np.sqrt(i)
        )
        self.bias = nn.parameter.Parameter(
            torch.zeros([l, o])
        )

    def forward(self, x: torch.Tensor):
        # It's matrix multiplications in
        # parallel!
        x = x.unsqueeze(1)
        x = ((x*self.weight).sum(-1) +
             self.bias)
        return x

class PhiNet2(nn.Module):
    def __init__(self, r):
        super().__init__()
        l1 = r*4*4
        self.r = r
        self.backbone = nn.Sequential(
            MultiLinear(1, 64, l1),
            nn.Softplus(),
            MultiLinear(64, 64, l1),
            nn.Softplus(),
            nn.Dropout(0.2)
        )
        self.lin_mu = MultiLinear(64, 4, l1)
        self.lin_var = MultiLinear(64, 4, l1)
        self.ones = torch.ones([l1, 1])
        
    def forward(self):
        h = self.backbone(self.ones)
        loc = self.lin_mu(h)
        scale = self.lin_var(h)
        scale = (0.5*scale).exp()
        loc = loc.view([self.r, 4, 4, 4])
        scale = scale.view([self.r, 4, 4, 4])
        return loc, scale
    
class PhiNet(nn.Module):
    def __init__(self, r):
        super().__init__()
        self.phi_mu = nn.parameter.Parameter(
            torch.zeros([r, 4**3])
        )
        self.bnmu = nn.BatchNorm1d(50)
        
    def forward(self, gamma):
        logphi_mu = torch.tensordot(gamma, self.phi_mu,
                                 ([-1], [0]))#type:ignore
        logphi_mu = self.bnmu(logphi_mu).view(
            gamma.shape[:-1]+(4, 4, 4))
        
        return F.softmax(logphi_mu, -1)
        
    
class GammaNet(nn.Module):
    def __init__(self, r) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(4, 32, 5),
            nn.Mish(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 1, 5),
            nn.Mish(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(22, 32),
            nn.Mish(),
            nn.Linear(32, r)
        )
        
    def forward(self, Xoh):
        h = self.backbone(Xoh)
        return F.softmax(h, -1)
