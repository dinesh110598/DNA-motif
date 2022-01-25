import numpy as np
import torch
import torch.nn as nn
import pyro.distributions as dist
import pyro
from scipy import integrate, special

def SequenceGenerator(phi: torch.Tensor, pwm: torch.Tensor, w, I: torch.Tensor,
                      Z: torch.Tensor, n: int = 1000, l: int = 100, r: int = 3):

    X = np.empty((n, l), dtype=int)
    g = np.random.default_rng()
    phi2 = phi.numpy()

    for i in range(n):

        for u in range(Z[i]):  # Markov sampling for bg DNA
            if u == 0:
                prob = np.mean(phi2, axis=(0, 1))
                # Marginalizing for positions 1,2
                X[i, 0] = g.choice(4, p=prob)
            elif u == 1:
                prob = np.mean(phi2, axis=0)[X[i, 0], :]
                X[i, 1] = g.choice(4, p=prob)
            else:
                prob = phi2[X[i, u-2], X[i, u-1], :]
                X[i, u] = g.choice(4, p=prob)

        X[i, Z[i]:Z[i]+w[I[i]]] = pwm[I[i], :w[I[i]]]  # Motif at pos Z[i]

        for u in range(Z[i]+w[I[i]], l):  # Markov sampling for bg DNA
            prob = phi2[X[i, u-2], X[i, u-1], :]
            X[i, u] = g.choice(4, p=prob)

    return torch.from_numpy(X)  # Declare X as torch tensor


def Generate_NN(depth=2, width=16):
    layers = []
    assert depth >= 0
    layers.append(nn.Flatten())
    if depth == 0:
        layers.append(nn.Linear(8, 4))
    elif depth > 0:
        layers.append(nn.Linear(8, width))
        layers.append(nn.ReLU())
        for _ in range(depth-1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(width, 4))

    layers.append(nn.Softmax(dim=-1))
    return nn.Sequential(*layers)

# function that integrates the Beta distribution's pdf


def beta_integral(x1, x2, a: torch.Tensor, b: torch.Tensor):
    def beta_pdf(x):
        return (x**(a-1))*((1-x)**(b-1))/special.beta(a, b)

    return integrate.quad(beta_pdf, x1, x2)[0]


class DiscretizedBeta(dist.TorchDistribution):
    def __init__(self, low, high, conc1, conc2):
        self.beta = dist.Beta(conc1, conc2)
        self.low = low
        self.high = high
        self.a = conc1.detach().numpy()
        self.b = conc2.detach().numpy()

        prob = torch.empty(size=[high-low], dtype=torch.float32)
        x = np.linspace(0., 1., high-low+1)
        for i in torch.arange(high-low):
            prob[i] = beta_integral(x[i], x[i+1], self.a, self.b)

        self.categorical = dist.Categorical(prob)

        super().__init__(self.beta.batch_shape, self.beta.event_shape,
                         validate_args=False)

    def sample(self, sample_shape=torch.Size()):
        return self.categorical.sample(sample_shape) + self.low

    def log_prob(self, value):
        return self.categorical.log_prob(value-self.low)
    
    