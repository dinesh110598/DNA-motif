# %%
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist
from scipy import integrate, special

# %%
def generate_ground_truth(n=1000, l=100, r=3, w0=5, w1=20):
    d = dist.Dirichlet(torch.full([4, 4, 4], 2.))
    phi = d.sample()

    pwm = torch.randint(4, size=(r, w1))

    d = dist.Binomial(w1-w0, 0.5)
    w = d.sample(torch.Size([r])).long() + w0

    d = dist.Categorical(probs=torch.tensor([0.4, 0.4, 0.2]))
    I = d.sample(torch.Size([n]))

    d = dist.Beta(2., 2.)
    Z = torch.stack([torch.round(d.sample()*(l - w[I[i]])).long()
                     for i in range(n)])

    X = SequenceGenerator(phi, pwm, w, I, Z)

    vars = {
        "phi": phi,
        "pwm": pwm,
        "w": w,
        "I": I,
        "Z": Z
    }
    torch.save(vars, "saved_latents.pt")
    torch.save(X, "saved_X.pt")

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

# function that integrates the Beta distribution's pdf
def beta_integral(x1, x2, a: float, b: float):
    def beta_pdf(x):
        return (x**(a-1))*((1-x)**(b-1))/special.beta(a, b)

    return integrate.quad(beta_pdf, x1, x2)[0]

class DiscretizedBeta(dist.Distribution):
    def __init__(self, low: int, high: int, a: float, b: float):
        beta = dist.Beta(a, b)
        self.low = low
        self.high = high

        prob = torch.empty(size=[high-low+1], dtype=torch.float32)
        x = np.linspace(0., 1., high-low+2)
        for i in torch.arange(high-low):
            prob[i] = beta_integral(x[i], x[i+1], a, b)

        self.categorical = dist.Categorical(prob)

        super().__init__(beta.batch_shape, beta.event_shape,
                         validate_args=False)

    def sample(self, sample_shape=torch.Size()):
        return self.categorical.sample(sample_shape) + self.low

    def log_prob(self, value):
        return self.categorical.log_prob(value-self.low)
    
class DiscreteLogistic(dist.Distribution):
    def __init__(self, low: int, high: int,
                 m: torch.Tensor, s: torch.Tensor):
        self.low = low
        d = torch.arange(low+0.5, high, 1)
        d1 = torch.cat([torch.tensor([low-1000]),
                        d]).unsqueeze(0)
        d2 = torch.cat([d, torch.tensor([high+1000])
                        ]).unsqueeze(0)
        m = m.unsqueeze(1)
        s = s.unsqueeze(1)
        probs = torch.sigmoid((d2-m)/s)-torch.sigmoid((d1-m)/s)
        self.cat = dist.Categorical(probs)
        super().__init__(self.cat.batch_shape, 
                         self.cat.event_shape, 
                         validate_args=False)
        
    def sample(self, sample_shape=torch.Size()):
        return self.cat.sample(sample_shape) + self.low
    
    def log_prob(self, value):
        return self.cat.log_prob(value - self.low)
    
def logit_normal_lp(x: torch.Tensor):
    y = (-x.log().sum() 
         - 0.05*((x[:-1]/x[-1]).log()**2).sum())
    return y
# %%
