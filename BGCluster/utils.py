# %%
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist

# %%
def SequenceGenerator(phi: torch.Tensor, I: torch.Tensor,
                      n: int = 1000, l: int = 100, r: int = 3):
    # Here, phi is a [r, 4, 4, 4] tensor
    X = np.empty((n, l), dtype=int)
    g = np.random.default_rng()
    phi2 = phi.numpy()

    for i in range(n):
        j = I[i]
        for u in range(l):  # Markov sampling for bg DNA
            if u == 0:
                prob = np.mean(phi2[j, :, :, :],
                               axis=(0, 1))
                # Marginalizing for positions 1,2
                X[i, 0] = g.choice(4, p=prob)
            elif u == 1:
                prob = np.mean(phi2[j, :, :, :],
                               axis=0)[X[i, 0], :]
                X[i, 1] = g.choice(4, p=prob)
            else:
                prob = phi2[j, X[i, u-2], X[i, u-1], :]
                X[i, u] = g.choice(4, p=prob)

    return torch.from_numpy(X)


def generate_ground_truth(n=1000, l=100, r=3):
    d = dist.Dirichlet(torch.full([r, 4, 4, 4], 0.5))
    phi = d.sample()

    d = dist.Categorical(probs=torch.tensor([0.4, 0.4, 0.2]))
    I = d.sample(torch.Size([n]))

    X = SequenceGenerator(phi, I)

    latents = {
        "phi": phi,
        "I": I,
    }
    torch.save(latents, "saved_latents.pt")
    torch.save(X, "saved_X.pt")
# %%

