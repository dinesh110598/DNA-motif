# %%
import torch
import torch.nn as nn
import pyro.distributions as dist
import pyro
import numpy as np
# %%
# This model assumes one motif (of fixed length) per module 
# and every sequence having an unknown module identity

def SequenceGenerator(phi: torch.Tensor, motifs: torch.Tensor, I: torch.Tensor,
                      Z: torch.Tensor, n: int=1000, l: int=100, r: int=3):

    X = np.empty((n, l), dtype=int)
    g = np.random.default_rng()
    phi2 = phi.numpy()

    for i in range(n):
        prob = np.mean(phi2, axis=(0, 1))
        X[i, 0] = g.choice(4, p=prob)  # Marginalizing for positions 1,2
        prob = np.mean(phi2, axis=0)[X[i, 0], :]
        X[i, 1] = g.choice(4, p=prob)
        
        for u in range(2, Z[i]):  # Markov sampling for bg DNA
            prob = phi2[X[i, u-2], X[i, u-1], :]
            X[i, u] = g.choice(4, p=prob)
        
        X[i, Z[i]:Z[i]+10] = motifs[I[i], :] # Motif at pos Z[i]
        
        for u in range(Z[i]+10, n):  # Markov sampling for bg DNA
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


class SimpleModel:
    def __init__(self, n=1000, l=100, r=3) -> None:
        self.n = n # Total number of generated sequences
        self.l = l # Length of each sequence
        self.r = r # Total number of modules
        self.w = 20 # Bounding length of each motif
        
        # Initializing various variables for simulation of 
        # sequences X. Loading from saved values here:
        self.phi = torch.load("saved_phi.pt")
        self.motifs = torch.load("saved_motifs.pt")
        self.I = torch.load("saved_I.pt")
        self.Z = torch.load("saved_Z.pt")
        
        # Above variables will be used only for generating X and later
        # for evaluating the performance of our model. The model trains on
        # X alone to infer the values of various latent variables
        self.X = SequenceGenerator(self.phi, self.motifs, 
                                   self.I, self.Z)
        self.markov_nn = Generate_NN()
        self.anchors = self.generate_anchors()
    
    def model(self, low, size):
        MarkovModule = pyro.module("BG Neural model", self.markov_nn)
        # Prior sampling of motifs PWMs
        m_prior = torch.full((self.r, self.w, 4), 0.1)
        motifs = pyro.sample("motifs", dist.Dirichlet( #type:ignore
            m_prior))
        for i in pyro.plate("Batch", size):
            i = i+low
            X_oh = nn.functional.one_hot(self.X[i:i+1], 4).float()  # type: ignore
            # One hot categorical encoding of the sequences
            X_oh = torch.concat([torch.zeros(1, 2, 4), X_oh], dim=1).float()
            # Padding each sequence with zeros in the beginning
            
            # Prior sampling of module identities I
            I_prior = torch.ones(self.r)
            I = pyro.sample("I_{}".format(i), 
                            dist.Categorical(logits=I_prior)) #type:ignore
            # Prior sampling of motif positions Z
            d = dist.Beta(5., 5.) # type: ignore
            d = dist.TransformedDistribution(d, #type: ignore
                        [dist.transforms.AffineTransform(0., 90., 0)])
            Z = pyro.sample("Z_{}".format(i), d)
            Z = Z.round().int()
            for u in pyro.markov(range(Z), history=2):  # type:ignore
                probs = MarkovModule(X_oh[:, u:u+2, :])[0]
                pyro.sample("X_{}_{}".format(i,u), 
                            dist.Categorical(probs),  # type: ignore
                            obs=self.X[i, u])
            with pyro.plate("motif"):
                pyro.sample("X_{}_m".format(i, Z, Z+self.w),
                            dist.Categorical(motifs[I]), # type: ignore
                            obs=self.X[i, Z:Z+self.w]
                            )
            for u in pyro.markov(range(Z+self.w, self.l),
                                 history=2): # type: ignore
                probs = MarkovModule(X_oh[:, u:u+2, :])[0]
                pyro.sample("X_{}_{}".format(i, u),
                            dist.Categorical(probs),  # type: ignore
                            obs=self.X[i, u])
    
    def guide(self, low, size):
        concs = pyro.param("Concentrations",
                           torch.full((self.r, self.w, 4), 0.1),
                           dist.constraints.positive)
        pyro.sample("motifs", dist.Dirichlet(concs))  # type: ignore
        for i in pyro.plate("Batch", size):
            i = i+low
            I_p = neural_Module1(self.X[i])
            pyro.sample("I_{}".format(i),
                            dist.Categorical(logits=I_p))  # type:ignore
            
            pos = neural_Module2(self.X[i])
            d = dist.Delta(pos)
            pyro.sample("Z_{}".format(i), d)
            
            
            
    def generate_anchors(self, anchor_len=[5, 10, 15], 
                         feature_stride=10):
        """Generates anchors for use with RPN network

        Args:
            anchor_len (list, optional): Set of allowed widths for
            anchors. Defaults to [5, 10, 15, 20].
            feature_stride (int, optional): Spacing of allowed starting
            positions for anchors starting from 0. Defaults to 10
        """
        # The anchors will be characterized by two indices: Z and w
        Z = torch.arange(0, self.l, feature_stride)
        w = torch.tensor(anchor_len)
        anchors = torch.cartesian_prod(Z, w) #That's a neat function
        # to achieve the goal!
        return anchors
    
    def RPN_net(self):
        pass
            
        