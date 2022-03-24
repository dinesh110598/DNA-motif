# %%
import torch
import pyro
from pyro.optim import ClippedAdam
from torch.distributions import constraints
import pyro.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pyro.infer import SVI, TraceEnum_ELBO
from latent_nn import PhiNet, GammaNet
# %%

class BGCluster(nn.Module):
    def __init__(self):
        super().__init__()
        # Global constants
        self.n = 1000
        self.l = 100
        self.r = 3
        self.batch = 50
        self.phi_net = PhiNet(self.r)
        self.gamma_net = GammaNet(self.r)
        pyro.module("Gamma Net", self.gamma_net)

    def model(self, X):
        # Sampling phi
        with pyro.plate("Modules", self.r):
            with pyro.plate("Markov 1", 4):
                with pyro.plate("Markov 2", 4):
                    phi = pyro.sample("phi", dist.Dirichlet(
                        torch.ones(4)
                    ))
                    #phi.shape = [4, 4, r, 4]
                    
                    
            gamma_wts = pyro.sample("gamma_wts", dist.Gamma(
                1. / self.r, 1.
            ))
        
        with pyro.plate("Batch", X.shape[0]):
            gamma = pyro.sample("gamma", dist.Dirichlet(
                gamma_wts
            ))
            I = pyro.sample("I", dist.Categorical(
                gamma), infer={"enumerate": "parallel"})
            
            pyro.sample("X", dist.Categorical(
                    phi[X[:, :-2], X[:, 1:-1], #type:ignore 
                        I.unsqueeze(1)]
                ).to_event(1), obs=X[:, 2:]) #type:ignore

    def guide(self, X):
        phi_post = pyro.param(
            "phi_post", lambda: torch.ones([4, 4, self.r,
                                            4]),
            constraint= constraints.greater_than(0.5)) #type:ignore
        gamma_wts_post = pyro.param(
            "gamma_wts_post", lambda: torch.ones(self.r)
        )
        
        with pyro.plate("Modules", self.r):
            with pyro.plate("Markov 1", 4):
                with pyro.plate("Markov 2", 4):
                    phi = pyro.sample("phi", dist.Dirichlet(
                        phi_post
                    ))
                    #phi.shape = [4, 4, r, 4]
            
            pyro.sample("gamma_wts", dist.Gamma(
                gamma_wts_post, 1.
            ))
        
        with pyro.plate("Batch", X.shape[0]):
            Xoh = F.one_hot(X, -1).permute((0, 2, 1)).float()
            # Dirichlet prior 𝑝(𝜃|𝛼) is replaced by a logistic-normal distribution,
            # where μ and Σ are the encoder network outputs
            gamma = self.gamma_net(Xoh)
            pyro.sample(
                "gamma", dist.Delta(gamma, event_dim=1))
            
    def infer(self, X):
        Xoh = F.one_hot(X, -1).permute((0, 2, 1)).float()
        loggamma_mu, _ = self.gamma_net(Xoh)
        gamma = F.softmax(loggamma_mu, -1)
        
        return gamma
            
def train_loop(cluster, epochs):
    device = torch.device("cpu")
    cluster.to(device)
    pyro.clear_param_store()
    
    elbo = TraceEnum_ELBO(max_plate_nesting=3)
    opt = ClippedAdam({"lr": 1e-3})
    svi = SVI(cluster.model, cluster.guide, opt,
              loss=elbo)
    
    X = torch.load("saved_X.pt")
    X = X.to(device)
    seq_generator = DataLoader(X, cluster.batch, False)
    for epoch in range(epochs):
        for X_batch in seq_generator:
            loss = svi.step(X_batch)
            print(loss)
    
# %%
cluster_model = BGCluster()
train_loop(cluster_model, 50)
# %%
X = torch.load("saved_X.pt")
X = X.to(torch.device("cuda:0"))
gamma = cluster_model.infer(X)
I1 = torch.argmax(gamma, -1)
# %%
lat_dict = torch.load("saved_latents.pt")
phi0 = lat_dict["phi"]
I0 = lat_dict['I']
# %%
