# %%
import torch
import torch.nn as nn
from torch.nn import functional as F
import pyro.distributions as dist
import pyro
from utils import DiscretizedBeta, Generate_NN, SequenceGenerator
# %%

class SimpleModel:
    def __init__(self, n=2000, l=100, r=3) -> None:
        self.n = n # Total number of generated sequences
        self.l = l # Length of each sequence
        self.r = r # Total number of modules
        self.w = 20 # Bounding length of each motif
        
        # Initializing various variables for simulation of 
        # sequences X. Loading from saved values here:
        self.phi0 = torch.load("saved_phi.pt")
        self.w0 = torch.load("saved_w.pt")
        self.pwm0 = torch.load("saved_pwm.pt")
        self.I0 = torch.load("saved_I.pt")
        self.Z0 = torch.load("saved_Z.pt")
        
        # Above variables will be used only for generating X and later
        # for evaluating the performance of our model. The model trains on
        # X alone to infer the values of various latent variables
        #self.X = SequenceGenerator(self.phi0, self.pwm0, self.w0, 
                                   #self.I0, self.Z0)
        self.X = torch.load("saved_X.pt")
        self.markov_nn = pyro.module("BG Neural model", Generate_NN())
    
    def model(self, low, size):
        # Prior sampling of motifs PWMs
        pwm_prior = dist.Dirichlet(torch.full((self.r, self.w, 4), 0.1))
        with pyro.plate("pwm loop2", self.w):
            with pyro.plate("pwm loop3", self.r):
                pwm = pyro.sample("pwm", pwm_prior)
        
        w_prior = DiscretizedBeta(5, 21, torch.tensor(2.),
                                  torch.tensor(2.)).expand(
                                      torch.Size([self.r]))
        with pyro.plate("w loop", self.r):
            w = pyro.sample("w", w_prior)
            
        # Prior sampling of module identities I
        I_prior = torch.ones([size, self.r])
        with pyro.plate("I loop", size):
            I = pyro.sample("I_{}:{}".format(low, low+size),
                        dist.Categorical(logits=I_prior))
            
        X_oh = F.one_hot(self.X[low:low+size], 4).float()
        # One hot categorical encoding of the sequences
        X_oh = torch.concat([torch.zeros(size, 2, 4), X_oh], dim=1).float()
        # Padding each sequence with zeros in the beginning

        for i in pyro.plate("Batch", size):
            i = i+low
            
            # Prior sampling of motif positions Z
            d = dist.Beta(5., 5.) 
            d = dist.TransformedDistribution(d, #type: ignore
                        [dist.transforms.AffineTransform(0., 90., 0)])
            Z = pyro.sample("Z_{}".format(i), d)
            Z = Z.round().int()
            for u in pyro.markov(range(Z), history=2):  # type:ignore
                probs = self.markov_nn(X_oh[i-low:i-low+1, u:u+2, :])[0]
                pyro.sample("X_{}_{}".format(i,u), 
                            dist.Categorical(probs),  
                            obs=self.X[i, u])
            with pyro.plate("motif"):
                pyro.sample("X_{}_m".format(i),
                            dist.Categorical(pwm[I, :w[I[i-low]], :]),
                            obs=self.X[i, Z:Z+w[I[i-low]]]
                            )
            for u in pyro.markov(range(Z+self.w, self.l),
                                 history=2): # type: ignore
                probs = self.markov_nn(X_oh[i-low:i-low+1, u:u+2, :])[0]
                pyro.sample("X_{}_{}".format(i, u),
                            dist.Categorical(probs), 
                            obs=self.X[i, u])
    
    def guide_bf(self, low, size):
        phi_param = pyro.param("phi_param", torch.ones([4, 4, 4]))
        softmax = nn.Softmax(-1)
        d = dist.Delta(softmax(phi_param))
        with pyro.plate("phi loop1", 4):
            with pyro.plate("phi loop2", 4):
                with pyro.plate("phi loop3", 4):
                    phi = pyro.sample("phi", d)
        phi_1 = torch.mean(phi, dim=0)
        phi_0 = torch.mean(phi_1, dim=0)

        pwm_param = pyro.param("pwm_param", 
                               torch.full([self.r, 20, 4], 0.1))
        d = dist.Dirichlet(pwm_param)
        with pyro.plate("pwm loop1", 4):
            with pyro.plate("pwm loop2", 20):
                with pyro.plate("pwm loop3", self.r):
                    pwm = pyro.sample("pwm", d)

        w_param = pyro.param("w_param", torch.ones([self.r, 2]))**2
        d = [DiscretizedBeta(5, 21, w_param[j, 0], w_param[j, 1])
            for j in range(self.r)]
        w = torch.empty([self.r], dtype=torch.int64).detach()
        for j in pyro.plate("w loop", self.r):
            w[torch.tensor(j)] = pyro.sample("w_{}".format(j), 
                                             d[torch.tensor(j)])

        Xoh = F.one_hot(self.X[low:low+size, :], 4).float()

        for i in pyro.plate("Batch", size):
            i = i + low
            y1 = torch.empty([self.r]).detach()
            y2 = []
            # Loop over every possible I and Z
            for j in torch.arange(self.r):
                y = torch.empty((self.l-w[j], self.l)).detach()
                for z in torch.arange(self.l-w[j]):
                    for u in torch.arange(z):
                        if u == 0:
                            y[z, u] = torch.dot(Xoh[i, u, :],
                                                phi_0)
                        elif u == 1:
                            y[z, 1] = torch.dot(Xoh[i, u, :],
                                                phi_1[0, :])
                        else:
                            y[z, u] = torch.dot(Xoh[i, u, :],
                                                phi[self.X[i, u-2], 
                                                    self.X[i, u-1], :])

                    for u in torch.arange(z, z+w[j]):
                        y[z, u] = torch.dot(Xoh[i, u, :],
                                            pwm[j, u-z, :])
                    for u in torch.arange(z+w[j], self.l):
                        y[z, u] = torch.dot(Xoh[i, u, :],
                                            phi[self.X[i, u-2], 
                                                self.X[i, u-1], :])
                y = torch.log(y)
                y2.append(y)
                y = torch.sum(y, 1)
                y1[j] = torch.logsumexp(y, 0)
            I = pyro.sample("I_{}".format(i), 
                            dist.Categorical(logits=y1))

            pyro.sample("Z_{}".format(i), 
                        dist.Categorical(logits=y2[I]))
        
# %%
obj = SimpleModel()
obj.model(100, 100)
# %%
