# %%
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributions as dist
from pyro.infer import SVI, Trace_ELBO
import pyro.optim as optim
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
        """Model function describes prioir probability distibutions
        for latent variables phi, pwm, w, I and Z, as well as the
        likelihood of observed data X.
        - phi: Dirichlet prior with conc=2. Though we can expect max probability
        for equally-distributed, other probabilities are still allowed
        - pwm: Dirichlet prior with conc=0.1. We expect the probability to
        favour definite values of BP with less uncertainty
        - w: DiscretizedBeta prior
        - gamma: Dirichlet prior with conc=0.1
        - I: Categorical prior with parameters given by gamma
        - Z:  
        Args:
            low (int), size (int): Describes indices for mini-batch sampling of
            the observed variables X.
        """
        phi_prior = dist.Dirichlet(torch.full([4, 4, 4], 2.))

        with pyro.plate("phi loop2", 4):
            with pyro.plate("phi loop3", 4):
                phi = pyro.sample("phi", phi_prior)
        phi_1 = torch.mean(phi, dim=0)
        phi_0 = torch.mean(phi_1, dim=0)

        pwm_prior = dist.Dirichlet(torch.full([self.r, self.w, 4], 0.1))
        with pyro.plate("pwm loop2", self.w):
            with pyro.plate("pwm loop3", self.r):
                pwm = pyro.sample("pwm", pwm_prior)
        # This is our guess on what possible values of PWM are more likely
        # Dirichlet(0.1) means extreme values close to 0 and 1 are favoured

        w_prior = DiscretizedBeta(5, self.w, 1., 1.)
        w = torch.empty([self.r], dtype=torch.int64)
        for j in pyro.plate("w loop", self.r):
            w[torch.tensor(j)] = pyro.sample("w_{}".format(j),
                                            w_prior)

        for i in pyro.plate("Batch", size):
            i = i + low
            
            gamma_prior = dist.Dirichlet(torch.full([self.r], 0.1))
            gamma = pyro.sample("gamma_{}".format(i), gamma_prior)
            
            I_prior = dist.Categorical(gamma)
            I = pyro.sample("I_{}".format(i), I_prior)

            Z_prior = dist.Categorical(logits=torch.ones([self.l - w[I]]))
            Z = pyro.sample("Z_{}".format(i), Z_prior)
        
            for u in pyro.markov(range(self.l), history=2):  # type: ignore
                if (u < Z) or (u > Z+w[I]):
                    if u == 0:
                        pyro.sample("X_{},0".format(i),
                                    dist.Categorical(probs=phi_0),
                                    obs=self.X[i, u])
                    elif u == 1:
                        pyro.sample("X_{},1".format(i),
                                    dist.Categorical(probs=phi_1[self.X[u-1]]),
                                    obs=self.X[i, u])
                    else:
                        pyro.sample("X_{},{}".format(i, u),
                                    dist.Categorical(
                                        probs=phi[self.X[i, u-2], self.X[i, u-1]]),
                                    obs=self.X[i, u])
                else:
                    pyro.sample("X_{},{}".format(i, u),
                                dist.Categorical(probs=pwm[I, u-Z]),
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
                               torch.full([self.r, self.w, 4], 0.1))
        d = dist.Dirichlet(pwm_param)
        with pyro.plate("pwm loop2", self.w):
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
                            y[z, u] = torch.dot(Xoh[i-low, u, :],
                                                phi_0)
                        elif u == 1:
                            y[z, 1] = torch.dot(Xoh[i-low, u, :],
                                                phi_1[0, :])
                        else:
                            y[z, u] = torch.dot(Xoh[i-low, u, :],
                                                phi[self.X[i, u-2], 
                                                    self.X[i, u-1], :])

                    for u in torch.arange(z, z+w[j]):
                        y[z, u] = torch.dot(Xoh[i-low, u, :],
                                            pwm[j, u-z, :])
                    for u in torch.arange(z+w[j], self.l):
                        y[z, u] = torch.dot(Xoh[i-low, u, :],
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
            
    def train_bf(self, epochs=10, batch_size=100):
        # Clear out other trainable parameters in the current REPL/ipython kernel
        # session 🔥
        pyro.clear_param_store()
        opt = torch.optim.Adam
        scheduler = optim.StepLR({"optimizer": opt, "step_size": 250, "gamma": 0.2,  # type: ignore
                                "optim_args": {"lr": 0.001, "betas": (0.9, 0.999)}})
        svi = SVI(self.model, self.guide_bf, scheduler, loss=Trace_ELBO())

        for epoch in range(epochs):
            steps_per_epoch = self.n//batch_size
            low = 0
            for step in range(steps_per_epoch):
                svi.step(low, batch_size)
            scheduler.step()
            
    def guide_neural2(self, low, size):
        """Guide function that describes parameterized probability distibutions
        for latent variables phi, pwm, w, I and Z.
        - phi: Delta posterior, whose single support is trainable and 
        softmax-activated
        - pwm: Though we used a Dirichlet prior the posterior will be a
        Delta, restricting the actual value of pwm to a single support.
        - w: 
        Args:
            low (int), size (int): Describes indices for mini-batch sampling of
            the observed variables X.
        """
        phi_param = pyro.param("phi_param", torch.ones([4, 4, 4]))
        softmax = nn.Softmax(-1)
        d = dist.Delta(softmax(phi_param))
        with pyro.plate("phi loop1", 4):
            with pyro.plate("phi loop2", 4):
                with pyro.plate("phi loop3", 4):
                    phi = pyro.sample("phi", d)
        
        pwm_param = pyro.param("pwm_param",
                               torch.full([self.r, self.w, 4], 0.1))
        d = dist.Dirichlet(pwm_param)
        with pyro.plate("pwm loop2", self.w):
            with pyro.plate("pwm loop3", self.r):
                pwm = pyro.sample("pwm", d)
# %%