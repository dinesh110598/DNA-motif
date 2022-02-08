# %%
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributions as dist
import numpy as np
from utils import SequenceGenerator, DiscretizedBeta
# %%

def generate_ground_truth(n=1000, l=100, r=3, w0=5, w1=20):
    d = dist.Dirichlet(torch.full([4, 4, 4], 2.))
    phi = d.sample()
    
    pwm = torch.randint(4, size=(r, w1))
    
    d = dist.Binomial(w1-w0, 0.5)
    w = d.sample(torch.Size([r])).long() + w0
    
    d = dist.Categorical(probs=torch.tensor([0.4, 0.4, 0.2]))
    I = d.sample(torch.Size([n]))
    
    d = dist.Beta(0.4, 0.4)
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
# %%
class PWM_Net(nn.Module):
    def __init__(self, r, w1) -> None:
        super().__init__()
        self.r = r
        self.w1 = w1  
        self.seq = nn.Sequential(
            nn.Linear(1, 32, False),
            nn.Mish(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 32),
            nn.Mish(),
            nn.Dropout(0.2),
            nn.Linear(32, 4*r*w1)
        )
        self.sfmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        self.seq(x)
        x = torch.squeeze(x).view(self.r, self.w1, 4)
        return self.sfmax(x)
    
class Gated_Block(nn.Module):
    def __init__(self, in_ch, in_len, ker_size) -> None:
        super().__init__()
        # Build a block of gated convolution
        # operations conditioned on phi
        self.conv1 = nn.Conv1d(in_ch, 32, ker_size)
        self.cond = nn.Linear(64, in_len-(ker_size-1))
        self.act1 = nn.Sigmoid()
        self.act2 = nn.Tanh()
        self.conv2 = nn.Conv1d(16, 32, 1)
        self.pool = nn.MaxPool1d(2)
        
    def forward(self, x, phi):
        x = self.conv1(x)
        x += self.cond(phi).unsqueeze(1)
        x1, x2 = torch.split(x, 2, dim=1)
        x1 = self.act1(x1)
        x2 = self.act2(x2)
        x = torch.mul(x1, x2)
        x = self.conv2(x)
        x = F.mish(x)
        return self.pool(x)

class Z_Net(nn.Module):
    def __init__(self, r) -> None:
    # We assume l=100 explicitly
        super().__init__()
        self.layers = [
            Gated_Block(r, 100, 5),
            Gated_Block(16, 48, 5),
            Gated_Block(16, 22, 3)
        ]
        self.conv1 = nn.Conv1d(16, 16, 1)
        self.class_pred = nn.Conv1d(16, 1, 1)
        self.conv2 = nn.Conv1d(16, 16, 1)
        self.Z_pred = nn.Conv1d(16, 1, 1)
    
    def forward(self, x, phi):
        for layer in self.layers:
            x = layer(x, phi)
        class_logits = F.mish(self.conv1(x))
        class_logits = self.class_pred(class_logits)
        class_probs = F.sigmoid(class_logits)
        
        Z_delta = F.mish(self.conv2(x))
        Z_delta = self.Z_pred(Z_delta)
        Z_delta = F.tanh(Z_delta)
        return class_probs.squeeze(), Z_delta.squeeze()
        

class SingleMofifModel(nn.Module):
    def __init__(self) -> None:
        self.l = 100 #Length of each seq
        self.r = 3 #Number of clusters/modules
        self.w0 = 5 #Min motif length
        self.w1 = 20 #Max motif length
        self.batch = 50 #Training batch size
        
        #Model parameters in the guide
        self.phi_param = nn.parameter.Parameter(
            torch.zeros([4, 4, 4]))
        self.pwm_net = PWM_Net(self.r, self.w1)
        self.w_net = nn.Sequential(
            nn.Linear(1, 16),
            nn.Mish(),
            nn.Linear(16, 16),
            nn.Mish(),
            nn.Dropout(0.2),
            nn.Linear(16, self.r),
            nn.Sigmoid()
        )
        self.z_net = Z_Net(self.r)
        self.gamma_net = nn.Sequential(
            nn.Linear(3, 3),
            nn.Mish(),
            nn.Linear(3, 3),
            nn.Softmax(-1)
        )
        
        #Prior distributions of various latent variables
        self.phi_pr = dist.Dirichlet(torch.full([4,4,4], 2.))
        self.pwm_pr = dist.Dirichlet(torch.full([self.r, self.w1, 
                                                 4], 0.1))
        self.w_pr = DiscretizedBeta(self.w0, self.w1, 2., 2.)
        self.gamma_pr = dist.Dirichlet(torch.full([self.batch, 
                                                   self.r], 0.1))
    def Z_pr(self, w, I):
        return DiscretizedBeta(0, self.l - w[I],
                                2., 2.)
        
    def likelihood(self, phi, pwm, w, gamma, Z, X):
        phi_0 = torch.mean(phi, dim=(0, 1))
        phi_1 = torch.mean(phi, dim=0)
        p = torch.empty([self.batch, self.r, self.l])
        for i in torch.arange(self.batch):
            for I in torch.arange(self.r):
                for u in torch.arange(Z[i]):
                    if u==0:
                        p[i,I,0] = phi_0[X[i,0]]
                    elif u==1:
                        p[i,I,1] = phi_1[X[i,0],
                                    X[i,1]]
                    else:
                        p[i,I,u] = phi[X[i,u-2],
                                    X[i,u-1],
                                    X[i, u]]
                
                for u in torch.arange(Z[i], Z[i]+w[I[i]]):
                    p[i,I,u] = pwm[I[i], u-Z[i], X[i,u]]
                    
                for u in torch.arange(Z[i]+w[I[i]], self.l):
                    p[i,I,u] = phi[X[i, u-2],
                                X[i, u-1],
                                X[i, u]]
        
        lp = torch.log(p)
        return torch.sum(lp*gamma.unsqueeze(-1))
    
    def prior_logits(self, phi, pwm, w, gamma, Z):
        phi_lp = self.phi_pr.log_prob(phi)
        pwm_lp = self.pwm_pr.log_prob(pwm)
        w_lp = self.w_pr.log_prob(w)
        gamma_lp = self.gamma_pr.log_prob(gamma)
        
        Z_lp = torch.empty([self.batch, self.r])        
        for i in range(self.batch):
            for I in range(self.r):
                Z_lp[i, I] = self.Z_pr(w,I).log_prob(Z[i])
        
        return (torch.sum(phi_lp) +
                torch.sum(pwm_lp) +
                torch.sum(w_lp) +
                torch.sum(gamma_lp) +
                torch.sum(gamma*Z_lp))
    
    def guide_sample(self, X):
        phi = F.softmax(self.phi_param)
        pwm = self.pwm_net(torch.ones([1,1], requires_grad=False))
        w = self.w_net(torch.ones([1, 1], requires_grad=False))[0]
        w = torch.round(w*(self.w1 - self.w0) + self.w0).int()
        # Constructing one-hot X for finding pwm probability
        # at every position.
        Xoh = torch.permute(F.one_hot(X), (0, 2, 1))
        # Xoh.shape = (batch, 4, l)
        pwm_diag = torch.permute(pwm, (0, 2, 1))
        f = lambda t : torch.diag_embed(t, dim1=0, dim2=2)
        pwm_diag = [f(pwm_diag[j, :, :w[j]])
                    for j in range(self.r)]
        with torch.no_grad():
            y = [F.conv1d(Xoh, pwm_diag[j], padding='same') 
                 for j in range(self.r)]
        y = torch.stack([torch.log(y[j]).mean(1)
                         for j in range(self.r)], 1)
        phi_flat = phi.view([1, -1])
        class_pr, Z_del = self.z_net(y, phi_flat)
        # Interpreting class outputs for prediction of best
        # Z and gamma
        interval = torch.stack([torch.linspace(0,100-w[j],11)[:-1]
                                 for j in range(self.r)])
        spacing = interval[1]-interval[0]
        gamma = torch.empty([self.batch, 3, self.r])
        Z = torch.empty([self.batch, 3], dtype=torch.int32)
        probs = torch.empty([self.batch, 3])
        for i in range(self.batch):
            Z1 = Z_del[i]
            cp = class_pr[i]
            
            ind = torch.argsort(cp)[-3:]
            probs[i, :] = cp[ind]/cp[ind].sum()
            Z1 = interval[ind] + (Z1[ind]*spacing)
            Z1 = Z1.round()
            Z[i, :] = Z1
            g2 = torch.empty([3, self.r])
            for j in range(self.r):
                t = (w[j]-2+(w[j]%2))//2
                # This expn won't add up to 1 😧
                gamma[i, :, j] = y[i, j, t+Z1]
                #So we will get this operated by a softmax
                #activated network!
        gamma = self.gamma_net(gamma)
        
        return phi, pwm, w, probs, Z, gamma
    
    def guide_logits(self, probs):
        #The guide probababilities mostly delta distributed
        #except for gamma and Z
        gamma_lp = torch.sum(probs.log()*probs)
        return 2*gamma_lp #Additional term for Z is the same!
    
    def elbo_loss(self, X: torch.Tensor):
        phi, pwm, w, probs, Z, gamma = self.guide_sample(X)
        P_X = torch.stack([self.likelihood(phi, pwm, w, 
                                 gamma[:, k], Z[:, k], X)
                 for k in torch.arange(3)])
        P_X = torch.dot(P_X, probs)
        P_prior = torch.stack([self.prior_logits(phi, pwm, w,
                                        gamma[:,k], Z[:,k])
                               for k in torch.arange(3)])
        P_prior = torch.dot(P_prior, probs)
        P_guide = self.guide_logits(probs)
        
        return P_X + P_prior + P_guide

def train_loop(model: SingleMofifModel, epochs):
    X = torch.load("saved_X.pt")
    X.requires_grad = False
    assert X.shape[0]%model.batch == 0
    X = X.view([-1, model.batch, X.shape[1]])
    opt = torch.optim.Adam(model.parameters(), 0.1)
    schedule = torch.optim.lr_scheduler.StepLR(opt, 30, gamma=0.2)
    
    for epoch in range(epochs):
        for step in range(X.shape[0]):
            loss = model.elbo_loss(X[step])
            loss.backward()
            opt.step()
        schedule.step()
        
        