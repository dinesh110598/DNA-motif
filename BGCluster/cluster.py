# %%
import torch
from torch import nn
from torch.nn import functional as F
from torch import distributions as dist
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader
from latent_nn import PhiNet, GammaNet
# %%

class BGCluster(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Global constants
        self.n = 1000
        self.l = 100
        self.r = 3
        self.batch = 50
        self.avg = 100
        
        # Guide (neural) approximators
        self.phi_par = nn.parameter.Parameter(
            torch.full([self.r, 4, 4, 4], 0.5))
        self.gamma_net = GammaNet(self.r)
        
    def ELBO_loss(self, X):
        avg = self.avg
        r = self.r
        
        phi_param = self.phi_par
        d = dist.Dirichlet(phi_param)
        phi = d.sample(torch.Size([avg]))
        
        y = phi[:, :, X[:, :-2], X[:, 1:-1], X[:, 2:]]
        # Mean instead of sum for normalized
        # likelihoods
        y2 = y.log().mean(-1)
        y2 = torch.permute(y2, (0, 2, 1))
        # y2.shape = [avg, batch, r]
        gamma = F.softmax(y2, -1)
        # gamma.shape = [avg, batch, r]
        
        log_likhood = (y2*gamma).mean((1,2))
        log_likhood, _ = torch.sort(log_likhood, 
                                    descending=True)
        #log_likhood = log_likhood[:5].mean()
        
        #print(-log_likhood.item())
        return -log_likhood
    
    def infer(self, X):
        avg = self.avg
        r = self.r

        phi_mu = self.phi_par
        phi = phi_mu
        phi = F.softmax(phi, -1)

        y = phi[:, X[:, :-2], X[:, 1:-1], X[:, 2:]]
        # Mean instead of sum for normalized
        # likelihoods
        y2 = y.log().mean(-1)
        y2 = torch.permute(y2, (1, 0))
        # y2.shape = [batch, r]
        gamma = F.softmax(y2, -1)
        # gamma.shape = [batch, r]
        
        return phi, gamma
    

def train_loop(model: BGCluster, epochs):
    X = torch.load("saved_X.pt")
    assert X.shape[0] % model.batch == 0
    seq_generator = DataLoader(X, model.batch, True)

    opt = torch.optim.Adam(
        [{'params': model.phi_par,
          'lr': 0.001},
         {'params': model.gamma_net.parameters()}
         ], lr=0.001)
    schedule = torch.optim.lr_scheduler.StepLR(
        opt, 50, 0.5
    )
    for epoch in range(epochs):
        for X_bat in seq_generator:
            #tepoch.set_description(f'Epoch {epoch}')
            opt.zero_grad()
            loss = model.ELBO_loss(X_bat)
            loss.backward()
            opt.step()
            print(loss.item())
            #tepoch.set_postfix(loss=loss.item())
        schedule.step()

# %%
X = torch.load("saved_X.pt")
model = BGCluster()
logp = model.ELBO_loss(X[:model.batch])
logp[0]
#train_loop(model, 50)
# %%
X = torch.load("saved_X.pt")
phi1, gamma1 = model.infer(X)
I1 = torch.argmax(gamma1, -1)
# %%
lat_dict = torch.load("saved_latents.pt")
phi0 = lat_dict["phi"]
I0 = lat_dict['I']
# %%
