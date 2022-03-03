# %%
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributions as dist
from tqdm.notebook import tqdm
from utils import DiscreteLogistic
from latent_nn import PWM_Net, Z_Net
# %%

class SingleMofifModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l = 100  # Length of each seq
        self.r = 3  # Number of clusters/modules
        self.w = 10  # Fixed motif length
        self.batch = 50  # Training batch size
        self.avg = 5  # Averaging size for exp values
        #over certain latents
        
        #Model parameters in the guide
        #self.phi_param = nn.parameter.Parameter(
        #torch.zeros([4, 4, 4]))
        self.pwm_net = PWM_Net(self.r, self.w)
        # We'll act z_net after obtaining y
        self.z_net = nn.Sequential(
            nn.Conv1d(1, 32, 1),
            nn.Mish(),
            nn.Conv1d(32, 32, 1),
            nn.Mish(),
            nn.Conv1d(32, 1, 1),
            nn.Sigmoid()
        )
        # And then use the output of above
        # to predict gamma
        self.gamma_net = nn.Sequential(
            nn.Linear(5, 32),
            nn.Mish(),
            nn.Linear(32, 1)
        )
        
    def ELBO_loss(self, X):
        pwm = self.pwm_net(torch.ones([1,1]))
        w = self.w
        
        Xoh = torch.permute(F.one_hot(X), (0, 2, 1)).float()
        # Xoh.shape = (batch, 4, l)
        pwm_diag = torch.permute(pwm, (0, 2, 1))
        def f(t): return torch.diag_embed(t, dim1=0, dim2=2)
        pwm_diag = [f(pwm_diag[j]) for j in range(self.r)]
        pwm_diag = torch.cat(pwm_diag, 0)
        y = F.conv1d(Xoh, pwm_diag).log()
        y = y.view([self.batch, self.r, w, self.l-w+1])
        y = torch.mean(y, 2, keepdim=True)
        
        # Predicting 5 best values of Z across batch, r
        # dims
        y1 = y.view([self.batch*self.r,1,-1])
        Zs = self.z_net(y1).squeeze()
        Zs = Zs.view([self.batch,self.r,-1])
        Z_sort, Z = torch.sort(Zs, -1, descending=True)
        Z_sc = Z_sort[:, :, :5]
        
        # Predict gamma using Z_sc as input
        gamma = self.gamma_net(Z_sc).squeeze()
        gamma = F.softmax(gamma, -1)
        
        # The likelihood will be computed only
        # for the best-scored Z, for each I
        Z = Z[:, :, 0:1]
        
        # y computes the pwm log-likelihood for
        # every possible Z, we gather the
        # components for given Z
        y = y.squeeze()*w
        ll = torch.gather(y, -1, Z).squeeze()
        # Multiply with gamma and sum
        log_likhood = (gamma*ll).sum()
        log_guide = (gamma*gamma.log()).sum()
        
        return -log_likhood + log_guide
    
    def infer(self, X):
        pwm = self.pwm_net(torch.ones([1, 1]))
        w = self.w

        Xoh = torch.permute(F.one_hot(X), (0, 2, 1)).float()
        # Xoh.shape = (batch, 4, l)
        pwm_diag = torch.permute(pwm, (0, 2, 1))
        def f(t): return torch.diag_embed(t, dim1=0, dim2=2)
        pwm_diag = [f(pwm_diag[j]) for j in range(self.r)]
        pwm_diag = torch.cat(pwm_diag, 0)
        y = F.conv1d(Xoh, pwm_diag).log()
        y = y.view([self.batch, self.r, w, self.l-w+1])
        y = torch.mean(y, 2, keepdim=True)

        # Predicting 5 best values of Z across batch, r
        # dims
        y1 = y.view([self.batch*self.r, 1, -1])
        Zs = self.z_net(y1).squeeze()
        Zs = Zs.view([self.batch, self.r, -1])
        Z_sort, Z = torch.sort(Zs, -1, descending=True)
        Z_sc = Z_sort[:, :, :5]

        # Predict gamma using Z_sc as input
        gamma = self.gamma_net(Z_sc).squeeze()
        gamma = F.softmax(gamma, -1)
        gamma_ind = torch.argmax(gamma, -1)

        # The likelihood will be computed only
        # for the best-scored Z, for each I
        Z = Z[torch.arange(self.batch),
              gamma_ind, 0]
        
        return pwm, gamma, Z

def train_loop(model: SingleMofifModel, epochs):
    X = torch.load("saved_X.pt")
    X.requires_grad = False
    assert X.shape[0] % model.batch == 0
    X = X.view([-1, model.batch, X.shape[1]])
    opt = torch.optim.Adam(model.parameters(), lr=0.02)
    #schedule = torch.optim.lr_scheduler.StepLR(opt, 30,
                                               #gamma=0.2)
    for epoch in range(epochs):
        with tqdm(range(X.shape[0]),
                  unit="batch") as tepoch:
            for step in tepoch:
                tepoch.set_description(f'Epoch {epoch}')
                opt.zero_grad()
                loss = model.ELBO_loss(X[step])
                loss.backward()
                opt.step()
                tepoch.set_postfix(loss=loss.item())
            #schedule.step()
# %%
model = SingleMofifModel()
train_loop(model, 200)
# %%
X = torch.load("saved_X.pt")
pwm, gamma, Z = model.infer(X[:model.batch])
# %%
