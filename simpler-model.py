# %%
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributions as dist
from tqdm.notebook import tqdm
from utils import DiscreteLogistic
from latent_nn import PWM_Net
from torch.utils.data import DataLoader
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
        
        self.pwm_pr = dist.Dirichlet(torch.full([self.r,
                                                self.w,
                                                4], 0.2))
        
    def pwm_pr_lp(self, x: torch.Tensor):
        #x2, _ = torch.sort(x, -1, True)
        lp = (x**3).sum(-1)
        return lp
    
    def ELBO_loss(self, X, lk_loss):
        w = self.w
        # 🤯 I can make a convolutional
        # net that predicts each component of
        # pwm independently!!!
        pwm = self.pwm_net()
        print(pwm[0, :2])
        # An entropy term for pwm
        pwm_pr_lp = self.pwm_pr_lp(pwm).mean()
        
        Xoh = torch.permute(F.one_hot(X), (0,2,1)).float()
        # Xoh.shape = (batch, 4, l)
        pwm_diag = torch.permute(pwm, (0, 2, 1))
        def f(t): return torch.diag_embed(t,dim1=0,dim2=2)
        pwm_diag = [f(pwm_diag[j]) for j in range(self.r)]
        pwm_diag = torch.cat(pwm_diag, 0)
        y1 = F.conv1d(Xoh, pwm_diag).log()
        # y1.shape = [batch, r*w, l-w+1]
        y = y1.view([self.batch, self.r, w, self.l-w+1])
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
        y = y.squeeze()
        ll = torch.gather(y, -1, Z).squeeze()
        # Multiply with gamma and mean
        log_likhood = (gamma*ll).mean()*3
        log_guide = (gamma*gamma.log()).mean()*3
        
        # I should probably analyze and fix the
        # relative weights of terms in the loss
        #print(f"likhood: {-log_likhood.item()}", end="\t")
        #print("log_guide: ", log_guide.item())
        #print(f"pwm_prior: {-pwm_pr_lp.item()}")
        return (-log_likhood + log_guide - pwm_pr_lp,
                -log_likhood.detach())
    
    def infer(self, X):
        pwm = self.pwm_net()
        w = self.w
        batch = X.shape[0]

        Xoh = torch.permute(F.one_hot(X), (0,2,1)).float()
        # Xoh.shape = (batch, 4, l)
        pwm_diag = torch.permute(pwm, (0, 2, 1))
        def f(t): return torch.diag_embed(t,dim1=0,dim2=2)
        pwm_diag = [f(pwm_diag[j]) for j in range(self.r)]
        pwm_diag = torch.cat(pwm_diag, 0)
        y = F.conv1d(Xoh, pwm_diag).log()
        y = y.view([batch, self.r, w, self.l-w+1])
        y = torch.mean(y, 2, keepdim=True)

        # Predicting 5 best values of Z across batch, r
        # dims
        y1 = y.view([batch*self.r, 1, -1])
        Zs = self.z_net(y1).squeeze()
        Zs = Zs.view([batch, self.r, -1])
        Z_sort, Z = torch.sort(Zs, -1, descending=True)
        Z_sc = Z_sort[:, :, :5]

        # Predict gamma using Z_sc as input
        gamma = self.gamma_net(Z_sc).squeeze()
        gamma = F.softmax(gamma, -1)
        gamma_ind = torch.argmax(gamma, -1)

        # The likelihood will be computed only
        # for the best-scored Z, for each I
        Z = Z[torch.arange(batch),
              gamma_ind, 0]
        
        return pwm, gamma, Z

def train_loop(model: SingleMofifModel, epochs):
    X = torch.load("saved_X.pt")
    assert X.shape[0] % model.batch == 0
    seq_generator = DataLoader(X, model.batch, True)
    
    opt = torch.optim.Adam(
        [{'params': model.pwm_net.parameters(),
          'lr': 0.1},
         {'params': model.z_net.parameters()},
         {'params': model.gamma_net.parameters()}
         ], lr=0.01)
    schedule = torch.optim.lr_scheduler.StepLR(
        opt, 50, 0.5
    )
    l_loss = torch.tensor(0.)
    for epoch in range(epochs):
        for X_bat in seq_generator:
            #tepoch.set_description(f'Epoch {epoch}')
            opt.zero_grad()
            loss, ll = model.ELBO_loss(X_bat, l_loss)
            l_loss = 0.2*ll
            loss.backward()
            opt.step()
            #print(loss.item())
            #tepoch.set_postfix(loss=loss.item())
        schedule.step()
# %%
model = SingleMofifModel()
train_loop(model, 50)
# %%
X = torch.load("saved_X.pt")
#model = SingleMofifModel()
#model.load_state_dict(torch.load("saved_param_dict.pt"))
pwm, gamma, Z = model.infer(X)
pwm1 = torch.argmax(pwm, -1)
I1 = torch.argmax(gamma, -1)
# %%
lat_dict = torch.load("saved_latents.pt")
pwm0 = lat_dict['pwm']
I0 = lat_dict['I']
Z0 = lat_dict['Z']
# %%
torch.save(model.state_dict(), "saved_param_dict2")
# %%
