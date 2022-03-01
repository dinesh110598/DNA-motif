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
        self.l = 100 #Length of each seq
        self.r = 3 #Number of clusters/modules
        self.w0 = 5 #Min motif length
        self.w1 = 20 #Max motif length
        self.batch = 50 #Training batch size
        self.avg = 5 #Averaging size for exp values
        #over certain latents
        
        #Model parameters in the guide
        #self.phi_param = nn.parameter.Parameter(
            #torch.zeros([4, 4, 4]))
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
        self.z_net = Z_Net()
        self.gamma_net = nn.Sequential(
            nn.Conv1d(self.r, 16, 5),
            nn.Mish(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 1, 5),
            nn.Mish(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(22, 16),
            nn.Mish(),
            nn.Linear(16, 3),
            nn.Softmax(-1)
        )
        
        #Domains of various categorical latents
        self.w_d1 = torch.arange(4.5, 19.5)
        
        #Prior distributions of various latent variables
        self.phi_pr = dist.Dirichlet(torch.full([4,4,4], 2.))
        self.pwm_pr = dist.Dirichlet(torch.full([self.r,
                                                self.w1,
                                                4], 0.2))
        self.w_pr = dist.Binomial(self.w1-self.w0, 0.5)
        self.gamma_pr = dist.Dirichlet(torch.full([self.batch, 
                                                   self.r], 
                                                  0.2))
        self.Z_pr = dist.Binomial(self.l-self.w1-1, 0.5)
    def pwm_pr_lp(self, x: torch.Tensor):
        lp = self.pwm_pr.log_prob(x)
        out = torch.minimum(torch.full_like(lp, 3.),
                            lp)
        return out
    def gamma_pr_lp(self, x: torch.Tensor):
        lp = self.gamma_pr.log_prob(x)
        out = torch.minimum(torch.full_like(lp, 3.),
                            lp)
        return out
    #Above functions regulate the infinities of
    #Dirichlet near 0 and 1
        
    def likelihood(self, phi, pwm, w2, gamma, Z, X):
        lp = torch.empty([self.avg])
        for w in w2:
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
                    
                    for u in torch.arange(Z[i], 
                                          Z[i]+w[I[i]]):
                        p[i,I,u] = pwm[I[i], u-Z[i], X[i,u]]
                        
                    for u in torch.arange(Z[i]+w[I[i]], 
                                          self.l):
                        p[i,I,u] = phi[X[i, u-2],
                                    X[i, u-1],
                                    X[i, u]]
            
            lp = torch.log(p)
        return torch.sum(lp*gamma.unsqueeze(-1))
        
    def _sample_w(self):
        # Calculate samples of w and resp probabilities
        ww = self.w_net(torch.ones([1,1], 
                                   requires_grad=False))[0]
        var = torch.full([3], 0.05)
        mean = ww*(self.w1 - self.w0) + self.w0
        w_dist = DiscreteLogistic(self.w0, self.w1, mean, var)
        with torch.no_grad():
            w2 = torch.stack([torch.floor(mean).int(),
                              torch.ceil(mean).int()], 1)
            # We are looking at all possible permutations
            # of w so cartesian product!
            w2 = torch.cartesian_prod(
                *(w2[j,:] for j in torch.arange(self.r))
            )
            # w2.shape = [8, self.r]
        w_lp = w_dist.log_prob(w2 - self.w0)
        w_prob = w_lp.exp().transpose(0,1)
        #w_prob.shape = [self.r, 8]
        return w2, w_lp, w_prob
        
    def ELBO_loss(self, X):
        """Samples from current value of guide
        distribution(s) and adds the corresp
        loss logits then and there. Returns
        the corresp partial ELBO loss. The
        expectation values are exact hence
        no reparameterization!

        Args:
            X (torch.Tensor): Input (batched)
            sequence
        """
        #Initialize terms for ELBO loss
        log_X = torch.tensor(0.)
        log_guide = torch.tensor(0.)
        log_prior = torch.tensor(0.)
        
        #phi = F.softmax(self.phi_param)
        pwm = self.pwm_net(torch.ones([1,1], 
                                      requires_grad=False))
        # Discrete variables like w are approximated by a thin
        # logistic distrubution (discretized)
        pwm_prior = self.pwm_pr_lp(pwm)
        # pwm_guide = 0
        log_prior += pwm_prior.sum()
        
        w2, w_lp, w_prob = self._sample_w()
        w_prior = torch.dot(w_prob.prod(0),
                            self.w_pr.log_prob(w2).sum(1))
        log_prior += w_prior
        w_guide = torch.dot(w_prob.prod(0),
                            w_lp.sum(1))
        log_guide += w_guide.sum()
        
        # Constructing one-hot X for finding pwm probability
        # at every position.
        Xoh = torch.permute(F.one_hot(X), (0, 2, 1)).float()
        # Xoh.shape = (batch, 4, l)
        for i_w in range(2**self.r): 
            w = w2[i_w]
            pwm_diag = torch.permute(pwm, (0, 2, 1))
            def f(t): return torch.diag_embed(t, dim1=0, dim2=2)
            pwm_diag = [f(pwm_diag[j, :, :w[j]])
                        for j in range(self.r)]
            
            # Contains w[j] x w[j] diagonal matrices repeated
            # 4 times in dim=1
            y = [F.conv1d(Xoh, pwm_diag[j]).log()
                for j in range(self.r)]
            left = torch.floor((w-1)/2)
            right = torch.ceil((w-1)/2)
            y = [F.pad(y[j], (int(left[j]), int(right[j])))
                for j in range(self.r)]
            # convolution result shouldn't contain zeros!
            # y2 just for the sake of debugging
            y = torch.stack([y[j].mean(1)
                            for j in range(self.r)], 1)
            
            # We'll use y to predict gamma first
            # and then predict Z for each I😕
            gamma = self.gamma_net(y)
            gamma_prior = self.gamma_pr_lp(gamma).sum()
            log_prior += w_prob[:,i_w].prod()*gamma_prior
            #log_guide += 0
            # Look carefully into above 2 expns!
            for I in range(self.r): #I is module identity!
                wI = w[I]
                I_pr = gamma[:,I]
                Z_cls, Z_del = self.z_net(y[:, I:I+1, :])
                #How come both above qty's do not
                #change across the batch?
                #Indices of maximum class prob across
                #the minibatch
                ind = torch.argmax(Z_cls, dim=1)+1
                gap = (99-self.w1)/10
                Z_m = (ind + Z_del[torch.arange(50),ind])*gap
                Z_dist = DiscreteLogistic(0, 99, Z_m,
                                          torch.tensor([0.05]))
                # All Z are independent- so no cartesian prod
                with torch.no_grad():
                    Z2 = torch.stack([torch.floor(Z_m).int(),
                                    torch.ceil(Z_m).int()], 0)
                Z_lp = Z_dist.log_prob(Z2)
                Z_prob = Z_lp.exp()
                I_p2  = I_pr.unsqueeze(0)
                Z_prior = (w_prob[I, i_w]*Z_prob*I_p2*
                           self.Z_pr.log_prob(Z2))
                log_prior += Z_prior.sum()
                Z_guide = (w_prob[I, i_w]*Z_prob*I_p2*
                           Z_lp)
                log_guide += Z_guide.sum()
                
                for i in torch.arange(self.batch):
                    Z = Z2[:,i]
                    val = pwm[I, torch.arange(wI),
                              X[i, Z[0]:Z[0]+wI]]
                    log_X += (w_prob[I, i_w]*Z_prob[0,i]*
                              I_pr[i]*val.log().sum())
                    val = pwm[I, torch.arange(wI),
                              X[i, Z[1]:Z[1]+wI]]
                    log_X += (w_prob[I, i_w]*Z_prob[1,i]*
                              I_pr[i]*val.log().sum())
        # ELBO loss is maximized hence the following
        # signs in each term
        return -log_X - log_prior + log_guide
    
    def infer(self, X):
        pwm = self.pwm_net(torch.ones([1, 1],
                        requires_grad=False))
        ww = self.w_net(torch.ones([1, 1]))[0]
        var = torch.full([3], 0.05)
        mean = ww*(self.w1 - self.w0) + self.w0
        w = torch.round(mean).int()
        
        Xoh = torch.permute(F.one_hot(X), (0, 2, 1)).float()
        pwm_diag = torch.permute(pwm, (0, 2, 1))
        def f(t): return torch.diag_embed(t, dim1=0, dim2=2)
        pwm_diag = [f(pwm_diag[j, :, :w[j]])
                    for j in range(self.r)]

        # Contains w[j] x w[j] diagonal matrices repeated
        # 4 times in dim=1
        y = [F.conv1d(Xoh, pwm_diag[j]).log()
             for j in range(self.r)]
        left = torch.floor((w-1)/2)
        right = torch.ceil((w-1)/2)
        y = [F.pad(y[j], (int(left[j]), int(right[j])))
             for j in range(self.r)]
        y = torch.stack([y[j].mean(1)
                         for j in range(self.r)], 1)
        gamma = self.gamma_net(y)
        I = torch.argmax(gamma, dim=1)
        y1 = y[torch.arange(self.batch),
               I, :]
        Z_cls, Z_del = self.z_net(y1.unsqueeze(1))
        #How come both above qty's do not
        #change across the batch?
        #Indices of maximum class prob across
        #the minibatch
        ind = torch.argmax(Z_cls, dim=1)+1
        gap = (99-self.w1)/10
        Z = (ind + Z_del[torch.arange(50), ind])*gap
        
        return pwm, w, gamma, Z

def train_loop(model: SingleMofifModel, epochs):
    X = torch.load("saved_X.pt")
    X.requires_grad = False
    assert X.shape[0]%model.batch == 0
    X = X.view([-1, model.batch, X.shape[1]])
    opt = torch.optim.Adam([
        {'params': model.w_net.parameters()},
        {'params': model.z_net.parameters()},
        {'params': model.pwm_net.parameters(), 'lr':1e-3},
        {'params': model.gamma_net.parameters(),
         'lr': 1e-3}
        ], lr=1e-2)
    schedule = torch.optim.lr_scheduler.StepLR(opt, 30, 
                                               gamma=0.2)
    for epoch in range(epochs):
        with tqdm(range(X.shape[0]), 
                           unit="batch") as tepoch:
            for step in tepoch:
                tepoch.set_description(f'Epoch {epoch}')
                opt.zero_grad()
                loss = model.ELBO_loss(X[step])
                loss.backward()
                nn.utils.clip_grad_norm_( #type:ignore
                    model.pwm_net.parameters(),
                    10.)
                opt.step()
                tepoch.set_postfix(loss=loss.item())
            schedule.step()
# %%
model = SingleMofifModel()
train_loop(model, 10)
# %%
X = torch.load("saved_X.pt")
model = SingleMofifModel()
model.load_state_dict(torch.load('saved_param_dict.pth'))
pwm, w, gamma, Z = model.infer(X[:model.batch])
# %%
