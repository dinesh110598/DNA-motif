## PyTorch dataloader basics

- The torch `Dataloader` object is basically an **iterable similar to ranges and generators**.

- Ranges as **iterable-style** datasets while generators are **map sytle** and torch dataloaders support both styles (this is prolly just guessing).

- The dataloaders primarily receive `Dataset` objects as argument. It implements `__init__`, `__len__` and `__getitem__` methods (**characteristic of tuples**). Dataset objects need not be speed-optimized in the sense that we can make the `__getitem__` method to directly read from files (from HDD or SSD, slow as hell)!

- The `Dataloader` object optimizes a `Dataset` in retrieving shuffled batches or through custom `Sampler` objects!

- But for our purpose, I believe it's as simple as calling `Sampler(X, batch_size=model.batch, shuffle=True)` where `X` is the tensor contain all our sequences!

## Metastable point of full(4, 0.25) for PWM
- When we predict the r*w simplex arrays independently, the pwm_net output converges to roughly `full(4, 0.25)`! Which doesn't minimize the Dirichlet prior we have but actually maximizes. But it's a locally metastable point since the likelihood term in the loss penalizes random perturbations from there except when we're lucky enough to have it initialized near an actual pwm. (But I'd like to make my own luck)!
- The problem with Dirichlet prior is though it discourages the 0.25 array, its log prob is maximized only at the discrete corners of the simplex space in 4d.
- So we need a distribution that equally favours **all points within the boundary** of the simplex space!
- So, we'll use another prior which considers the sum of **maximum 2 of the 4 simplex points** where (0.4, 0.4) is equally good compared to (0.7, 0.1) so our neural network state has a continous space to traverse!

## Terms in the loss function

1. pwm_pr_lp
   
   - Sum of r*w terms
   
   - Max magnitude - 0.2*10 = 2 per term
   
   - Max total magnitude - 3 * 10 * 2 = 60

(tbc...)
