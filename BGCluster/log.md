- The BGcluster model trains on sequences that have different background second order Markov behaviors and clusters them according to the same.

- The global constants are n=1000, l=100, r=3

- For efficiently predicting the probability of X given the phi, we can use `stack([X[:-2], X[1:-1]])`as an input to the indices of parameter array or neural network.

- For all the models, I've calculated the likelihood directly and then used this likelihood as input to predict gammas which I operate once again on the likelihood! This seems pretty twisted. What if I try a neural network that acts just on `Xoh`?

- Also, I'm doubting my assumption of using Delta distribution to perform the optimization of variables like gamma and phi. While I doctored the dataset to have deterministic values of `phi` and `I`, the elements of the dataset were still *sampled* from these variables, `phi` especially. So, there's still an inherent uncertainty in determining phi by looking at the dataset alone.
  
  - How does using an uncertain distribution remedy the problem of component collapse for phi and gamma? Since the variance is non-zero, it also means there's non-zero chance that this slips right in there! But it doesn't work anyways...

- I'll try to mimic the example in Pyro as closely as possible- including the various priors, learning rates, etc.

- Mimicking the example in Pyro
