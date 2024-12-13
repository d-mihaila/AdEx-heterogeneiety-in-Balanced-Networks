# AdEx-heterogeneiety-in-Balanced-Networks

Repository trying to apply the Lorenzian Ansatz Mean Field approach on Adaptive Exponential LIF neurons as done by [1]. 
I used a Taylor Expansion up to 3 terms but there is a miss-match of the parameters for ther MF so the synaptic activity blows up. 

To continue: seems like AdEx cannot handle TE (becomes quadratic form) and hence maybe other MF approaches work better (transfer function etc). 
Alternatively, adjust the time constants and other parameters. 

Note: for the actual project, more parameters (not just the threshold) need to be heterogeneous. 



[1] Gast, Richard, Sara A. Solla, and Ann Kennedy. "Effects of neural heterogeneity on spiking neural network dynamics." arXiv preprint arXiv:2206.08813 (2022).
