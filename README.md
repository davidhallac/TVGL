# TVGL
TVGL is a python solver for inferring dynamic networks from raw time series data. For implementation details, refer to the paper, available at: http://stanford.edu/~hallac/TVGL.pdf.

-----

Download & Setup
======================
Download the source code by running the following code in the terminal:
```
git clone https://github.com/davidhallac/TVGL.git
```


Usage
======================
TVGL can be called through the following file:
```
TVGL.py
```
**Parameters**

data : a T-by-n numpy array with the raw data (each row is a new timestamp)

lengthOfSlice : Number of samples in each ``slice'', or timestamp

lamb : the lambda regularization parameter controlling the network sparsity (as described in the paper)

beta : the beta parameter controlling the temporal consistency (as described in the paper)

indexOfPenalty : The regularization penalty to use (1 = L1, 2 = L2, 3 = Laplacian, 4 = L_inf, 5 = perturbed node)

verbose = False : Whether or not to run ADMM in ``verbose'' mode (to print intermediate steps)

eps = 3e-3 : Threshold at which we treat output network weight as zero

epsAbs = 1e-3 : ADMM absolute tolerance threshold (see full details in http://stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf)

epsRel = 1e-3 : ADMM relative tolerance threshold (see http://stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf)




Example
======================
Running the following script provides an example of how the TVGL solver can be used:
```
exampleTVGL.py
```
