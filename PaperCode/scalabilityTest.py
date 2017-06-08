from inferGraphScalability import * #USECVX = TRUE, SCS
# from snapvxScalability import * #USECVX = FALSE, Naive ADMM

import numpy as np
import numpy.linalg as alg
import scipy as spy
import __builtin__

import matplotlib.pylab as pl
import time

# sizeList = [2,3,4,5,6,7,8,9,10,50,100,200,300,500,707,900,1000,1100]
sizeList = [100]#[15,17,20,25]
timeList = [10]
useCVX = True

np.random.seed(0)
timingVals = np.zeros([__builtin__.len(sizeList), __builtin__.len(timeList)])
for sizeTemp in range(__builtin__.len(sizeList)):
    for timeTemp in range(__builtin__.len(timeList)):


        size = sizeList[sizeTemp]
        timesteps = timeList[timeTemp]
        print "Solving for size", size, ", timesteps ", timesteps

        samplesPerStep = 10#int(np.log2(size))
        timeShift = timesteps/3 #Number of steps till new covariance matrix appears
        eps = 1e-3

        # Optimization parameters
        alpha = 0.1 # Lasso parameter
        beta = 0.6 # Weight between basis nodes
        FroError = []
        Score = []

        # Generate sparse, random, inverse covariance matrix (inverse of inverseCov is original covariance matrix)
        np.set_printoptions(suppress=True, precision = 3, threshold = 5)
        S_true = np.zeros((size,size))
        while (alg.det(S_true) <= 1e-2 ):
            # print "CASE 1"
            G6 = GenRndGnm(PUNGraph, size, int((size**2)*0.05))
            S_true = numpy.zeros((size,size))
            for EI in G6.Edges():
                S_true[EI.GetSrcNId(), EI.GetDstNId()] = 0.6 
            if(size < 10):
                #So we get positive definite matrices, since S_true.max is often 0
                S_true =  S_true + S_true.T + size*0.2*np.matrix(np.eye(size))
            else:
                S_true =  S_true + S_true.T + 2*S_true.max()*np.matrix(np.eye(size))
        #    print S_true
        #    print alg.det(S_true)
        Cov = alg.inv(S_true)
        S_true2 = numpy.zeros((size,size))
        while (alg.det(S_true2) <= 1e-2 ):
            # print "CASE 2"
            #print int(numpy.log2(size))*size
            G6 = GenRndGnm(PUNGraph, size, int((size**2)*0.05))
            S_true2 = np.zeros((size,size))
            for EI in G6.Edges():
                S_true2[EI.GetSrcNId(), EI.GetDstNId()] = 0.6
            if(size < 10):
                S_true2 =  S_true2 + S_true2.T + size*0.2*np.matrix(np.eye(size))
            else:
                S_true2 =  S_true2 + S_true2.T + 2*S_true2.max()*np.matrix(np.eye(size))
        #    print S
        #    print alg.det(S_true2)
        Cov2 = alg.inv(S_true2)
        S_true3 = numpy.zeros((size,size))
        while (alg.det(S_true3) <= 1e-2 ):
            # print "CASE 3"            
            #print int(numpy.log2(size))*size
            G6 = GenRndGnm(PUNGraph, size, int((size**2)*0.05))
            S_true3 = np.zeros((size,size))
            for EI in G6.Edges():
                S_true3[EI.GetSrcNId(), EI.GetDstNId()] = 0.6 
            if(size < 10):
                S_true3 =  S_true3 + S_true3.T + size*0.2*np.matrix(np.eye(size))
            else:            
                S_true3 =  S_true3 + S_true3.T + 2*S_true3.max()*np.matrix(np.eye(size))
        #    print S
        #    print alg.det(S_true2)
        Cov3 = alg.inv(S_true3)

        gvx = TGraphVX()
        for i in range(timesteps):
        	# Generate random samples
        	x_samples = 0
        	if (i < timeShift):
        		x_samples = np.random.multivariate_normal(np.zeros(size), Cov, samplesPerStep).T
        	elif (i < 2*timeShift):
        		x_samples = np.random.multivariate_normal(np.zeros(size), Cov2, samplesPerStep).T
        	else:
        		x_samples = np.random.multivariate_normal(np.zeros(size), Cov3, samplesPerStep).T
        	empCov = np.cov(x_samples)
         	if (samplesPerStep == 1):
        	 	empCov = x_samples*x_samples.T

        #	print empCov
        	#Add Node, edge to previous timestamp
        	n_id = i
        	S = semidefinite(size,name='S')
        	obj = -log_det(S) + trace(empCov*S) #+ alpha*norm(S,1)
        	gvx.AddNode(n_id, obj)

        	if (i > 0): #Add edge to previous timestamp
        		prev_Nid = n_id - 1
        		currVar = gvx.GetNodeVariables(n_id)
        		prevVar = gvx.GetNodeVariables(prev_Nid)
        		edge_obj = beta*norm(currVar['S'] - prevVar['S'],1) # one norm penalty function
        		gvx.AddEdge(n_id, prev_Nid, Objective=edge_obj)

        	#Add rake nodes, edges
        	gvx.AddNode(n_id + timesteps)
        	gvx.AddEdge(n_id, n_id+timesteps, Objective=alpha*norm(S,1))

        print "Starting to solve:"
        t = time.time()
        if(useCVX):
            gvx.Solve( UseADMM = False)
        else:
            gvx.Solve(Rho = 0.5)#, Verbose=True)

        end = time.time() - t
        print "SOLUTION TIME", end

        timingVals[sizeTemp][timeTemp] = end

np.set_printoptions(threshold='nan')
print timingVals

for nodeID in range(timesteps):
    val = gvx.GetNodeValue(nodeID,'S')
    S_est = np.zeros((size,size))
    S_est[np.triu_indices(size)] = val #S_est matrix convert semidefinite (n(n-1) entries) to a full semidefinite matrix
    temp = S_est.diagonal()
    ind = (S_est<eps)&(S_est>-eps)
    S_est[ind]=0
    S_est = np.asarray((S_est + S_est.T) - np.diag(temp))
#    print S_est


