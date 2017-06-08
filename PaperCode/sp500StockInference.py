# -*- coding: utf-8 -*-
"""
Infer the network for ALL companies in the S&P 500
"""
      
import numpy as np
import numpy.linalg as alg
import scipy as spy

import matplotlib as mpl
import matplotlib.pylab as pl
import time
 

def timing_set(center, samplesPerStep_left, count_left, samplesPerStep_right, count_right ):
    time_set = []
    count_left = min(count_left, center/samplesPerStep_left)
    print 'left timesteps: = ', count_left
    start = max(center- samplesPerStep_left*(count_left), 0)
    for i in range(count_left):
        time_interval = [start, start + samplesPerStep_left -1]
        time_set.append(time_interval)
        start = start + samplesPerStep_left
    count_right = min(count_right, 245/samplesPerStep_left)
    print 'right timesteps: = ', count_right
    for i in range(count_right):
        time_interval = [start, start + samplesPerStep_right -1]
        time_set.append(time_interval)
        start = start + samplesPerStep_right
    return time_set
    
# Problem parameters - used for generating random data in Section 5a (we can ignore this part of the code when working with stock data)
size = 10   
timesteps = 100
samplesPerStep = 200
numberOfCov = 2
timeShift = int(np.ceil(float(timesteps)/numberOfCov)) #Number of steps till new covariance matrix appears
eps = 3e-3

#Stricter tolerance thresholds for finance
eps_abs = 1e-4
eps_rel = 1e-4



# Choose a penalty function
# 1: l1, 2: l2, 3: laplacian, 4: l-inf, 5: perturbation node penalty
index_penalty = 1


set_length = 1
aa = 1
bb = 20
compare = False
# Covariance matrix parameters
#cov_mode = 'Syn'
cov_mode_number = 5 # 1,2,4: normal cov, 3: cov for laplacian, 5: perturbation 
low = 0.3
upper = 0.6


cov_mode = 'Stock'
if cov_mode == 'Stock':
    stock_list = range(524) #All companies
    samplesPerStep = 3
    time_set = timing_set(171, samplesPerStep , 5 , samplesPerStep, 5)

# Kernel parameters
#use_kernel = True
sigma = 1
kernel_width = 20
use_kernel = False

comp_with = 'zeroBeta GL'
if set_length == 1:
    if cov_mode == 'Syn':
        alpha_set = [aa]
        beta_set = [bb]
    elif cov_mode == 'Stock':
        alpha_set = [0.18] # apple case and flash crash, to only get one event!
        beta_set = [13]  # apple case
        
else:
    compare = False
    alpha_set = np.linspace(0.2, 1 , set_length)
    if use_kernel == False:
        beta_set = np.logspace(0, 1.3, set_length)
    else:
        print 'kernel = ', use_kernel, 'beta is kernel width'
        interval = min(timeShift/set_length,5)
        beta_set = np.arange(1,set_length*interval, interval)
         


if index_penalty == 1:
    print 'Use l-1 penalty function'
    from inferGraphL1 import *
elif index_penalty == 2:
    print 'Use l-2 penalty function'
    from inferGraph2 import *
elif index_penalty == 3:
    print 'Use laplacian penalty function'
    from inferGraphLaplacian import *
elif index_penalty == 4:
    print 'Use l-inf penalty function'
    from inferGraph4 import *
else:
    print 'Use perturbation node penalty function'
    from inferGraphPN import *


#--------------------------------------- Define private functions ------------------------------------------
def genCovariace(size): 
    MaxIter = 1e+6
    S = np.zeros((size,size))
    itn = 0
    while(alg.det(S) <= 1e-3 and itn < MaxIter):
        itn = itn + 1
        G6 = GenRndGnm(PUNGraph, size, int((size*(size-1))*0.05))
        S = np.zeros((size,size))
        for EI in G6.Edges():
            S[EI.GetSrcNId(), EI.GetDstNId()] = 0.6
        S =  S + S.T + S.max()*np.matrix(np.eye(size))
    if itn == MaxIter:
        print 'fail to find an invertible sparse inverse covariance matrix'
    S = np.asarray(S)
    return S
    
def genInvCov(size, low = 0 , upper = 0.6, portion = 0.05):
    S = np.zeros((size,size))
    G = GenRndGnm(PUNGraph, size, int((size*(size-1))*portion))
    for EI in G.Edges():
        value = (np.random.randint(2) - 0.5)*2*(low + (upper - low)*np.random.rand(1)[0])  
        S[EI.GetSrcNId(), EI.GetDstNId()] = value
    S = S + S.T
    vals = alg.eigvalsh(S)
    S = S + (0.1 - vals[0])*np.identity(size)
    return np.matrix(S)
    
def genMulCov(size, numberOfCov, low, upper, mode, portion = 0.05):
    S_set = []   
    Cov_set = []
    minEVal_set = [] 
    m = size/3
    mm = m/2
    S_init = np.zeros((size,size))
    for k in range(numberOfCov):
        S = np.zeros((size,size))
        if k == 0:
            S = genInvCov(size, low, upper, portion)
            if mode == 5:      
                ind_zero = np.where(spy.sparse.rand(m, size-m, 0.5).todense() == 0)
                value = np.multiply((np.random.randint(2, size = (m, size -m)) - 0.5)*2,(low + (upper - low)*np.random.rand(m,size -m)))
                value[ind_zero] = 0
                hub = value
                S[:m, m:size] = hub
                S[m:size, :m] = hub.T        
                minEVal_set.append(alg.eigvalsh(S)[0])
            S_init = S
        elif mode == 3: #'laplacian'
            ind1 = range(m)
            ind2 = np.random.permutation(m)
            S = np.copy(S_init)
            S[ind1, :] = S[ind2, :]            
            S[:, ind1] = S[:, ind2]
        elif mode  == 5: #'perturbation'
            S = np.copy(S_init)
            ind_zero = np.where(spy.sparse.rand(mm, size-mm, 0.5).todense() == 0)
            pert = np.multiply((np.random.randint(2, size = (mm, size -mm)) - 0.5)*2,(low + (upper - low)*np.random.rand(mm,size -mm)))
            pert[ind_zero] = 0 
            S[:mm, mm:size] = pert
            S[mm:size, :mm] = pert.T
            minEVal_set.append(alg.eigvalsh(S)[0])
        else:
#            print 'Activate normal mode'
            S = genInvCov(size, low, upper, portion)
        S_set.append(S)
    
    for k in range(numberOfCov):
        if mode == 5:
            S_set[k] = S_set[k] + (0.1 - min(minEVal_set))*np.identity(size)
        Cov_set.append(alg.inv(S_set[k]))
    return S_set, Cov_set   
     
def genEmpCov(samples, useKnownMean = False, m = 0):
    size, samplesPerStep = samples.shape
    if useKnownMean == False:
        m = np.mean(samples, axis = 1)
    empCov = 0
    for i in range(samplesPerStep):
        sample = samples[:,i]
        empCov = empCov + np.outer(sample - m, sample -m)
    empCov = empCov/samplesPerStep
    return empCov
    
def getStocks(time_set, stock_list, data):
    timesteps = time_set.__len__()
    sample_set = []
    empCov_set = []
    stock_data = np.genfromtxt('Datasets/finance.csv', delimiter=',')
#    print stock_data
    size = stock_list.__len__()
#    print 'timesteps = ',timestpes
    for i in range(timesteps):
        time_interval = time_set[i] 
#        print time_interval
        sample_data = stock_data[time_interval[0]:time_interval[1], stock_list].T
        sample_data_set = sample_set.append(sample_data)
        empCov_set.append(genEmpCov(sample_data))
    return size, timesteps, sample_data_set, empCov_set
    

def indicesOfExtremeValue(arr, set_length, choice):
    if (choice == 'max'):
        index = np.argmax(arr)
    elif (choice == 'min'):
        index = np.argmin(arr)
    else:
        print 'invalid argument, choose max or min'
            
    index_x = index/set_length
    index_y = index - (index_x)*set_length
    return index, index_x, index_y
    
def upper2Full(a, eps = 0):
    ind = (a<eps)&(a>-eps)
    a[ind] = 0
    n = int((-1  + np.sqrt(1+ 8*a.shape[0]))/2)  
    A = np.zeros([n,n])
    A[np.triu_indices(n)] = a 
    temp = A.diagonal()
    A = np.asarray((A + A.T) - np.diag(temp))             
    return A   
    
def genEmpCov_kernel(sigma, width, sample_set, knownMean = True):
    timesteps = sample_set.__len__()
#    print timesteps
    mean_tile = 0
    K_sum = 0
    if knownMean != True:
        for j in range(int(max(0,timesteps-width)),timesteps):            
            K =  np.exp(-np.square(timesteps-j-1)/sigma)
            samplesPerStep = sample_set[j].shape[1]
            mean_tile = mean_tile + K* sample_set[j]
            K_sum = K_sum + K
            
        mean_tile =  np.sum(mean_tile, axis = 1)/samplesPerStep
        mean_tile = np.tile(mean_tile, (samplesPerStep,1)).T
    K_sum = 0
    S = 0
#    print 'timesteps and width is %d, %d, %d'%(timesteps,width, max(0,timesteps- width))
   
    for j in range(int(max(0,timesteps-width)),timesteps):
        K = np.exp(-np.square(timesteps-j-1)/sigma)
#        print 'j = ',j, 'K = ', K
        samplesPerStep = sample_set[j].shape[1]
        S = S + K*np.dot(sample_set[j]- mean_tile, (sample_set[j] -  mean_tile).T)/samplesPerStep
        K_sum = K_sum + K
    S = S/K_sum
    return S

def solveProblem(gvx, index_penalty, cov_mode, alpha, beta, timesteps, timeShift, Cov_set, use_kernel, sigma, sample_set, empCov_set, eps_abs = 1e-4, eps_rel = 1e-4):
    if use_kernel:
        empCov_set = []
    for i in range(timesteps):
        if cov_mode == 'Syn':
            # Generate random samples_set
            x_samples = 0
            j = i/timeShift
            x_samples = np.random.multivariate_normal(np.zeros(size), Cov_set[j], samplesPerStep).T
            sample_set.append(x_samples)
            # Generate empirical covariance matrix
            empCov = 0
            if use_kernel:
                kernel_width = beta
                empCov = genEmpCov_kernel(sigma, kernel_width, sample_set)
            else:
                empCov = genEmpCov(x_samples)
            empCov_set.append(np.asarray(empCov))
        else: 
            # Just use passed sample_set and empirical covariance set
            # Except when using kernel method
            empCov = empCov_set[i]
            if use_kernel:
                kernel_width = beta
                empCov = genEmpCov_kernel(sigma, kernel_width, sample_set)
                empCov_set.append(np.asarray(empCov))
        #   print empCov
            #Add Node, edge to previous timestamp
        n_id = i
        S = semidefinite(size,name='S')
        obj = -log_det(S) + trace(empCov*S) #+ alpha*norm(S,1)
        gvx.AddNode(n_id, obj)
        
        if (i > 0): #Add edge to previous timestamp
            prev_Nid = n_id - 1
            currVar = gvx.GetNodeVariables(n_id)
            prevVar = gvx.GetNodeVariables(prev_Nid)
            if use_kernel == False:
                if index_penalty == 1 or index_penalty == 2:
                    edge_obj = beta*norm(currVar['S'] - prevVar['S'],index_penalty) # norm 1, 2 penalty function
                elif index_penalty == 3:
                    edge_obj = beta*square(norm(currVar['S'] - prevVar['S'],index_penalty)) # laplacian penalty function
                elif index_penalty == 4:
                    edge_obj = beta*norm(currVar['S'] - prevVar['S'], np.inf) # norm 1, 2 penalty function
                else:
                    edge_obj = beta*norm(currVar['S'] - prevVar['S'],index_penalty)
                gvx.AddEdge(n_id, prev_Nid, Objective=edge_obj)
        
        #Add rake nodes, edges
        gvx.AddNode(n_id + timesteps)
        gvx.AddEdge(n_id, n_id+timesteps, Objective=alpha*norm(S,1))
        
    
    t = time.time()
    gvx.Solve(EpsAbs=eps_abs, EpsRel=eps_rel,Verbose=True,MaxIters=500)
    end = time.time() - t
    print 'time span = ',end
    return gvx, empCov_set

def genGraph(S_actual, S_est, S_previous, empCov_set, nodeID, e1, e2, e3, e4, display = False):
    D = np.where(S_est != 0)[0].shape[0]
    T = np.where(S_actual != 0)[0].shape[0]
    TandD = float(np.where(np.logical_and(S_actual,S_est) == True)[0].shape[0])
    P = TandD/D
    R = TandD/T
    offDiagDiff = S_actual - S_est
    offDiagDiff = offDiagDiff - np.diag(np.diag(offDiagDiff))
    S_diff = (S_est - S_previous)  
    S_diff = S_diff - np.diag(np.diag(S_diff))
    ind = (S_diff < 1e-2) & (S_diff > - 1e-2)
    S_diff[ind] = 0    
    K = np.count_nonzero(S_diff)
    e1.append(-np.log(alg.det(S_est)) + np.trace(np.dot(S_est, empCov_set[nodeID])) + K)
    e2.append(2* P*R/(P+R))
    
    
    K = float(np.where(np.logical_and((S_est>0) != (S_previous>0), S_est>0) == True)[0].shape[0])
    e3.append(-np.log(alg.det(S_est)) + np.trace(np.dot(S_est, empCov_set[nodeID])) + K)
    e4.append(alg.norm(S_est -  S_previous, 'fro'))
    
            
    return e1, e2, e3, e4
#--------------------------------------------- End Defining functions -------------------------------------- 


# Generate sparse, random, inverse covariance matrix (inverse of inverseCov is original covariance matrix)
np.random.seed(1)
#numpy.set_printoptions(suppress=True, precision = 3, threshold = 20)
numpy.set_printoptions(suppress=True, precision = 3)


S_set = []
Cov_set = []
Data_type = ''
if cov_mode == 'Syn':
    if cov_mode_number == 0:
        Data_type = cov_mode + '_fixed'
        for i in range(numberOfCov):
            S_true = genCovariace(size)
            S_set.append(S_true)
            Cov_set.append(alg.inv(S_true))
    else:
        Data_type = cov_mode + '%s'%(cov_mode_number)
        S_set, Cov_set = genMulCov(size, numberOfCov, low, upper, cov_mode_number)
    Data_type = Data_type + '(%s)'%(size)
    print Data_type
else:
    size, timesteps, sample_set_stock, empCov_set_stock = getStocks(time_set, stock_list,'finance.csv')
    Data_type = cov_mode + '(%s)'%(size)
#print S_set
   
e1_set = []
e2_set = []
e3_set = []
e4_set = []
e11 = []
e21 = []   
e31 = []   
e41 = []   
e51 = []   

FroError = []
Score = []
AIC = []
FroThetaDiff = []
print 'number of samples per time: ', samplesPerStep
for alpha in alpha_set:
    for beta in beta_set:
        print 'lambda = %s, beta = %s'%(alpha, beta)
        gvx = TGraphVX()   
        gvx_naive = TGraphVX()
        if cov_mode == 'Stock':
            print 'analyze stock data'  
            empCov_set = empCov_set_stock
            sample_set = sample_set_stock  
            empCov_set_naive = empCov_set_stock
            sample_set_naive = sample_set_stock
        else:
            empCov_set = []
            sample_set = []     
            empCov_set_naive = []
            sample_set_naive = []
            # do nothing
        
        gvx, empCov_set = solveProblem(gvx, index_penalty, cov_mode, alpha, beta, timesteps, timeShift, Cov_set, use_kernel, sigma, sample_set, empCov_set, eps_abs, eps_rel)
        
        if set_length == 1 and compare == True:
            gvx_naive, empCov_set_naive = solveProblem(gvx_naive, index_penalty, cov_mode, alpha, 0, timesteps, timeShift, Cov_set, use_kernel, sigma,sample_set_naive, empCov_set_naive,eps_abs, eps_rel) 
        e1 = []
        e2 = []
        e3 = []
        e4 = []
        S_previous = np.zeros((size,size))
        S_naive_previous = np.zeros((size,size))
        for nodeID in range(timesteps):
            val = gvx.GetNodeValue(nodeID,'S')
            S_est = upper2Full(val, eps)
            if nodeID == 0:
                S_previous = S_est
            if cov_mode == 'Syn':
                #Get actual InvCov
                j = nodeID/timeShift
                S_actual = np.asarray(S_set[j])
    #            print 'S_actual=', S_actual
            else:
                S_actual = np.identity(size)
                print '\nAt node = ', nodeID, '-----------------\nEmpCov = \n', empCov_set[nodeID], '\nTheta_est=\n', S_est, '\n'
#                
#                print '\nAt node = ', nodeID, '-----------------\nTheta_est=\n', S_est, '\n'                   
                print '\nCov = \n', alg.inv(S_est)
                if nodeID != 0:
                    print 'Theta_diff = \n', S_est - S_previous 
                    print 'FRONORM = ', alg.norm(S_est-S_previous, 'fro')
            
            e1, e2, e3, e4 = genGraph(S_actual, S_est, S_previous, empCov_set, nodeID, e1, e2, e3, e4, False)
            
            S_previous = S_est
            
            if set_length == 1 and compare == True:               
                val_naive = gvx_naive.GetNodeValue(nodeID,'S')
                S_naive = upper2Full(val_naive, eps)
                
                e11, e21, e31, e41 = genGraph(S_actual, S_naive, S_naive_previous, empCov_set_naive, nodeID, e11, e21, e31, e41, False)
                S_naive_previous =  S_naive
            
            
        e1_set.append(e1)
        e2_set.append(e2)
        e3_set.append(e3)
        e4_set.append(e4)
#        print e4
        FroError.append(sum(e1))
        Score.append(sum(e2))
        AIC.append(sum(e3))
        FroThetaDiff.append(sum(e4))
        
index1, index11, index12 = indicesOfExtremeValue(FroError, set_length, 'min')
index2, index21, index22 = indicesOfExtremeValue(Score, set_length, 'max')
index3, index31, index32 = indicesOfExtremeValue(AIC, set_length, 'min')
index4, index41, index42 = indicesOfExtremeValue(FroThetaDiff, set_length, 'min')      

ind = index3
alpha = alpha_set[index31]
beta =  beta_set[index32]
try:
    x =  range(timesteps)
    if cov_mode == 'Syn':
        pl.subplot(411)    
        pl.title(r'%s, $n_t$ = %s, ($\lambda$, $\beta$) = (%s, %s)'%(Data_type, samplesPerStep, alpha, beta))
        pl.plot(x, e1_set[ind], label = 'Use GL with penalty function')
        pl.ylabel(r'$\|\Theta_{true} - \Theta_{est}\|_{F}$')
        pl.subplot(412)
        pl.plot(x, e2_set[ind])
        pl.ylabel('Binary Error')
    else:
        pl.subplot(411)    
        pl.plot(x, e1_set[ind], label = 'Use GL with penalty function')
        pl.title(r'%s, $n_t$ = %s, ($\lambda$, $\beta$) = (%s, %s)'%(Data_type, samplesPerStep, alpha, beta))
    pl.subplot(413)
    pl.plot(x, e3_set[ind])
    pl.ylabel('AIC: -logdet + BC')
    pl.subplot(414)
    pl.semilogy(x, e4_set[ind])
    pl.ylabel(r'$\|\Theta_i - \Theta_{i-1}\|_{F}$')
    if set_length == 1 and compare == True:
        pl.subplot(411)     
        pl.plot(x, e11, label = comp_with)
        pl.subplot(412)
        pl.plot(x, e21)
        pl.subplot(413)
        pl.plot(x, e31)
        pl.subplot(414)
        pl.semilogy(x, e41)
#    pl.savefig('MeasurePlot')
    Data_type = cov_mode + '%s'%(cov_mode_number) + '%s'%(samplesPerStep)
    pl.savefig(Data_type)
    pl.show()
    print '\nSuceed to save' + Data_type
except:
    print 'fail to save plots'
if set_length > 1:
    #print index1, index11, index12, index2, index21, index22
    print 'alpha = ', alpha_set[index11], ' beta = ', beta_set[index12], ' FroError = ', FroError[index1]
    print 'alpha = ', alpha_set[index21], ' beta = ', beta_set[index22], ' Score = ', Score[index2]
    print 'alpha = ', alpha_set[index31], ' beta = ', beta_set[index32], ' AIC = ', AIC[index3]
    print 'alpha = ', alpha_set[index41], ' beta = ', beta_set[index42], ' FroThetaDiff = ', FroThetaDiff[index4]
    
    Fro_error = numpy.reshape(FroError,(set_length, set_length))
    Score = numpy.reshape(Score,(set_length, set_length))
    AIC = numpy.reshape(AIC,(set_length, set_length))
    FroThetaDiff = numpy.reshape(FroThetaDiff,(set_length, set_length))
    np.savez('ErrorMeasure%s'%(set_length), size = size, timesteps = timesteps, numberOfCov = numberOfCov,
             alpha_set = alpha_set, beta_set = beta_set, S_set = S_set, 
             Fro_error = Fro_error, Score = Score, AIC =  AIC, FroThetaDiff=  FroThetaDiff,
             e1_set = e1_set, e2_set = e2_set, e3_set = e3_set, e4_set = e4_set)
    np.savez('OptPars%s'%(set_length), size = size, timesteps = timesteps, numberOfCov = numberOfCov,
             alpha_fro = alpha_set[index11], beta_fro = beta_set[index11], 
             alpha_score = alpha_set[index21], beta_score = beta_set[index22], 
             alpha_AIC = alpha_set[index31], beta_AIC = beta_set[index31], 
             alpha_theta = alpha_set[index41], beta_theta = beta_set[index42])
    
    Y, X =  np.meshgrid(alpha_set, beta_set)
#    pl.figure(1)
    try:
        pl.subplot(221)
        pl.contourf(X, Y, Score)
        pl.ylabel(r'$\alpha$')
        pl.xlabel(r'$\beta$')
        pl.colorbar()
        pl.title('Score')
        pl.subplot(222)
        pl.contourf(X, Y, Fro_error)
        pl.ylabel(r'$\alpha$')
        pl.xlabel(r'$\beta$')
        pl.title(r'$S_{true} - S_{est}$')
        pl.colorbar()
        pl.subplot(223)
        pl.contourf(X, Y, AIC)
        pl.ylabel(r'$\alpha$')
        pl.xlabel(r'$\beta$')
        pl.title('AIC')
        pl.colorbar()
        pl.subplot(224)
        pl.contourf(X, Y, FroThetaDiff)
        pl.ylabel(r'$\alpha$')
        pl.xlabel(r'$\beta$')
        pl.title(r'$S_i - S_{i+1}$')
        pl.colorbar()
        pl.savefig('GridGraph%s'%(set_length))
        pl.show()
        print ('\nSuceed to save GridGraph%s'%(set_length))
    except:
        print 'fail to save graph'
print '\nEnd'
