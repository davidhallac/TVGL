import matplotlib
import matplotlib.pyplot as plt
import csv
#For "Type 1" plot
# matplotlib.rcParams['ps.useafm'] = True
# matplotlib.rcParams['pdf.use14corefonts'] = True
# matplotlib.rcParams['text.usetex'] = True

sizeList = [2,3,4,5,6,7,8,9,10,50,100,200,300,500,800,900,1000,1100]

# print [i*(i+1)/2*10 for i in sizeList]
# 1+j

central = [30, 60, 100, 150, 210, 280, 360, 450, 550]
cent = [2.8,10.8,37.9,142.9,523.9,1316.7,3055.3,5362.7,12186.9]

naiveNums = [30, 60, 100, 150, 210, 280, 360, 450, 550, 660, 780, 910, 1200, 1530, 2100]
naiveVals = [13.0, 8.5, 8.0, 11.0, 18.4, 24.7, 36.6, 56.6, 129.4, 240.3, 370.3, 535.4, 1285.1, 2165.2, 8202.4]

scsnums = [30, 60, 100, 210, 280, 360, 550, 2100, 3250, 12750, 50500]#, 201000],
#scsvals = [0.8,0.9,0.8,1.1,0.9,0.9,1.1,1.1,2.6, 19.9, 114.9, 657.3, 3134.7, 22631.5] #With 1e-3 convercence tolerance
scsvals = [0.9, 1.3, 1.2, 1.1, 1.1, 1.2, 6.6, 21.1, 26.6, 215.6, 6611.2]#, 26714.7]

decentral = [30, 60, 100, 150, 210, 280, 360, 450, 550, 12750, 50500, 201000, 451500, 1252500, 3204000, 4054500,5005000,6055500]
admm = [1.0,0.9,0.9,1.0,0.9,1.0,0.9,0.9,1.2,4.6,14.3,48.3,98.4,272.9,502.0,610.8,706.4,834.6]
# 1.1,0.9,1.0,0.8,0.8,0.8,	0.9,	0.8,   0.9,	2.4,	9.2,	36.2,	86.2,	255.1,  507.1, 610.8,  706.4,   834.6]

# central = [36,   72,   90,  108,  144,  162,  180,  198,  216,  270,  360,  450,  504,  630,  720, 900, 1008, 1350, 1800, 2250, 2700]
# cent = [9.54, 29.21,37.18,47.57,69.99,78.80,94.81,105.7,124.8,163.0,264.7,344.2,509.0,741.2,918.4,1437, 1997, 3559, 7665, 16505, 27634]

# decentral = [36,   72,   90,  108,  162,  504,  1008, 5022, 9000, 10008,20000, 30000] #20mil with 40 procs
# admm = [ 23.14, 39.16,46.05,47.67,62.23,177.2, 363.6, 1638, 2948, 3325, 6380, 9717]


#Get right units
# central = [1000*x for x in central]
# decentral = [1000*x for x in decentral]

# cent = [x/60 for x in cent]
# admm = [x/60 for x in admm]

plt.figure(figsize=(8,4))
plt.plot(central, cent)#, 'r--', linestyle = 'dotted')
plt.plot(naiveNums, naiveVals)#, 'c--')
plt.plot(scsnums, scsvals)#, 'g-.')
plt.plot(decentral, admm)#, 'b')
plt.legend(['CVXOPT', 'Naive ADMM', 'SCS', 'TVGL'], loc=4, fontsize = 13)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of Unknowns')
plt.ylabel('Time (Seconds) for Convergence')
# plt.title('Performance Comparison over Different Problem Sizes')
plt.ylim([0.5,20000])
plt.rcParams.update({'font.size': 14})

plt.savefig('convergence_results.eps', format='eps', bbox_inches='tight', dpi=1000)
#plt.show()