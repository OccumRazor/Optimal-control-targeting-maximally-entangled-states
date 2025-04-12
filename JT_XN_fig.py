
import csv,matplotlib.pyplot as plt,numpy as np
csv_name = 'rf/iter_info.csv'
with open(csv_name, mode ='r') as file:
    iters = []
    JT_ss = []
    JT_min = []
    JX = []
    JN = []
    csvFile = csv.DictReader(file)
    for line in csvFile:
        iters.append(int(line['current_run']))
        JT_ss.append(float(line['current_JT_ss (best cat)']))
        JT_min.append(float(line['current_JT_ss (min cat)']))
        JX.append(float(line['var_X']))
        JN.append(float(line['var_N']))


plt.plot(iters,JT_ss,label=r'|0+$\rangle$')
plt.plot(iters,JT_min,label='min')
plt.xlabel('iteration')
plt.ylabel('inFidelity')
plt.legend(loc='best')
plt.savefig('XN_OPT.pdf')

#fig, ax1 = plt.subplots()
#f, (ax1, ax2) = plt.subplots(2,1)
#ax1.plot(iters,JT_ss)
#ax1.set_ylabel('inFidelity')
#ax2.plot(iters,np.array(JX)+np.array(JN),'g',label=r'$J_X$')
#ax2.set_xlabel('iter')
#ax2.set_ylabel(r'$J_X$')
#ax3=ax2.twinx()
#ax3.plot(iters,JN,'b',r'$J_N$')
#ax3.set_ylabel(r'$J_N$')
#ax3.legend(loc='best')
#plt.show()
