### FUNCTION TO CALCULATE TARGET TAU FROM EXPONENTIALLY FALLING DISTRIBUTION
from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from scipy.stats import expon

def inv_exp_dist(coeff =1, coeff_tau = 1):
    
    
    tau_linespace = np.linspace(start=0, stop=100, num=200, retstep=True)
    tau_distribution = coeff*np.exp(-coeff_tau*tau_linespace[0])
    
    
    # plt.plot(tau_distribution, linestyle='dotted', color='blue')
    
    # plt.ylim(ymin=0)
    # plt.xlim(xmin=0)
    
    # plt.xlabel('tau test')
    # plt.ylabel('Probability Density')
    # plt.title('Falling Exponential Distribution')

    
    # plt.show()

    return tau_distribution


def pick_target_tau_obs():
    
    check = False
    list = inv_exp_dist(200)
    target = list[int(np.random.random()*(len(list)-1))]
    
    if target in list:
        check = True
    
    return target, check

##Use of predefined scipy function to select from a distribution

def pick_target_tau():
    

    tau_target = expon().rvs()
 
    return tau_target


###PLOTTING PICK_TARGET_TAU()
# tau = []    
# for i in range (pow(10,5)):
    
#     tau.append(pick_target_tau())

# x = np.linspace(start=0, stop=100, num=400)
# y = np.exp(-x)
# plt.plot(x,y, color='xkcd:crimson', label = r'$e^{-\tau}$')
# seaborn.histplot(tau, color='xkcd:cornflower blue', stat='density', label=r'Distribution after selecting $10^5$ target $\tau$')

# # plt.legend(ideal, r'$e^{-\tau}$')
# # plt.legend(distrib, r'Distribution after selecting $10^4$ target $\tau$')
# plt.ylim(ymin=0)
# plt.xlim(0,6)
# # plt.title(r'Selection of optical path ($\tau$) from Falling Exponential Distribution',fontsize=30)
# plt.xlabel(r'$\tau$',fontsize=35)
# plt.ylabel('Probability Density', fontsize=35)
# plt.xticks(fontsize = 30) 
# plt.yticks(fontsize = 30) 
# plt.legend(prop={'size':30})


# plt.show()