###FILE THAT INCLUDES ANY CALCULATIONS THAT ARE LINKED TO THE NUCLEI
from math import copysign
from operator import index
import matplotlib.pyplot as plt
import numpy as np
from second_sphere import conversion_sphertocart, m_H, m_He, m_O, k_B, m_chi_jup
from collision import magnitude
import seaborn

##2D distribution for nuclei speed and angular scattering. Candidate values for both wil be selected randomly from a range and will 
##then go through the selector and either be rejected or accepted.

n=0

#not used as we actually collect the temperature dependent velocity directly from the particle directory
def temperature(T):
    temp_vel = 2*k_B*T/(m_chi_jup)
    return temp_vel
def turningpoints(lst):
    dx = np.diff(lst)

    zero=[]
    for i in range(len(dx)-1):
        if copysign(1,dx[i])!=copysign(1,dx[i-1]) and i !=0:
            zero.append(i)
    return zero
# x=np.linspace(0, 2*np.pi)
# fx = np.power(5.31553155e-12, 2)*pow((pow(5.31553155e-11, 2)+np.power(5.31553155e-12, 2)-2*5.31553155e-12*5.31553155e-11*np.cos(x)),((1/2)+0))
# print(np.average([x[turningpoints(fx)[0]],x[turningpoints(fx)[0]-1]]))


def u_theta_distrib(u, theta, v, T, specie):
    
    m_chi = m_chi_jup
    if specie =='H2O':
        mass = 2*m_H+m_O
    if specie == 'hydrogen':
        mass = m_H
    # if specie =='2hydrogen':
    #     mass = 2*m_H
    # if specie =='oxygen':
    #     mass = m_O
    if specie=='helium':
        mass = m_He
    if specie == 'idealised':
        mass =1
    mu = m_chi_jup/mass       
    
    
    temp_vel = np.sqrt(2*k_B*T/m_chi)
    # print(f'temperature dependent velocity is:{temp_vel}')
    v_mag = np.sqrt(pow(v[0],2)+pow(v[1], 2)+pow(v[2],2))

    first_factor = np.power(u, 2)*pow((pow(v_mag, 2)+np.power(u, 2)-2*u*v_mag*np.cos(theta)),((1/2)+n))
    # print(f'first factor is {first_factor}')
    scnd_factor = np.exp(-np.power(u,2)/(pow(temp_vel,2)*mu))
    # print(f'second factor is {scnd_factor}')
    return first_factor*scnd_factor



##rejection method for theta u selection
def u_theta_selector(num, v_vect, T, specie):

    # u_lin = np.linspace(0, 20000, num)
    u_lin = np.linspace(0, 10000, num)
    theta_lin = np.linspace(0, 1.01*np.pi, num)
    # f=u_theta_distrib(u_lin, theta_lin, v_vect, T, specie)
    testing2 = u_theta_distrib(u_lin, np.pi, v_vect, T, specie)
    # print(f'testing is {testing2}')

    max_u = np.average([u_lin[turningpoints(testing2)[0]],u_lin[turningpoints(testing2)[0]-1]])
    # print(f'max speed is : {max_u}')
    testing = u_theta_distrib(max_u, theta_lin, v_vect, T, specie)
    max_theta = np.average([theta_lin[turningpoints(testing)[0]],theta_lin[turningpoints(testing)[0]-1]])
    # print(f'turning points for theta are at :{max_theta}')

    max_f = u_theta_distrib(max_u, max_theta, v_vect, T, specie)

    #finding turning points in function
    # distribution = u_theta_distrib(u, theta, v_vect, T)/max_f
    variables = []
    accept=[]
    for i in range(num):
        acceptance_prob = np.random.uniform(0,1)
        # u = np.random.uniform(0, 20000)
        u = np.random.uniform(0, 10000)
        theta = np.random.uniform(0, np.pi)
        
        if acceptance_prob <= u_theta_distrib(u, theta, v_vect, T, specie)/max_f:
            variables.append((u, theta))
        else:
            variables.append(-1)
   
    for v in variables:
        if type(v) == tuple:
            accept.append(v)
    # print(f'length of final accept list is {len(accept)}')
    # print(accept)
    
    ###PLOTTING
    # j=seaborn.jointplot(x=[t[0] for t in accept], y=[t[1] for t in accept], kind='hex', ratio=2, color='xkcd:sage', joint_kws=dict(gridsize=30), marginal_kws=dict(stat='density'))#,, label=r'Distribution for $10^6$ $u-\theta$ pairs')
    # # # # # plt.scatter(x=[t[0] for t in accept], y=[t[1] for t in accept], color='xkcd:brick', label=r'True Distribution: $n_N (\theta)$')
    # plt.xlabel(r'$u$ ($m.s^{-1}$)', fontsize=35) 
    # plt.ylabel(r'$\theta$ (rad)', fontsize=35) 
    # plt.ylim(bottom=0)
    # plt.xlim(left=0, right=6000)
    # plt.xticks(fontsize = 30) 
    # plt.yticks(fontsize = 30)
    # # # # plt.ylabel(r'Distribution $n_(r_\chi)$')
    # # # plt.title('Rejection Method for selection of Nuclei parameters using $10^6$ points', fontsize=30, wrap=True)
    # # plt.figure(100)
    # # plt.plot(theta_lin, f, color='xkcd:pinkish', label='scattering angle')
    # # plt.plot(u_lin, f, color='xkcd:cornflower blue', label='speed')

    # # plt.legend(prop={'size':18})
    # plt.show()
    # u_selected = accept[np.random.randint(len(accept)-1)][0]
    # theta_selected = accept[np.random.randint(len(accept)-1)][1]
    return np.array(accept)
# u_theta_selector(pow(10,6), np.array([3498.56670798, 3722.71667835, 7654.32945363]), 1000, 'H2O')

##OLD method: selection function for nuclei speed and angular scattering
def u_theta_selector_deprecated(num, v_vect, T, specie):
    
    samples = []
    max_value = 1
    max_of_distrib= u_theta_distrib(5, np.pi, v_vect, T, specie)
    # print(max_of_distrib)
    #selection of candidate values. Need to slect a better range for u
    while len(samples)<num:

        u_candidate = np.random.uniform(0,5000)
        theta_candidate = np.random.uniform(0,np.pi)
        acceptance_prob = np.random.uniform(0,max_value)
        
        value = u_theta_distrib(u_candidate, theta_candidate, v_vect, T, specie)
        normalised_value=value/max_of_distrib
        
        if acceptance_prob < normalised_value:
            
            samples.append((u_candidate, theta_candidate))
            # print(f'theta:{theta_candidate}, distribution is {u_theta_distrib(u_candidate, theta_candidate, v_vect)}')
            # Update max_value for more efficient rejection sampling
            max_value = max(max_value, normalised_value)

    return np.array(samples)
    
# num_samples = pow(10, 4)
# samples = u_theta_selector_deprecated(num_samples, np.array([2784.5575543195362, 5855.9275485959475, 2415.550264051815]))

# plt.figure(300)
# PLOTTING the samples
# g=seaborn.jointplot(x=samples[:, 0], y=samples[:, 1], color='xkcd:cornflower blue', label=r'$10^5$ samples', alpha=0.2, ratio=2, marginal_kws=dict(stat='count'))#, kde=True))


# # # # g.plot_joint(seaborn.histplot, color="r", stat='density', kde=True)
# g.ax_joint.set_xlabel(r'Nuclear Speed u ($m.s^{-1}$)', fontsize=14)
# g.ax_joint.set_ylabel(r'Scattering Angle $\theta$ (rads)', fontsize=14)
# g.fig.subplots_adjust(top=.9)
# g.fig.suptitle(r'Samples from 2D Distribution of Hydrogen Nuclei variables (u, $\theta$)', fontsize=20)
# g.ax_marg_x.set_xlim(0, 4500)
# g.ax_marg_y.set_ylim(0, 3.3)

# plt.figure(200)
# a = seaborn.kdeplot(x=samples[:, 0], y=samples[:, 1], fill=True)
# a.set_xlim(0, 4500)
# a.set_ylim(0, 3.3)
# a.set_ylabel(r'Scattering Angle $\theta$ (rads)')
# a.set_xlabel(r'Nuclear Speed u ($m.s^{-1}$)')
# a.set(title=r'Samples from 2D Distribution of Hydrogen Nuclei variables (u, $\theta$)')

# plt.show()

#Calculation of nuclei velocity
#calculation of cartesian vector coordinates by generating random second angle and then transforming from spherical to cartesian
def calc_hydr_vel(v_vect, T, specie):
    # max = pow(10,6)
    #low max to test, needs setting back to 10^6 for good definition and less noise
    max = pow(10,4)
    #getting initial angle and speed
    m = np.random.randint(0, max)
    selection= u_theta_selector(max, v_vect, T, specie)
    
    sample_size = len(selection)-1
    
    u_speed, theta = selection[np.random.randint(0,sample_size)][0], selection[np.random.randint(0,sample_size)][1]
    
    #generating second angle
    phi = np.random.uniform(0, 2*np.pi)
    
    u_vect_cartesian = conversion_sphertocart(u_speed, theta, phi)
    
    return u_vect_cartesian

# test = calc_hydr_vel(np.array([8179.26397901, 19140.44053419, 16681.50315698]), 1000, 'helium')
# # test = calc_hydr_vel(np.array([5e-12, 6e-11, 9e-11]), 1, 'idealised')
# print(test)

#DEPRECATED and incorrect method: selection of random nuclei and determination of its velocity. Done by rearranging the formula for the dot product operation between 2 vectors.
def calc_nuclei_vel(samples, v_vect):
    
    
    candidates = int(np.random.uniform(0, len(samples)-1))
    
    u,theta = samples[candidates][0], samples[candidates][1]
    
    u_vect = (u*magnitude(v_vect)*np.cos(theta))/v_vect
    v_rel = pow(pow(magnitude(v_vect), 2)+pow(u, 2)-2*u*magnitude(v_vect)*np.cos(theta),1/(2+n))
    mag_u = v_rel - magnitude(v_vect)    
    
    # print(mag_u)
    
    return u_vect, u, theta

# test = calc_nuclei_vel(samples, np.array([27845.575543195362, 58559.275485959475, 24155.50264051815]))
# print(test) #code works and gives a reasonable value, units are obviously off as results are very small. This was flagged with results for the speed u as well.