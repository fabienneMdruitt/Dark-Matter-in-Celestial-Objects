from __future__ import print_function, division
from statistics import linear_regression
from scipy import special, integrate, stats, optimize
import numpy as np
import seaborn as sb
from bisect import bisect

import re
import matplotlib.pyplot as plt
from sklearn import linear_model
from collision import magnitude


from target_tau import pick_target_tau

###global constants
G = 6.67*pow(10, -11)

k_B = 1.380649* pow(10,-23) #J/K 8.617333262 ev/K
jup_esc_vel = 59*pow(10,5) #escape velocity in m/s

#########

###idealised constants
# m_N = 1
# m_chi = 1
# mu = m_chi/m_N #ratio of mass of nuclei and mass of WIMP

# T_c = 1 ##temperature at solar core
# #idealised object constants mass, radius and density of planet in SI units
# rho_sho = pow(10,-13)
# rho = rho_sho/m_N
# idealised_radius = 2.5
# idealised_mass=pow(10,7)
# # ## initial arrays

# r = np.linspace(0,2.5,1000,endpoint=True)

# ##constant temperature
# # T = [T_c]*1000
# #linear gradient
# T= 1.65-0.65*r
# n = [rho]*1000

#########

### Variables for Young Jupiter model

#masses
# conversion = 1.783*pow(10,-27)  #factor to convert kg into GeV
# # conversion =5.625 * pow(10,26)
# m_H = 1.6735575*pow(10, -27)/conversion #mass of hydrogen nucleus is 1.67×10−27kg
# m_O = 2.67*pow(10,-26)/conversion
# m_He = 6.6465* pow(10,-27)/conversion

# m_chi_jup = 1*1.783*pow(10,-27) #mass of WIMP in GeV
m_H = 1.6735575*pow(10, -27)#*pow(10,3)#mass of hydrogen nucleus is 1.67×10−27kg
m_O = 2.67*pow(10,-26)#*pow(10,3)
m_He = 6.6465* pow(10,-27)#*pow(10,3)

m_chi_jup = 1*1.783*pow(10,-27)#*pow(10,3) #mass of WIMP in kg
#radii in centimetres, temperature in Kelvin, mass in kg

r_core = np.linspace(0,12*pow(10,6),800,endpoint=False)
T_core = [pow(10,4)]*800
#mass core is 19.98 earth masses according to jupiter paper
mass_core = 2*5.972 * pow(10,25)
rho_sho_core = [3*mass_core*pow(10,3)/(4*np.pi*np.power(r_core[-1]*pow(10,2),3))]*800 
density_core = [3*mass_core/(4*np.pi*np.power(r_core[-1],3))]*800

r_envelope = np.linspace(12*pow(10,6), 12*pow(10,6)+9*pow(10,7), 1000,endpoint=True)
T_envelope = [pow(10,3)]*1000
#mass of enveloppe is 297 earth masses according to jupiter paper
mass_envelope = 297*5.972 * pow(10,24)
rho_sho_envelope = [3*mass_envelope*pow(10,3)/(4*np.pi*((9*pow(10,9))**3))]*1800 #g.cm^-3
density_envelope = [3*mass_envelope/(4*np.pi*((9*pow(10,7))**3))]*1000
# print(f'core density is {rho_sho_core[0]}, envelope density is: {rho_sho_envelope[0]}') #density is in g/cm^3
# print(r_envelope[-1]/(69911000))
r_jup = np.concatenate((r_core, r_envelope), axis=0) 
T_jup = np.concatenate((T_core, T_envelope), axis=0)
mass_jup = mass_core+mass_envelope

radius_jup = r_envelope[-1]
rho_sho_jup = [rho_sho_envelope[0]]*1800#np.concatenate((rho_sho_core, rho_sho_envelope), axis=0)
density_jup = np.concatenate((density_core, density_envelope), axis=0)

cross_section_jup=pow(10,-35) #cm^-2, initially set to 10^-27

#########
def calc_mu(m_chi, m_N):
    return m_chi/m_N

def density_calc(objectmass, radius):
    n_0 = 3*objectmass/(4*np.pi*pow(radius,3))
    print(f'density has been defined as {n_0}')
    return n_0
#calculation of cross-section
def cross_section_calc(density, K, r_chi):
    l = K*r_chi
    cross_section = 1/(l*density)
    print(f'cross section has been defined as {cross_section}')
    return cross_section


# #Creation of idealised body. Parent body of instantiation is None, we consider this idealised body to be the centre of its own system
# idealised_body = Body(None, k=G* u.Newton*(1* u.m**2)/(1 * u.kg**2)*test_idealised_mass, name='Idealised', R=test_idealised_radius)


### Selection function: takes a value for r and returns an index position to choose in the arrays. Decision should be weighted. If r_val not in r, returns 2 index 
# positions and their respective weightings.

def index_selection(r_val, array):

    ## test to catch that r_val is valid 
    if r_val<array[0] or r_val>array[len(array)-1]:
        # print("Input value outside of the range")
        return False
    
    #if r_val is in array there is no need to do any calculations, you can directly chose index
    
    if r_val in array:
        index = np.where(array==r_val)
        ratio = 1
        # print(index[0][0])
        # print('value is in array')
        
        return index, ratio
    
    #if r_val is not is not in array but is still within the range of the array, check which 2 values in array enclose r_val. Decide the weightings of each value compared with r_val
    # and calculate
    
    else:
        dual_index = bisect(array, r_val)
        
        weight_lb = round(r_val/array[dual_index-1], 5)
        weight_hb = round(r_val/array[dual_index], 5)
        ratio_hblb = weight_lb/weight_hb
        
        if ratio_hblb > 1:
            ratio = 1-ratio_hblb
            strong = dual_index-1
            weak = dual_index
            
        else:
            strong = dual_index
            weak = dual_index-1
        dual_index = [strong, weak]
        # print(f'dual index: {dual_index}, weight low bound: {weight_lb}, weight high bound: {weight_hb}, ratio: {ratio_hblb}')

        return dual_index, ratio_hblb
# print(index_selection(11984889, r_jup))
## function for calculating values from the arrays if r_val not in r. Should only be called in that specific case: no can include both case of r_val being in array
# or not

def arraybased_value_calc(array, index, ratio):
    

    #r_val in r
    if len(index)==1:

        val = array[index[0][0]]
        # print('r_val is in r')
    
    #r_val not in r
    else:
        
        val_strong = array[index[0]]
        val_weak = array[index[1]]
        
        val = abs(val_strong*(1-ratio)+val_weak*ratio)

        # print('r_val not in r')
    
    return val
##print(arraybased_value_calc(T_jup, [798, 799], 1.001250012500125))

## main function collecting all calculations to give values for every array-based variable

def random_sign():
    return 1 if np.random.random() < 0.5 else -1

def main_val_calc(r_val, v_val, array, T, n, rho_sho, velocity_given=False ):
    
    
    '''
    DEPRECATED EXPLANATION: For the moment, to not complicate the code too much, the assumption that the particle is always rotating clockwise and always 
    travelling towards the outskirts of the planet. At a later date, negatives will need to be introduced within code to symbolise
    particles travelling towards the planet's core or anti-clockwise. Radially, the particle needs to be confined to the escape 
    velocity. This simplifies the calculations as there is no need to take into account the x, y and z values to get the radial velocity.
    Because of dependencies, we have to create the spherical coordinates randomly instead of having them depend on the cartesian coordinates.
    '''
    #code if we are in first loop of MC and there is no velocity given
    if velocity_given == False:
        v_vect_cart = np.array([np.random.random()*jup_esc_vel*random_sign(),np.random.random()*jup_esc_vel*random_sign(),np.random.random()*jup_esc_vel*random_sign()])
        while magnitude(v_vect_cart) > jup_esc_vel:
            v_vect_cart = np.array([np.random.random()*jup_esc_vel*random_sign(),np.random.random()*jup_esc_vel*random_sign(),np.random.random()*jup_esc_vel*random_sign()])
    else:
        v_vect_cart = v_val

    
    if index_selection(r_val, array) != False:
        index, ratio = index_selection(r_val, array) # type: ignore
            
        index = index
        
    else:
        return False
    
    temp_val = arraybased_value_calc(T, index, ratio)
    density_val = arraybased_value_calc(n, index, ratio)
    rho_sho_val = arraybased_value_calc(rho_sho, index, ratio)
    v_vect_val = v_vect_cart
    # print(index, ratio)
        # print(v_vect_val)
    return v_vect_val, r_val, temp_val, density_val, rho_sho_val
# print(main_val_calc(1.7654))
### calculating w for 1 specie

'''
Equation for w for sigma constant is w = 2*sigma_0*n*v_T*sqrt(mu)*[(y+1/(2y))erf(y)+(1/sqrt(pi))*exp(-y^2)]
'''
  
def calculate_w_const(r_val,v_val,velocity_given,sigma_0, array, temp, density, rho_sho,specie):
    
    if specie =='hydrogen':
        mass = m_H
    if specie == 'H2O':
        mass = 2*m_H+m_O
    # if specie =='2hydrogen':
    #     mass = 2*m_H
    # if specie =='oxygen':
    #     mass = m_O
    if specie=='helium':
        mass = m_He
    if specie == 'idealised':
        mass =1
    # print(mass)
    mu = m_chi_jup/mass
 
    if main_val_calc(r_val,v_val,array, temp, density, rho_sho, velocity_given) == False:
        print('error: unable to calculate the values. Please ensure that you have valid parameters.')
        return 'error: unable to calculate the values. Please ensure that you have valid parameters.'
    else:
        
        v_vect_val, r_val, temp_val, density_val, rho_sho_val = main_val_calc(r_val, v_val, array, temp, density, rho_sho, velocity_given) # type: ignore
        
        num_density_val = density_val/mass #given in m^-3
        temp_vel = np.sqrt(2*k_B*temp_val/(m_chi_jup))

        y_sqrd = pow(np.linalg.norm(v_vect_val/temp_vel), 2)/mu
        sigma_m = sigma_0
        ## calculation of a single w
        w = 2 * sigma_m*num_density_val*temp_vel*np.sqrt(mu)*((np.sqrt(y_sqrd)+1/(2*np.sqrt(y_sqrd)))*special.erf(np.sqrt(y_sqrd)) + (1/(np.sqrt(np.pi)))*np.exp(-y_sqrd))

        # adding vect_val and r_val to return so that new position and velocity vector can easily be calculated
        return w, v_vect_val, r_val, temp_val, density_val, rho_sho_val
    ## sum in case number_of_species>1: not needed, the sum is done in section calculating the optical length

##print(calculate_w_const(3497189, np.array([-2142.21363642,  2635.83153037, -3219.96514693]), True, cross_section_jup, r_jup, T_jup, density_jup, rho_sho_jup, 'H2O'))   

# print(calculate_w_const(5*pow(10,7), np.array([-2020.45679008, -479.26660591,  1000.77826344]), True, cross_section_jup, r_jup, T_jup, density_jup, rho_sho_jup, 'helium'))  
### There is no reason to calculate the w_i within the optical path function despite us doing the sum there. The w_i should be calculated beforehand(perhaps with another main function
# and not directly within code block), and put into a format that is easy to handle by optical path function. Best format would be an array or dictionnary (so that optical
# path function can call the w_i it wants whilst know what specie it refers to). Easier to not create a whole other function for this.

#09/11/23 BEST OPTION: create a function that calculates for specific values of r_val and only 1 cross-section (perhaps to get updated to multiple cross-sections).
#Block within code will then calculate the dictionary containning all the cross-sections and r_values. 

### MAIN w function, returns dict
#assumption that values given for r are always within range

def main_w_calc(r_values:list, velocity_value,velocity_given, specie:str, cross_section, array, temp, density, rho_sho):

    w_values={}
    for i in r_values:
        # print(f'result for calculate_w_const is {calculate_w_const(i, cross_section)}')
        w, v_init, r_init, temp_velocity, density_val, rho_sho_val = calculate_w_const(i,velocity_value,velocity_given, cross_section, array, temp, density, rho_sho, specie)
        # print(f'main_val_calc: value for w is {w}')
        w_values.update({f'w_{specie}_{r_values.index(i)}': [w, v_init, r_init, temp_velocity, density_val, rho_sho_val]})
        # v_values.update({f'w_{specie}_{r_values.index(i)}': v_init})
        # rad_values.update({f'w_{specie}_{r_values.index(i)}': r_init})
    return w_values

### all calculations of w will be done using the same list of r_values. This makes it easier with the sums in the optical path function as all indexes are linked 
# despite being from different species.
# dict for hydrogen values of w
# hydrogen = main_w_calc([7*pow(10,7)], np.array([-817.17273235, -1785.04387157,  -396.22554864]), True, 'hydrogen', cross_section_jup,r_jup, T_jup, density_jup, rho_sho_jup)
# helium = main_w_calc([7*pow(10,7)], np.array([ -817.17273235, -1785.04387157,  -396.22554864]), True, 'helium', cross_section_jup,r_jup, T_jup, density_jup,rho_sho_jup)
# hydrogen = main_w_calc([5*pow(10,4)], np.array([ -817.17273235, -1785.04387157,  -396.22554864]), True, 'hydrogen', cross_section_jup,r_jup, T_jup, rho_sho_jup)
# oxygen = main_w_calc([0.5*pow(10,4)], np.array([8.65817462e-11, 2.69116750e-10, 1.65796555e-10]), True, 'oxygen', cross_section_jup,r_jup, T_jup, rho_sho_jup)
# print(hydrogen)
# H2O = main_w_calc([7*pow(10,4)], np.array([-2817.17273235, -4785.04387157,  -396.22554864]), True, 'H2O', cross_section_jup,r_jup, T_jup, density_jup, rho_sho_jup)
# MAIN = {'hydrogen': {'w_hydrogen_0': [0.10008596909080825, np.array([-0.00386101, -0.00547548, -0.00307676]), 2.9804605263569846]}}
# MAIN = {'H2O': hydrogen}#, 'helium':helium}#, 'test_specie':{'test 16': [-2, np.array([1,1,1]), -4], 'test 45':[-1, np.array([1,1,1]), -6]}}
# MAIN dictionary will have structure {{'hydrogen-dict':{hydrogen dict}},'helium-dict':{helium-dict}...} with internal dictionnaries having structure {'w_r-val': w_r-val,...}
#  to account for velocity changing with time the internal dictionaries can have key-value pairs with lists as values. Each list would contain the different values for w
#  at different velocities.
# print(MAIN)
''' 
optical depth: relates a time interval t to the sum of the total rates omega_i with which a WIMP of velocity v scatters with the ith nuclear species

to create more flexibility in optical length function the input is not the direct dictionary created in main_w_calc but rather the parts of the dict we want to access
Because main dictionary contains values for multiple r, calculation of optical path outputs values for each path corresponding to a different r_val. Perhaps best to
output dict so that it is possible to find by labelling.
'''

slopes = []
intercepts = []
def calculate_opt_path(dictionary: dict[str, dict]): 
    # print(dictionary)
    opt_path = {}
    
    for small_keys in dictionary.keys():
        
        specie = dictionary.get(small_keys) #specie = hydrogen, test specie...
        for i in specie.keys():   #type: ignore
            r_value = i
            
            index = [int(s) for s in re.findall(r'\d+\b', r_value)]#[0] #position of the list associated with specific value of r
            
            element = specie.get(r_value)[0] #type: ignore
            # print(index)

            if type(element) != str:

                if f'r_value_{index}' not in opt_path:     # if list doesn't exist it needs to be initialised
                    opt_path[f'r_value_{index}'] = []
                    
                current_list = opt_path[f'r_value_{index}']
                
                current_list.append(element)
      
    for r_val in opt_path:

        ### selecting target tau and imposing restriction on low values       
        target_tau = pick_target_tau()
        # while target_tau < 1e-6:
        #     target_tau = pick_target_tau()
        # target_tau = 5.16
        # print(f'Target optical depth is: {target_tau}')
        opt_path[r_val] = sum(opt_path.get(r_val)) #type:ignore
        # print(f'target_tau:{target_tau}')
        sum_w = opt_path.get(r_val)
        # print(f'sum of w is {sum_w}')
        #optical path function depends on the sum of the omega values for each r
        def opt_path_funct(t,y):

            dtau_dt = sum_w
            return dtau_dt
        


        
        
           
            #RK45 to get optical path,then add value to opt_path dictionary
        '''
            the parameters are chosen to be the following: opt_path_funct is the equation we are using for the optical length, it
            is given in the initial paper; initial time t_0 is set to 0, initial optical path y_0 is set to 0, t_bound is dt and
            is set to 0.5 (unsure about units, probably seconds). Discussion during supervisor meeting confirmed that 0.5s for dt is 
            too long to be realistic.
        '''
        
        
        t_min = 0.0
        dt = pow(10,50)  ### any dt that is smaller always outputs the same values regardless of the change in value of sum_w
        

        if specie.get(r_value)[2] < 12*pow(10,6): #type: ignore
            max_step = 1e-1
        else:
            max_step = 2e0
        optical_path = integrate.RK45(fun= opt_path_funct, t0=t_min, y0=[0.0], first_step = 1e-20, t_bound=dt, max_step=max_step,atol=1e-7)#, rtol=1e-1,atol=1e-3)
                
        t_values = [optical_path.t]
        tau_values = [optical_path.y[0]] 
      
        ##checking whether integrator should be stopped
        while optical_path.status == 'running':
            optical_path.step()
            # print(f'current value of t is: {optical_path.t}, current value of tau is: {optical_path.y[0]}, target tau is {target_tau}')
            t_values.append(optical_path.t)
            tau_values.append(optical_path.y[0])
            # Check if desired_tau is reached
            
            # print(error)
            if optical_path.y[0] >= target_tau:
                error = (abs(optical_path.y[0] - target_tau)/target_tau)*100
                # print(error)
                
                # time = optical_path.t
                # t_values.append(time)
                # tau_values.append(optical_path.y[0])
                
                # print(f'error is {error}, tau final is {optical_path.y[0]}')
                # print(f'end time points are:{time_points}, end values for tau are: {values}')
                
                #adding values to the optical path dictionary
                if error < 4:
                    # print('target overshot,but within error limit')

                    opt_path[r_val] = t_values[len(t_values)-1]
                else:
                    # print('target overshot by a lot, taking the average of the 2 previous values')
                    #choosing the average of the the 2 values
                    # print((tau_values[len(tau_values)-2]+tau_values[len(tau_values)-1])/2)
                    error2 = (abs(tau_values[-2] - target_tau)/target_tau)*100
                    if error2 < 4:
                        opt_path[r_val] = (t_values[len(t_values)-2] + t_values[len(t_values)-1])/2
                    else:
                        opt_path[r_val] = (t_values[len(t_values)-3] + t_values[len(t_values)-2])/2

                # f, ax = plt.subplots()
                # plt.scatter(t_values, tau_values, s=50, color= 'xkcd:crimson')
                # plt.xlabel(r'Time t (s)', fontsize=35)
                # plt.xlim(xmin=0)
                # plt.ylim(ymin=0)
                # plt.xticks(fontsize = 30) 
                # plt.yticks(fontsize = 30) 

                # plt.ylabel(r'Optical depth $\tau$', fontsize=35)
                # # plt.title('Calculating Time Until Next Collision with Runge-Kutta Integration Method', fontsize=30, wrap=True)
                # plt.text(0.1, 0.8, f"target tau: {target_tau} \n error: {error}", horizontalalignment='left', size='large', color='black', weight='semibold', transform = ax.transAxes)
  
                # plt.show()
                

                ## creating model for evolution of tau wrt t
                
                slope, intercept = linear_regression(t_values, tau_values)[0], linear_regression(t_values, tau_values)[1]
                slopes.append(slope)
                intercepts.append(intercept)                    
                break
                

            
            # for the moment this test does nothing except from warn you that your value for t might not be accurate. Further calculations could be done so that t=0 or
            # another abnormal value if that's the case. This would allow us to remove t from the final optical path dictionary completely.
            if optical_path.t ==dt and optical_path.y[0] < target_tau:
                print(f'target tau is {target_tau}, and the optical depth reached is {optical_path.y[0]}')
                error = (abs(optical_path.y[0] - target_tau)/target_tau)*100
                if error>4:
                    ValueError('target optical depth not reached and not within limit')
                    opt_path[r_val] = 'Optical path not valid. Error: not able to reach target tau.'
                else:
                #if value is close enough to target tau, just take the value reached
                    opt_path[r_val] = optical_path.y[0]
                    print('target not reached but within limit')
                break
         


             
               
                    ##changing value for optical path
        # print(t_values, tau_values)
                
        #would a normal integration work for this initial constant function that has no dependence on t? No it wouldn't work as well.
        
        # optical_path = integrate.quad(lambda t: opt_path_funct(sum_w), a=t_min, b=dt)
        
        # checking_lst.append(optical_path)



        
    return opt_path

        
    # return opt_path, np.average(checking_lst)

## dictionary containing the list of all values for t for each specific value of r (and also different species when MAIN dictionary gets updated)
# optical_paths = calculate_opt_path(MAIN)
# print(optical_paths)#, '\n', MAIN)


#####CALCULATION OF NEW POSITION AND VELOCITY using equations in appendix of paper. 

#For the moment t used in equations A.1 and A.2 get replaced by the optical length. Choice of creating a singular function for both
#as both depend on initial velocity and initial positions. Initial position/velocity is a vector, optical length is a float.

# print(slopes, intercepts)
def new_coords_cart(optical_path, initial_pos, initial_vel, rho_sho):
    #defining constants omega, A_i, and a_i
    #initial_pos = initial_pos#*pow(10,-2) #conversion from cm to m to be coherent with velocity of particle expressed in m/s
    # print({f'new coords cart:initial position has been converted to metres and is: {initial_pos}'})
    omega = np.sqrt((4/3)*np.pi*G*rho_sho)
    
    a_x = np.arctan(-initial_vel[0]/(omega*initial_pos[0]))
    A_x = initial_pos[0]/np.cos(a_x)
    
    a_y = np.arctan(-initial_vel[1]/(omega*initial_pos[1]))
    A_y = initial_pos[1]/np.cos(a_y)
    
    a_z = np.arctan(-initial_vel[2]/(omega*initial_pos[2]))
    A_z = initial_pos[2]/np.cos(a_z)
    
    #calculating new position and velocity vectors

    new_pos = np.array([A_x*np.cos(omega*optical_path+a_x), A_y*np.cos(omega*optical_path+a_y), A_z*np.cos(omega*optical_path+a_z)])
    # print({f'new coords cart:modified position has been converted back to centimetres and is: {new_pos}'})
    new_vel = [-A_x*omega*np.sin(omega*optical_path+a_x), -A_y*omega*np.sin(omega*optical_path+a_y), 
               -A_z*omega*np.sin(omega*optical_path+a_z)]
    
    return np.array(new_pos), np.array(new_vel)




#converting radius to a position vector that is in cartesian coordinates.
def conversion_sphertocart(r,theta,phi):
    
    x = r*np.cos(theta)*np.sin(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(phi)

    return x,y,z
# print(new_coords_cart(0.00025060000000000284, conversion_sphertocart(2.8015759649914944, np.random.rand()*np.pi, np.random.rand()*np.pi*2), np.array([  -65.31288088, -2186.73356172, -2156.20642331])))
def conversion_carttospher(x:float, y:float, z:float, angles=False):
    
    #use of keplerian orbit function, theta and phi have been added
    # we are only interested in the radius so there is no point in calculating the 2 angles
    r = np.sqrt(pow(x,2)+pow(y,2)+pow(z,2))
    #condition to calculate and return the angles of the spherical coordinate system as well
    if angles != False:
        
        theta = np.arccos(z/r)
        phi = np.arctan2(y,x)
        
        return r, theta, phi
    else:
        return r
# print(conversion_carttospher(-1.68664938, -1.54723906, -2.57183726))
#calculating effective scale length
def effective_length(T, rho, m):

    if type(m) == np.array:
        r_chi = [np.sqrt((3*k_B*T)/(2*np.pi*G*rho*elm)) for elm in m] #type:ignore
        
    else:
        r_chi = np.sqrt((3*k_B*T)/(2*np.pi*G*rho*m))
    return r_chi
# print(effective_length(T_envelope[0], rho_sho_envelope[0], m_chi_jup))
## selection of random position
def position_distrib(r):
    
    r_chi =1
    distribution = (np.exp(-r/r_chi)**2)/pow(r_chi,3)*pow(np.pi, 3/2)
    

    
    # plt.plot(distribution, linestyle='dotted')
    
    # plt.ylim(ymin=0)
    # plt.xlim(xmin=0)
    
    # plt.xlabel('Radial distribution')
    # plt.ylabel('')
    # plt.title('Model radial distribution')

    # print(distribution)
    # plt.show()
    # position = distribution[int(np.random.random()*(len(distribution)-1))]
    
    return distribution
  
def position_selector(num, T_val, rho_sho_val, m_chi_val):
    
    #m_chi_val = m_chi_val * pow(10,-3) #conversion of DM particle mass into kg
    
    
    r_sho= effective_length(T_val[0], rho_sho_val[0], m_chi_val)#calculation of effective scale for simple harmonic oscillator. Set to 1 for idealised scenario

    if type(r_sho)==np.float64: #conversion of r_sho to metres
        r_sho = r_sho*pow(10,-2)
    else:
        r_sho = [r*pow(10,-2) for r in r_sho]
    print(r_sho)
    class heightDistrib(stats.rv_continuous):
        def _pdf(self, r):
            return pow(r,2)*np.exp(-pow(r/r_sho, 2))

        
        def _dpdf(self, r):
            return 2*r*np.exp(-(pow((r/r_sho),2)))+pow(r,3)*(-(2*np.exp(-pow((r/r_sho),2)))/np.power(r_sho,2))

    height_distribution = heightDistrib(momtype=0, name="distribution", a=0, b=7*pow(10,7))
    pos_candidate = np.random.uniform(0,7*pow(10,7), num)
    acceptance_prob = np.random.uniform(0, 1, num)
    
    negs = np.empty(num)
    negs.fill(-1)
    variables = np.where(acceptance_prob <= height_distribution.pdf(pos_candidate)/height_distribution.pdf(r_sho), pos_candidate, negs) # accepted samples are positive or 0, rejected are -1


    accept = np.extract(variables>=0.0, variables)
    reject = num - len(accept)
    # x = np.linspace(0, 11*pow(10,9), num)
    # fx = 2.68*pow(10,-7)*height_distribution.pdf(x)/height_distribution.pdf(r_sho)   

    # # ##CREATING GRAPHS
    # plt.figure(200)

    # plt.plot(x, fx,color='xkcd:crimson', lw=2, label=r'Real Distribution (normalised)')
    # sb.histplot(accept, color='xkcd:cornflower blue', stat='density', label=r'Distribution after selecting $10^6$ radii')
    # plt.xlabel(r'$r$ (m)', fontsize=35)
    # plt.ylim(0)
    # plt.xlim(0, 1.05*pow(10,7))
    # plt.ylabel('Density', fontsize=35)
    # # plt.ylabel(r'Distribution $n(r_\chi)$', fontsize=25)
    # # # plt.title(r'Rejection Method to Select Sample of $10^6$ initial heights ($r_\chi$) for DM particles', fontsize=30, wrap=True)
    # plt.xticks(fontsize = 30) 
    # plt.yticks(fontsize = 30) 
    # plt.legend(prop={'size':18}, loc=1)
    # plt.show()
    # ##demonstrating the rejection technique with a graph
    # plt.figure(300)
    # correct_x = [1.43, 2.4, 0.5]
    # correct_y = [0.4, 0.015, 0.3]
    
    # wrong_x = [1.3, 2.3, 0.45]
    # wrong_y = [0.9, 0.3, 0.6]
    # plt.fill_between(x, fx, step='pre', alpha=0.2, color='xkcd:cornflower blue', label='Function')#type:ignore
    # sb.scatterplot(x=correct_x, y=correct_y, color='xkcd:grey green', label='selected', s=100)
    # sb.scatterplot(x=wrong_x, y=wrong_y, color='xkcd:crimson', label='rejected', s=100)
    # plt.xticks(fontsize = 30) 
    # plt.yticks(fontsize = 30) 
    # plt.legend(prop={'size':30})

    # plt.ylim(0)
    # plt.xlim(0, 2.5)
    # # plt.title('Visualisation of rejection Method', fontsize=45)
    # plt.show()
    return accept, r_sho 
# position_selector(pow(10,6), T_envelope, rho_sho_envelope, m_chi_jup)


#Selection of velocity from Maxwell-Boltzmann distribution
def velocity_selector(r:float, array, m_chi_val):
    #m_chi_val= m_chi_val*pow(10,-3) #conversion into kg of DM particle mass
    index, ratio = index_selection(r, array) #type:ignore

    temperature = arraybased_value_calc(T_jup, index, ratio)

    speed = np.sqrt((2 * k_B * temperature) / m_chi_val)
    
    class speedDistrib(stats.rv_continuous):
        def _pdf(self, v):
            return pow(m_chi_val/(2*np.pi*k_B*temperature),1/2)*np.exp(-m_chi_val*pow(v,2)/(2*k_B*temperature))*4*np.pi*pow(v,2)
    velocity_distribution = speedDistrib(momtype=0, name="vel_init")
    # x=velocity_distribution.rvs() 

    # print(velocity_distribution.pdf(speed))
    accept=[]
    velocity = []


    # candidate and acceptance values
    vel_candidate = np.random.uniform(0,32000, pow(10,6))
    # vel_candidate = np.random.uniform(0,1.5*pow(10,-11), pow(10,6))
    acceptance_prob = np.random.uniform(0, 1, pow(10,6)) 
    
    # # print(velocity_distribution.pdf(0))
    negs = np.empty(pow(10,6))
    negs.fill(-1)
        
    variables = np.where(acceptance_prob <= velocity_distribution.pdf(vel_candidate)/velocity_distribution.pdf(speed), vel_candidate, negs) # accepted samples are positive or 0, rejected are -1

    accept = np.extract(variables>=0.0, variables) 
    
    # print(accept)
    for i in range(0,3):
        vel = accept[np.random.randint(len(accept)-1)] 
        velocity.append(vel)
        
        #PLOTTING
    # x = np.linspace(0, 32000, pow(10,6))
    # fx =6.705*pow(10,-5)*velocity_distribution.pdf(x)/velocity_distribution.pdf(speed)
    # # x = np.linspace(0, 1.5*pow(10,-11), pow(10,6))
    # # fx =1.59*pow(10,11)*velocity_distribution.pdf(x)/velocity_distribution.pdf(speed)
    # plt.plot(x, fx,color='xkcd:crimson', lw=2, label=r'Real Distribution (normalised)')
    # # # # # # plt.fill_between(x=x, y1=fx, step='pre', alpha=0.2, color='xkcd:cornflower blue', label='Function')
    # sb.histplot(accept, color='xkcd:cornflower blue', stat='density', label=r'Distribution after selecting $10^6$ speeds')
    # plt.xlabel(r'$v_\chi$ ($m.s^{-1}$)', fontsize=35)
    # plt.ylim(bottom=0)
    # plt.xlim(left=0, right = 32000)
    # plt.xticks(fontsize = 30) 
    # plt.yticks(fontsize = 30) 
  
    # plt.legend(prop={'size':18}, loc=1)
    # plt.ylabel(r'Density', fontsize=35)
    # # # plt.title(r'Rejection Method to Select Sample of $10^6$ initial speeds ($v_\chi$)', fontsize=30, wrap=True)

    # plt.show()  
    return np.array(velocity)
# velocity_selector(1, r,1)
# print(velocity_selector(7*pow(10,6), r_jup, m_chi_jup))
#getting initial velocity and position from MAIN dictionary by assuming particle undergoes SHM until time of next collision
def main_calculations_velpos(main_dictionary, optical_path_dictionary, result = 'dictionary', coords = 'cartesian'):
    optical_path = 0.0
    new_pos, new_vel = 0, 0  
    n=0 
    # print(f'velpos {main_dictionary}')
    for i in main_dictionary.keys():
        if n!= 0:
            # print(f'velpos2: {main_dictionary}')
            return main_dictionary, optical_path
        
        element = main_dictionary.get(i)
        if element is not None:
            
            for j in element.keys(): 

                particle = element.get(j)
                if particle is not None:
                    initial_position = particle[2]

                    initial_velocity = particle[1]
                    density = particle[5]
                    x,y,z = conversion_sphertocart(initial_position, np.random.uniform(0,np.pi), np.random.uniform(0,np.pi*2))
                    # print(f'\n SHM old position is {[x,y,z]}')
                    initial_position_cart = np.array([x,y,z])
                            
                    # finding correspondence between optical path and initial vectors
                    index = [int(s) for s in re.findall(r'\d+\b', j)]#[0]
                    keys_opticalpath = optical_path_dictionary.keys()
                            
                    if f'r_value_{index}' in keys_opticalpath and type(optical_path_dictionary.get(f'r_value_{index}')) != str:
                        optical_path = optical_path_dictionary.get(f'r_value_{index}')
                                # print(f'velpos: optical path dictionary is: {optical_path_dictionary}, optical path is {optical_path}')
                        new_pos, new_vel = new_coords_cart(optical_path, initial_position_cart, initial_velocity, density)
                        if coords != 'cartesian':
                            new_pos = conversion_carttospher(new_pos[0], new_pos[1], new_pos[2], angles = False)

                                    
                        # print(f'SHM: new position is {new_pos}, new velocity is {new_vel}')

                
                        # print(f'new position is {conversion_carttospher(new_pos[0], new_pos[1], new_pos[2])}')
                        particle[2] = new_pos
                        particle[1] = new_vel 
                n+=1

# option should never validate, but just in case, optical path is set to 0 so that detection of error is easy                    
    
    if (type(new_pos) ==int and new_pos ==0) or (type(new_vel) ==int and new_vel ==0) or optical_path == 0.0:
        
        print('ERROR WITH VELPOS: no new values for position or velocity assigned.')
        print(main_dictionary, optical_path_dictionary) 
 
    return main_dictionary, optical_path
                    #decision to not append the new position and velocity to the main_dictionary because the difference is not that big. It is also not very important
                    #as we are more interested in the change in vectors due to collision with nuclei.So we replace the old position within the main_dictionary
                    #completely. Any type of value record will be done within the main collision function. Note that here we don't change the position vector back to 
                    #a singular value for radius as we use cartesian coordinates for the colision. The transformation back into spherical coordinates in time to
                    #draw from lookup table again will be done after calling collision function.
    # print(f'updated dictionary: \n {main_dictionary}')

# print(main_calculations_velpos(MAIN, optical_paths))
#velpos without changing values in dictionary
def velpos_var(initial_pos, initial_vel, density, optical_path_dict:dict, coords='cartesian'):
 
    x,y,z = conversion_sphertocart(initial_pos, np.random.uniform(0,np.pi), np.random.uniform(0,np.pi*2))
    initial_position_cart = np.array([x,y,z])
    keys_opticalpath = optical_path_dict.keys()
    
    if f'r_value_0' in keys_opticalpath and type(optical_path_dict.get(f'r_value_0')) != str:
        optical_path = optical_path_dict.get(f'r_value_0')
        # print(f'velpos: optical path dictionary is: {optical_path_dictionary}, optical path is {optical_path}')
        new_pos, new_vel = new_coords_cart(optical_path, initial_position_cart, initial_vel, density)
        if coords != 'cartesian':
            new_pos = conversion_carttospher(new_pos[0], new_pos[1], new_pos[2], angles = False)
    else:
        optical_path = 0.0
        new_pos, new_vel = 0, 0 
    # print(f'SHM: new position is {new_pos}, old position was {initial_pos}, new velocity is {new_vel}')
    return new_pos, new_vel, optical_path

##Kepplerian treatment of particle temporarily leaving the planet

#simplification of problem by taking m1*m2=m1 because the mass of the particle is negligible compared to the mass of the planet
def gravitational_force(planet_mass, planet_radius):
    return (G*planet_mass)/(planet_radius**2)

def newtonian_exit(vel_0, pos_0, planet='Jupiter'):
    print('Netwonian exit: particle has temporarily left the planet')
    #initialising values to not get errors
    reentry_time=0.0
    reentry_position = 0.0
    reentry_radial_vel = 0.0
    #force felt by particle
    force = gravitational_force(mass_jup, radius_jup)
    #velocity
    if planet == 'Jupiter':
        height_above_planet = pos_0-r_jup[-1]
    else:
        height_above_planet = pos_0-2.5
    vel_r, vel_theta, vel_phi = conversion_carttospher(vel_0[0], vel_0[1], vel_0[2], angles=True)
    
    #Equations of motion
    def radial_position_change(t,y):
        return -(1/2)*force*pow(t,2) +vel_r*t+ pos_0

    
    #integration, initial test of setting initial position to the height above the planet so that target position is 0
    t_values = []
    r_values = []
    t_bound = pow(10,2)
    y0=[0.0]
    sol = integrate.RK45(radial_position_change, t0=0.0, y0=y0, t_bound=t_bound, max_step=1e-2)
    ##checking whether integrator should be stopped
    while sol.status == 'running':
        sol.step()
        t_values.append(sol.t)
        r_values.append(sol.y[0])

        # current_radial_vel = velocity_change(sol.t) 
        # vel_values.append(current_radial_vel)
        # print(f'current value of t is: {sol.t}, current value of radial position is: {sol.y[0]}')

        #checking if max height has been reached and storing the time value when it does
        
        # if len(r_values)>2 and r_values[-2] > r_values[-1] and maxheightcheck ==False:
        #     radial_vel_maxheight = velocity_change(sol.t) 
        #     # print(f'Maximum height reached at time {sol.t}')
        #     time_maxheight = sol.t
        #     maxheightcheck =True
            
        if sol.y[0] < -height_above_planet:
            error = (abs(sol.y[0] - height_above_planet)/height_above_planet)*100
            
            if error<8:
                reentry_time = sol.t
                # print(reentry_time)
                reentry_position = pos_0 + sol.y[0]
                
            else:
                reentry_time = (t_values[-2]+t_values[-1])/2
                # print(reentry_time)
                reentry_position = pos_0 + (r_values[-2]+r_values[-1])/2
            # print(f'calculated position is {sol.y[0]}, goal was {-height_above_planet}position on planet is {reentry_position}')
            # # f, ax = plt.subplots()
            # plt.scatter(t_values, r_values, s=8, color = 'xkcd:grey green')
            # plt.xlabel(r'Time t ($s$)')
            # plt.xlim(xmin=0)
            # # plt.ylim(ymin=0)
            # plt.ylabel(r'Radial speed ($m.s^{-1}$)')
            # plt.title(r"Evolution of particle's radial speed when out of planet ($m.s^{-1}$)")
            # # plt.text(0.1, 0.8, f"target tau: {target_tau} \n error: {error}", horizontalalignment='left', size='small', color='black', weight='semibold', transform = ax.transAxes)
            # plt.show()


            # print(f'reentry position is {reentry_position}')
            break 
   
        #test cases to be defined
        if sol.t == t_bound and sol.y[0] > -height_above_planet:
            #check 
            reentry_time = 0.0
            reentry_position = 0.0
            # print(f'goal was {-height_above_planet}, value reached was {sol.y[0]}')
            raise ValueError('error time taken to re-enter planet is too long')

    
    # timeofdescent =reentry_time-time_maxheight
    # reentry_radial_vel = velocity_change(timeofdescent, 0.0, const=-1)
    evolution_radial_vel = np.gradient(r_values, t_values[-1])-vel_r
    reentry_radial_vel = evolution_radial_vel[-1] 
    # beginning_radial_vel = conversion_sphertocart(evolution_radial_vel[0], vel_theta, vel_phi)
    
    reentry_vel = conversion_sphertocart(reentry_radial_vel, vel_theta, vel_phi)
    
    

    if reentry_time ==0.0 or reentry_position == 0.0 or reentry_radial_vel==0.0:
        raise ValueError('Keplerian orbit error: values were not updated')
    return reentry_position, reentry_vel, reentry_time 

# print(newtonian_exit(np.array([-1.16413250e-11,  1.08518670e-11,  1.16368664e-12]), 2.535092899801998))

def keplerian_radius(theta,a=2.5,e=0.2,r0=3):

    radius = a*(1-e**2)/(1+e*np.cos(theta))-r0
    # radius = theta**3-1
    # return radius

# def keplerian_orbit(vel_0, pos_0):
#     print(f'initial position was {conversion_carttospher(pos_0[0], pos_0[1], pos_0[2])}, initial velocity was {vel_0}')
#     pos_0 = pos_0*u.meter
#     vel_0 = vel_0*u.meter/u.second

#     orbit = Orbit.from_vectors(idealised_body, pos_0, vel_0)
#     print(orbit)
#     print(orbit.r, orbit.v)
#     time = orbit.period
#     b=orbit.propagate(time/2)
#     radius = conversion_carttospher(b.r[0].value, b.r[1].value, b.r[2].value, angles=False)
#     vel = -orbit.v.to(u.m/u.s).value
    
#     '''#position 
#     # r_0, theta_0, phi_0 = conversion_carttospher(pos_0[0], pos_0[1], pos_0[2], angles=True)    
#     # print(theta_0)
    
#     # Instantiate a Keplerian elliptical orbit with
#     # semi-major axis of 2.5 length units,
#     # a period of 0.2 time units, eccentricity of 0.2,
#     # longitude of ascending node of 30 degrees, an inclination
#     # of 90 deg, and a periapsis argument of 131 deg.
#     ke = pyasl.keplerOrbit.KeplerEllipse(2.5, 0.2, e=0.2, Omega=30., i=theta_0*180/np.pi, w=131.0)
#     # t = np.linspace(0, 0.2, 200)
    
#     #getting theta value for when particle re-enters planet
#     # new_theta = optimize.newton(keplerian_radius, np.pi)
    
#     # new_theta = (theta_0 + new_theta)/(2*np.pi)
#     # new_r = keplerian_radius(new_theta, r0=0)
#     # print(new_r)
    

    
#     # Find the nodes of the orbit (Observer at -z)
#     ascn, descn = ke.xyzNodes_LOSZ(getTimes=True)
#     time_reenter = descn[1]
    
#     #     # Calculate orbit radius and velocity at the time of re-entering
#     radius = ke.radius(time_reenter)
#     vel = ke.xyzVel(time_reenter) + vel_0
#     # Calculate the orbit position at the given points
#     # # in a Cartesian coordinate system.
#     # pos = ke.xyzPos(t)    
    
    
    
#     # Plot x and y coordinates of the orbit
#     # fig = plt.figure()
#     # fig.suptitle("Keplerian Orbit", fontsize="x-large")
#     # rad = fig.add_subplot(2, 1, 1)
#     # plt.title("Periapsis (red diamond), Asc. node (green circle), desc. node (red circle)")
#     # plt.xlabel("East ->")
#     # plt.ylabel("North ->")
#     # plt.plot([0], [0], 'k+', markersize=9)
#     # plt.plot(pos[::, 1], pos[::, 0], 'bp')
#     # Point of periapsis
#     # plt.plot([pos[0, 1]], [pos[0, 0]], 'rd')
#     # Nodes of the orbit
#     # plt.plot([ascn[1]], [ascn[0]], 'go', markersize=10)
#     # plt.plot([descn[1]], [descn[0]], 'ro', markersize=10)
    
#     #plot radial position during orbit
#     # plt.plot(t, radius, color='xkcd:cornflower blue')
#     # plt.xlabel("Time (s)")
#     # rad.set_title('Evolution of particle\'s height above object surface during Keplerian orbit.')
#     # plt.ylabel(r"Height $h_chi$")
#     # plt.xlim(0,0.21)
#     # # Plot RV
#     # velocity = fig.add_subplot(2, 1, 2)
#     # plt.xlabel("Time (s)")
#     # plt.ylabel("Radial velocity")
#     # plt.plot(t, vel[::, 2], color='xkcd:grey green')
#     # velocity.set_title('Evolution of particle\'s radial velocity above object surface during Keplerian orbit.')
#     # plt.show()
    
#     #getting values for returning parameters'''
#     time_reenter = time.value
    
#     return radius, vel, time_reenter
# print(keplerian_orbit(np.array([0.43895555, 0.00146178, 0.41054838]), np.array([1.2, 0.5, 2.77])))
## function that calculates the new position and velocity vectors using equations adapted to spherical coordinates. Not used because working with cartesian coordinates
## was deemed a better strategy.

def new_coords_spher(optical_path, initial_pos, initial_vel, rho_sho):
    #defining constants omega, B_i, and b_i
    omega = np.sqrt((4/3)*np.pi*G*rho_sho)
    
    b_r = np.arctan(-initial_vel[0]/(omega*initial_pos[0]))
    B_r = initial_pos[0]/np.cos(b_r)
    
    b_phi = np.arctan(-initial_vel[1]/(omega*initial_pos[1]))
    B_phi = initial_pos[1]/np.cos(b_phi)
    
    b_theta = np.arctan(-initial_vel[2]/(omega*initial_pos[2]))
    B_theta = initial_pos[2]/np.cos(b_theta)
    
    #calculating new position and velocity vectors
    
    r = np.sqrt(pow(B_r*np.cos(omega*optical_path+b_r),2) + pow(B_phi*np.cos(omega*optical_path+b_phi),2) + pow(B_theta*np.cos(omega*optical_path+b_theta),2))
    phi = np.arccos((B_theta*np.cos(omega*optical_path+b_theta))/(r))
    theta = np.arctan((pow(B_phi*np.cos(omega*optical_path+b_phi), 2))/(pow(B_theta*np.cos(omega*optical_path+b_theta), 2)))
    
    r_vel = -omega*(pow(B_r, 2) * np.sin(omega*optical_path + b_r) * np.cos(omega * optical_path + b_r) + pow(B_phi, 2) * np.sin(omega*optical_path + b_phi) * np.cos(omega * optical_path + b_phi)
                    + pow(B_theta, 2) * np.sin(omega*optical_path + b_theta) * np.cos(omega * optical_path + b_theta))
    print(omega*optical_path)
    phi_vel = r*pow(phi, 2) * np.sin(B_theta*(-omega*np.sin(omega*optical_path + b_theta) * np.arccos(omega * optical_path+b_theta) * r + r_vel*B_theta*np.cos(omega*optical_path+b_theta))/ pow(r,2))
    
    # theta_vel = r * np.sin(phi) * (-sec(-2*omega*(np.sin(omega*optical_path+b_phi)*pow(B_theta, 3)*np.cos(omega*optical_path+b_theta) - 2*pow(B_theta,3)*omega*np.sin(omega*optical_path+b_theta)*np.cos(omega*optical_path+b_theta)* pow(B_phi, 2)*pow(np.cos(omega*optical_path+b_phi), 2))/ pow(B_theta*np.cos(omega*optical_path+b_theta), 4)))

    new_pos = [r, phi, theta]
    # new_vel = [r_vel, phi_vel, theta_vel]
    
    # return new_pos, new_vel 
    

                                                                                                                                                                                                  