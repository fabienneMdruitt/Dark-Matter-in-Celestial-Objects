"This file was used mostly for testing, but includes code created to plot fig.1 (showing evolution of Knudsen number with respect to the Dark matter particle mass in different celestial objects) and table III (showing calculations of the Knudsen number in our Young Jupiter models using different methods)."

# from math import log10

# from matplotlib.lines import lineStyles
import second_sphere as sp
import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt


#alpha value set to that calculated by G&R for a mass ratio of 1
# alpha = 2.319
# r_chi = sp.effective_length(1, sp.rho_sho_jup, sp.m_chi_jup)
# array = np.linspace(0.0,2.5, pow(10,3), endpoint=True)

# def gravitational_potential(r):
#     phi = 4*np.pi*pow(10,-13)*sp.G*np.power(r,5)/3
    
#     return phi

# def temperature_calc(r): 
#     return 1.65-0.65*r

# def density_lte(linspace:np.ndarray):
#     temperature_r = temperature_calc(linspace)
#     phi_r = gravitational_potential(linspace)
    

#     integral_fact = (sp.m_chi_jup*phi_r/(sp.k_B*np.trapz(temperature_calc(linspace),linspace)))
#     exponential_fact = np.exp(-integral_fact)

#     density_fact = 4*np.pi*np.power(linspace,2)*np.power((temperature_r/temperature_calc(0)), ((3/2)-alpha))/(np.power(r_chi,4)*pow(np.pi,3/2))
    
#     linspace = exponential_fact*density_fact
#     return linspace 

# test = density_lte(array)

#constant variables

# small_cross_section = pow(10,-30)
# mass_hydrogen =1.67*pow(10,-27)
# mass_oxygen =2.67*pow(10,-26)
# mass_ice = 2*mass_hydrogen+mass_oxygen
# mass_helium = 6.6465* pow(10,-27)
# mass_iron = 2.29*pow(10,16)
# mass_dm = np.linspace(1.783*pow(10,-27), 50*1.783*pow(10,-27)) # range from 1GeV to 50GeV
# log_dm = np.log10(mass_dm/(1.783*pow(10,-27))) #normalised mass
# ## Earth variables
# # earth_mass =  5.972 * pow(10,24)
# # earth_radius = 6371 * pow(10,3)
# # earth_temp = 6150
# # earth_density = sp.density_calc(earth_mass, earth_radius)
# # earth_numdensity = earth_density/mass_iron
# # l_chi_earth = 1/(small_cross_section*earth_numdensity)
# # r_chi_earth = sp.effective_length(earth_temp, earth_density, mass_dm)
# # K_earth = l_chi_earth/r_chi_earth

# ##jupiter core variables
# jup_mass = 2*5.972 * pow(10,25)#1.898*pow(10,27)
# jup_radius = 3.5*pow(10,6)#69911*pow(10,3) [to get radius of 1.64]
# jup_temp = sp.T_core[0]#25*pow(10,3)
# jup_density = sp.density_calc(jup_mass, jup_radius)


# jup_numdensity = jup_density/mass_ice
# l_chi_jup =  1/(small_cross_section*jup_numdensity)
# r_chi_jup = sp.effective_length(jup_temp, jup_density, mass_dm)

# K_jup=l_chi_jup/r_chi_jup
# # print('\n jupiter core:', K_jup[-1])
# ##jupiter envelope variables
# jup_env_mass = 297*5.972 * pow(10,24)
# jup_env_radius = 11*pow(10,7)
# jup_env_temp = sp.T_envelope[0]
# jup_env_density = sp.density_calc(jup_env_mass, jup_env_radius)


# jup_env_numdensity = jup_env_density/((mass_hydrogen+mass_helium)/2)
# l_chi_jup_env =  1/(small_cross_section*jup_env_numdensity)
# r_chi_jup_env = sp.effective_length(jup_env_temp, jup_env_density, mass_dm)

# K_jup_env=l_chi_jup_env/r_chi_jup_env
# # print('\n jupiter envelope',K_jup_env[-1])

# #Sun variables
# sun_mass = 1.989*pow(10,30)
# sun_radius = 696340*pow(10,3)
# sun_density = sp.density_calc(sun_mass, sun_radius)
# sun_numdensity = sun_density/mass_hydrogen
# sun_temp = 15*pow(10,6)
# l_chi_sun = 1/(small_cross_section*sun_numdensity)
# r_chi_sun = sp.effective_length(sun_temp, sun_density, mass_dm)
# K_sun = l_chi_sun/r_chi_sun
# # print('\n sun',K_sun[-1])

# #logs

# log_core = np.log(K_jup)
# log_sun = np.log(K_sun)
# # log_earth = np.log(K_earth)
# log_envelope = np.log(K_jup_env)


# plt.figure(100)
# plt.plot(log_dm, log_sun, color ='xkcd:ochre', marker='*', linestyle='dashed', label='Sun')
# plt.plot(log_dm, log_core, color='xkcd:dark pastel green', marker='o', linestyle='dashed', label='Jupiter core')
# plt.plot(log_dm, log_envelope, color ='xkcd:vivid purple', marker='x', linestyle='dashed', label='Jupiter envelope')

# # plt.plot(mass_dm, log_earth, color ='xkcd:grass green', linestyle='dashed', label='Earth')
# # plt.plot(array, test, color='xkcd:sage', label='LTE')
# plt.xlim(left=0)
# # plt.suptitle('Radially dependent number density in LTE regime', fontsize=30)
# # plt.ylim(bottom=0)
# plt.xticks(fontsize = 30) 
# plt.yticks(fontsize = 30) 
# # plt.xlabel(r'$r/r_{iso}$ (m)', fontsize=25)
# # plt.ylabel(r'$4.\pi r^2 n_{LTE}(r)$', fontsize=25)
# plt.ylabel(r'log($K$)', fontsize=35)
# plt.xlabel(r'log($m/m_\chi$)', fontsize=35)

# plt.legend(prop={'size':18})
# plt.show()
print('single layer concatenate')
r = np.linspace(0, pow(10,4)+70*pow(10,6), 5000,endpoint=True)
T = [pow(10,3)*0.1+pow(10,2)*0.9]
mass = 2*5.972 * pow(10,25)+297*5.972 * pow(10,24)
rho_sho = [3*(0.1*(2*5.972 * pow(10,28))/(4*np.pi*((pow(10,4)*pow(10,2))**3))+ 0.9*(297*5.972 * pow(10,27))/(4*np.pi*(((pow(10,4)+70*pow(10,6))*pow(10,2))**3)))]
density = [3*mass*(0.1/(4*np.pi*((pow(10,4))**3))+ 0.9/(4*np.pi*((pow(10,4)+70*pow(10,6))**3)))]#[3*mass/(4*np.pi*70*(10**6))**3]*900
density_core = 3*(2*5.972 * pow(10,25))/(4*np.pi*((pow(10,4))**3))
density_env = 3*(297*5.972 * pow(10,24))/(4*np.pi*((pow(10,4)+70*pow(10,6))**3))
print(rho_sho)
# sp.position_selector(10,T,rho_sho, sp.m_chi_jup)
r_chi= sp.position_selector(pow(10,1),T, rho_sho, sp.m_chi_jup)[1] #670224.1042037351#1383482.6551906224# 3089183
l_val = pow(0.9*sp.cross_section_jup*density_env*(1/(sp.m_H+sp.m_He))+0.1*sp.cross_section_jup*density_core*(1/(2*sp.m_H+sp.m_O)),-1)#+ sp.cross_section_jup*sp.density_core[0]*(1/(2*sp.m_H+sp.m_O)), -1)
print(l_val)
K_val = l_val/r_chi
print(f'Value for Knudsen number is: {K_val}, and value for scale radius is {r_chi}')

# dt=pd.read_csv('C:/Users/druit/Desktop/Uni Documents/Modules year 4/Dissertation/YoungJupiterValues-106cols.csv')
# print(dt['position'][0][int(5*len(dt['position'][0])/8):int(5*len(dt['position'][0])/8)+5000])
# idealised_radius = 2.5
# idealised_mass=pow(10,7)
# sp.density_calc(idealised_mass,idealised_radius)