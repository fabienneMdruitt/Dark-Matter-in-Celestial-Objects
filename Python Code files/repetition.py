
import pandas as pd
import second_sphere as sp
import collision as col
import nuclei as ncl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import time


##Collision-dependent Time tracking##
# def timed_recording(task,time_interval):
#     next_time = time.time()+time_interval
#     while kill==False:
#         print(f'kill is {kill}')
#         time.sleep(max(0, next_time-time.time()))
#         try:
#             task(position_record, velocity_record, new_r, new_dm_vel)
#             # time.sleep(0.25)
#         except Exception:
#             # traceback.print_exc()
#       # in production code you might want to have this instead of course:
#             logging.exception("Problem while recording variables for stationary distributions.")
#     # skip tasks if we are behind schedule:
#         next_time += (time.time() - next_time) // time_interval * time_interval + time_interval
#         # break
 
    
        

# def record(position:list, velocity:list, new_position:int, new_velocity:int):
#     print(new_position)
#     position.append(new_position)
#     velocity.append(new_velocity)

        #######MONTE CARLO#######
samples, r_chi = sp.position_selector(pow(10,6), sp.T_envelope, sp.rho_sho_envelope, sp.m_chi_jup)
# print(r_chi)
# r_chi = sp.effective_length(sp.T_envelope[0], sp.rho_sho_envelope[0], sp.m_chi_jup) #in constant temp scenario r_chi = 0.9941442788591482
alpha = 2.319
init_position_lst = []

def repetition(number:int, array, temp_array, density_array, rho_array, energy_list:list[list], stationary_position_list:list, stationary_velocity_list:list, K):

    bin_number = 100

    #making sure that the bin number is correct for energy_list user wants to pass
    if len(energy_list) == bin_number:
        print('Number of increments in energy list is correct.')
    else:
        print(f'WARNING: the number of bins chosen ({bin_number}) is not the same as the number of increments ({len(energy_list)}) in energy list.')
    
    #simulation time
    time_sim = 0.0
    time_sim_list = []
    
    #simulation_time-dependent recording of variables
    position_tsim = []
    velocity_tsim = []
    #implementation time    
    time_record = []

    position = [samples[int(np.random.random()*(len(samples)-1))]]
    # print(samples)
    velocity_selected = sp.velocity_selector(position[0], array, sp.m_chi_jup)

    # sigma_0 = sp.cross_section_calc(sp.rho, K, r_chi)
    sigma_0 = sp.cross_section_jup

    init_position_lst.append(position[0])
    PARTICLES = {}
    start_time = time.time()


    ##starting the collisions

    for i in range (number): 
        


        time_elapsed = time.time() -start_time
        
        # initialising dictionary if it is the first iteration
        if i ==0:
            if position[0] < 12*pow(10,6):
                print(f'position in core because {position[0]}')
                # specie1 = "2hydrogen"
                # specie2 = "oxygen"
                specie1 = 'H2O'
                specie2 = 'null'
                secondspecie = {'w_null_0':[0, np.array([0,0,0]), 0, 0, 0]}
            else:
                print(f'position in envelope because {position[0]}')

                specie2 = "hydrogen"
                specie1 = "helium"
                secondspecie = sp.main_w_calc(position, velocity_selected, True, specie2, sigma_0, array, temp_array, density_array, rho_array)


            print(f'Process has started')
            firstspecie = sp.main_w_calc(position, velocity_selected, True, specie1, sigma_0, array, temp_array, density_array, rho_array)
            # print(hydrogen)
            MAIN = {specie1:firstspecie, specie2:secondspecie}
            # print(f'repetition {MAIN}')
            optical_paths = sp.calculate_opt_path(MAIN)
            #initialising the variable tracking lists: HARDCODED AND SPECIFIC TO HYDROGEN
            initial_particle = MAIN.get(specie1).get(f'w_{specie1}_0') #type:ignore
            
            #when multiple elements exist we are only interest in 1 side of the dictionary after we have calculated the optical depth
            tracked_element = {specie1: MAIN.get(specie1)}
            position_tsim.append(initial_particle[2]) #type:ignore
            velocity_tsim.append(col.magnitude(initial_particle[1])) #type:ignore
            time_sim_list.append(time_sim)             
        # if particles is already non-null, it can be used    
        else:
            
            MAIN = PARTICLES

            optical_paths = sp.calculate_opt_path(MAIN) #type:ignore
            
            if i == number-1:
                print('simulation has reached end goal.')    
                    


        #taking into account how the optical path would change the velocity
        PARTICLES, time_bf_collision = sp.main_calculations_velpos(tracked_element, optical_paths, result = 'dictionary', coords='spherical')
        # print(f'\n time before next collision is {time_bf_collision}')
        # print('\n first',PARTICLES)

        

        #selection of particles for collision
        
        for l in PARTICLES.keys(): 
        
            element = PARTICLES.get(l) 
            if element is not None:
                delete = []
                for j in element.keys():
                    
                    particle = element.get(j)
                    
                    
                    #only select a particle if it has a valid optical path. Particles with a non-valid optical have their velocity and position set to 0
                    
                    if particle is not None and type(particle[1]) != int:
                        timeout=0.0
                        reentry_time_shm = 0.0
                        position = particle[2]
                        velocity_particle = particle[1]
                        temperature_val = particle[3]
                        density_val = particle[5]
                        
                        
                        # cur_radius = sp.conversion_carttospher(particle[2][0], particle[2][1], particle[2][2], angles=False)
                        # evaporation event: we stop the simulation
                        if (col.magnitude(particle[1]) >= sp.jup_esc_vel) and (position >= sp.r_jup[-1]): #type:ignore
                            
                            #when break is inserted again, line below should have the radius reset to a certain value so as to not throw off the w calculation function
                            #temporary values for new position and velocity so as to test time for full implementation of 10^5 collisions. Without these values code runs into error when calculating w at line 112
                            particle[1] = np.array([15, 15, 15])
                            particle[2] = 1.5

        
                            print(f"simulation ended because of evaporation. Final values are: {PARTICLES}, and time elapsed was: {time_elapsed} s")
                            return 
                        # temporary exit of the planet, we calculate new position and velocity differently
                        if (position>sp.r_jup[-1]) and (col.magnitude(velocity_particle) < sp.jup_esc_vel): #type:ignore
                            # print(f'invalid radius is {new_r}')
                            recal=False
                            #keplerian orbit function from second sphere file
                            reentering_position, reentering_velocity, timeout = sp.newtonian_exit(particle[1], position)
                            
                            #OLD WARNING:temporary values for new position and velocity so as to test time for full implementation of 10^5 collisions. Without these values code runs into error when calculating w at line 112. SHOULD GET UPDATED IN A WAY THAT ALLOWS FOR THE CORRECT RECORDING OF THE NEW VARIABLES BELOW!!
                            # particle[1] = reentering_velocity
                            # particle[2] = reentering_position
                            # particle[1] = np.array([-0.15, -0.15, 0.15])
                            # particle[2] = 2.9
                            
                            ##SHM within planet after reentry
                            path_reentry_shm_temp = sp.calculate_opt_path(PARTICLES) #type:ignore
                            #particle position at re-entry is defined by its radius, particle velocity vector is cartesian
                            new_pos, new_vel, reentry_time_shm_temp = sp.velpos_var(reentering_position, reentering_velocity, density_val, path_reentry_shm_temp, coords='spherical')


                            #making sure that particle is well inside planet
                            temp_position = new_pos
                            recal_pos = temp_position
                            if temp_position>sp.r_jup[-1]: #type:ignore
                                #making sure code doesn't get stuck in loop
                                count = 0
                                recal = True
                                
                                while recal_pos>sp.r_jup[-1] and count<100: #type:ignore

                                    # print(f'particle still outside planet due to SHM at {recal_pos}')
                                    recal_pos = temp_position
                                    
                                    # print(f'recal: {PARTICLES}')
                                    path_reentry_shm_recal = sp.calculate_opt_path(PARTICLES)#type:ignore
                                #particle position at re-entry is defined by its radius, particle velocity vector is cartesian
                                    position_recal, velocity_recal, reentry_time_shm_recal = sp.velpos_var(recal_pos, new_vel, density_val,path_reentry_shm_recal, coords='spherical')
                                    recal_pos = position_recal
                                    count +=1
                                    if count == 100:
                                        position_recal = np.random.uniform(1*pow(10,6), 6*pow(10,6))
                            
                            if recal == True:
                                particle[1], particle[2], reentry_time_shm =  velocity_recal,position_recal, reentry_time_shm_recal
                            else:
                                particle[1], particle[2], reentry_time_shm = new_vel,new_pos, reentry_time_shm_temp                            
 
                            print(f'Temporary exit from sphere of DM particle because particle was at {position}. Time out of planet was {timeout} s and time elapsed was: {time_elapsed} s. Particle re-entered at {particle[2]}.')

                        v_vect = particle[1]
                        #selection of nuclei
                        if position>12*pow(10,6):
                            num = np.random.randint(0,2)
                            if num == 1:
                                nuc_specie='hydrogen'
                            else:
                                nuc_specie='helium'
                        else:
                            nuc_specie = 'H2O'
                        u_vect = ncl.calc_hydr_vel(v_vect, temperature_val, nuc_specie)
                        
                        # #old r is the same as new r but in cartesian coordinates instead of spherical ones. It is only used during the keplerian orbit section
                        # old_r = particle[2]                        
                        #calculating initial kinetic energy for DM particle using classical KE=0.5*m*v^2
                        kin_bfcollision = 0.5*sp.m_chi_jup*pow(col.magnitude(v_vect),2)
                        
                        # performing collision
                        
                        new_dm_vel = col.collision(v_vect, u_vect)
                         #If particle has already left and re-entered the planet its position value is already r. So changing position back to radius only if particle has not exited planet (to find values in lookup table)
                        # if timeout != 0.0: 
                        #     new_r = sp.conversion_carttospher(particle[2][0], particle[2][1], particle[2][2], angles=False)
                        #     particle[2] = new_r

                        # assigning new variable values for the DM particle
                        particle[1] = new_dm_vel
                        
                        #kinetic energy of DM particle after collision and recording data
                        kin_afcollision = 0.5*sp.m_chi_jup*pow(col.magnitude(new_dm_vel),2)
                        
                        energy_transfer = kin_bfcollision - kin_afcollision
                        
                                
                            # print('\n', PARTICLES, '\n')

                            # break                            
                        #recording position and velocity every 2 seconds
                        counter_time = time.time()
                        # print(f'counter_time: {round((counter_time-start_time),2)}, and time list is {time_record}')
                        if round(round((counter_time-start_time),2)%2,2) == 0.0 and round((counter_time-start_time),2) not in time_record:
                            # print(f'counter_time was recorded because: {round((counter_time-start_time),2)%0.1}')
                            stationary_position_list.append(particle[2])
                            stationary_velocity_list.append(col.magnitude(new_dm_vel))
                            time_record.append(round((counter_time-start_time),2))
                        
                        ###RECORDING VARIABLES AFTER EACH COLLISION [account for potential temporary evaporation of particle for time recording]
                        #time
                        time_sim = time_sim+time_bf_collision+reentry_time_shm+timeout #type:ignore
                        time_sim_list.append(time_sim)
                        #recording position and velocity
                        position_tsim.append(particle[2])
                        velocity_tsim.append(col.magnitude(new_dm_vel))
                        
                        #getting which bin the energy value should get put in, placed after the evaporation detection sections to not get errors.
                        bin = int((particle[2]/3)*bin_number) #type:ignore
                        # print(bin, new_r)
                        
                        if bin <=99:
                            energy_list[bin].append(energy_transfer)
                               
                        #updating the scatter rate w_i with the new particle position and velocity. Placed after evaporation conditions as it returns an error if the radius value is too big (greater than the size of the object)
                        if particle[2] < 12*pow(10,6):
                            specie1 = 'H2O'
                            specie2 = 'null'
                            part2 = {'w_null_0':[0, np.array([0,0,0]), 0, 0, 0]}
                        else:
                            specie2 = "hydrogen"
                            specie1 = "helium"
                            part2 = sp.main_w_calc([particle[2]], particle[1], True, specie2, sigma_0, array, temp_array, density_array, rho_array)#.get(f'w_{specie2}_0')

                            

                        part1 = sp.main_w_calc([particle[2]], particle[1], True, specie1, sigma_0, array, temp_array, density_array, rho_array)#.get(f'w_{specie1}_0')
                        PARTICLES = {specie1: part1, specie2: part2}
                        tracked_element = {specie1: PARTICLES.get(specie1)}

                        # print(f'Bottom rep: {PARTICLES}')
 
 
                        # if part is not None:
                        #     particle[0] = part[0]
                        #     particle[3] = part[3]
                
                        
                        
                    
                #remove the particle if it has invalid optical path    
                    else:
                         delete.append(j)
                
                  
                
                for n in delete:     
        
                    element.pop(n)

                    
        # print('\n',PARTICLES)     
    end_time = time.time()

    time_elapsed = end_time -start_time

    print(f'Final particle variables are {PARTICLES}, time elapsed was: {time_elapsed} s')
    return K,energy_list, time_sim_list, position_tsim, velocity_tsim, stationary_position_list, stationary_velocity_list, time_record, PARTICLES


def gravitational_potential(r):
    phi = 4*np.pi*pow(10,-13)*sp.G*np.power(r,3)/(3)
    
    return phi

def temperature_calc(r, scenario = 'idealised'): 
    if scenario == 'idealised':
        return 1.65-0.65*r
    else:
        return pow(10,3)-(pow(10,3)/(7*pow(10,7)+pow(10,6)))*r

def density_lte_calc(linspace:np.ndarray):
    temperature_r = temperature_calc(linspace)
    phi_r = gravitational_potential(linspace)
    

    integral_fact = (sp.m_chi_jup*phi_r)/(sp.k_B*np.trapz(temperature_calc(linspace),linspace))
    exponential_fact = np.exp(-integral_fact)

    density_fact = (19/28)*4*np.pi*np.power(linspace,2)*np.power((temperature_r/temperature_calc(0)), ((3/2)-alpha))/(np.power(r_chi,4)*pow(np.pi,3/2))
    
    linspace = exponential_fact*density_fact
    return linspace 

def luminosity_func(energy:list[list], simulation_time:list):
    
    total_time = simulation_time[len(simulation_time)-1]
    Luminosity_sections = []
    for i in energy:
        bin = energy.index(i)
        if i != []:
            radial_bin_energy = sum(n for n in i)
        else:
            radial_bin_energy = 0.0
        
        luminosity = radial_bin_energy/total_time

        Luminosity_sections.insert(bin, [luminosity])
    max_luminosity = max(Luminosity_sections)
    total_luminosity = sum(l[0] for l in Luminosity_sections)
    return Luminosity_sections, total_luminosity, max_luminosity

def standard_err(x:np.ndarray, num_samples):
    
    s = (np.mean(pow(x,2)) - pow(np.mean(x),2))/num_samples
    return s

energy_transfer_record = [[] for _ in range(100)]
position_record = []
velocity_record = []


#dictionary for the variable recording lists

RECORD = {'energy_transfer_record':[]}

#for i in range (len(K_values)-1):
for i in range(1):
    # print(f'this is simulation number {i} \n')

    #getting list of tracked variables
    #order: return energy_list, time_sim_list, position_tsim, velocity_tsim, stationary_position_list, stationary_velocity_list, time_record, PARTICLES
    #K=0.1
    k,energy_transfer_record, simulation_time, time_dependent_position, time_dependent_velocity, position, velocity, set_time, final_values = repetition(pow(10,5), sp.r_jup, sp.T_jup, sp.density_jup, sp.rho_sho_jup,energy_transfer_record, position_record, velocity_record, 0.1) #type:ignore

    # K=0.6S
    # energy_transfer_record1, simulation_time1, time_dependent_position1, time_dependent_velocity1, position1, velocity1, set_time1, final_values1 = repetition(pow(10,3), energy_transfer_record, position_record, velocity_record, 0.6) #type:ignore
    
    # # # # K=2
    # energy_transfer_record2, simulation_time2, time_dependent_position2, time_dependent_velocity2, position2, velocity2, set_time2, final_values2 = repetition(pow(10,5), energy_transfer_record, position_record, velocity_record, 2) #type:ignore
    
    # energy_transfer_record_{i}, simulation_time_{i}, time_dependent_position_{i}, time_dependent_velocity_{i}, position_{i}, velocity_{i}, set_time_{i}, final_values_{i} = repetition(pow(10,3), energy_transfer_record, position_record, velocity_record, K[i])
    
    #processing position results to get valid histogram values
    # diff_position = np.diff(time_dependent_position)
    # end_sim_time = simulation_time[-1]

    # diff_time = np.diff(simulation_time)
    # weighting=(diff_time/end_sim_time)

    # normed_position = 4*np.pi*np.array(time_dependent_position)/(pow(r_chi,3)*pow(np.pi, 3/2))
    # new_position = time_dependent_position[:len(time_dependent_position)-1]*weighting
    # luminosity_r_bins, total_luminosity, max_luminosity = luminosity_func(energy_transfer_record, simulation_time)
    
    # print(f'lum bins are {[item[0] for item in luminosity_r_bins]}')
    
    data = {'position':[time_dependent_position], 'velocity':[time_dependent_velocity], 'knudsen value': k, 'final value': [final_values]}
    #calculation of the Knudsen number from the cross-section
    l_val = pow(sp.cross_section_jup*sp.density_envelope[0]*(1/(sp.m_H+sp.m_He)),-1)#+ sp.cross_section_jup*sp.density_core[0]*(1/(2*sp.m_H+sp.m_O)), -1)
    K_val = l_val/r_chi
    print(f'Value for Knudsen number is: {K_val}, and value for scale radius is {r_chi}')
    #PLOTTING
    
    # #density
    
    array = np.linspace(0,7*pow(10,7)+4*pow(10,6), pow(10,4), endpoint=True)

    norm_const = 4*np.pi*np.power(array,2)
    
    #Isothermal density
    density_iso = norm_const*np.exp(-np.power(array/r_chi,2))/(pow(np.pi, 3/2)*np.power(r_chi,4))
    
    #LTE density
    # density_lte = density_lte_calc(array)

    #normalisation of radial distribution
    # normalised_distribution = 4*np.pi*np.power(time_dependent_position,2)*np.array(time_dependent_position)/(pow(r_chi,3))
    # normalised_distribution = 4*np.pi*np.power(array,2)*np.array(time_dependent_position[:len(time_dependent_position)-1])/(r_chi**3) #type:ignore
    clean_normalised_distribution = np.array(time_dependent_position)/r_chi
    # normalised_distribution1 = np.array(time_dependent_position1)/r_chi#pow(np.pi, 3/2)
    # normalised_distribution2 = np.array(time_dependent_position2)
    # plt.suptitle(r'normalised exp graph', wrap=True)
    # sb.lineplot(x=array,y=density_iso, color='xkcd:crimson', label='Isothermal')
    # # sb.lineplot(x=array, y=density_nonormed, color='xkcd:dark tan', label='initial distribution')
    # sb.histplot(test_dens2, stat='density', kde=True, color='xkcd:cornflower blue')
    # plt.xlim(0, 2.5)
    # plt.ylim(0, 1)

    # plt.xlabel(r'$r_\chi$ (m)')
    # plt.ylabel(r'density distribution $4 \pi  r^2 n_\chi(r)$')
    # plt.legend()
    # plt.show()
    
    plt.figure(200)
    # sb.histplot(energy_transfer_record, element="step", fill=False, color='xkcd:crimson', stat='density')
    # plt.show()
    
    #velocity

    # v= np.linspace(0, 0.5, pow(10,4))
    # f = np.sqrt(sp.m_chi/(2*np.pi*sp.k_B)*np.exp(-sp.m_chi*pow(v,2)/(2*sp.k_B)))/9
    # plt.figure(100)
    sb.histplot(time_dependent_velocity, element='step', fill = False, color='xkcd:cornflower blue', stat='density', label=f'K={np.round(K_val,3)}')
    # # plt.plot(v, f, color='xkcd:crimson', label='initial distribution')
    plt.xlabel(r'$v_\chi$ $m.s^{-1}$', fontsize=35)
    plt.ylabel('Density', fontsize=35)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.xticks(fontsize = 30) 
    plt.yticks(fontsize = 30) 
    plt.legend(prop={'size':18})
    # plt.show()
    
    #position
    plt.figure(300)

    # plt.plot(test, color='xkcd:dusty pink')
    sb.histplot(clean_normalised_distribution, element='step', fill = True, stat='density',color='xkcd:mauve', label=f'K={np.round(K_val,3)}')
    # sb.histplot(normalised_distribution1,element='step', fill=False, color='xkcd:greenish blue', stat='density', label='K=0.6')
    # sb.histplot(normalised_distribution2, element='step', fill=True, color='xkcd:teal', stat='density', label='K=2')
    # plt.plot(array, density_lte, color='xkcd:grass green', label='LTE')
    # plt.plot(array,density_iso, color='xkcd:crimson', label='Isothermal')
    # plt.suptitle(r'velocity has square$')
    # plt.suptitle(r'Radial distribution of the height of a DM particle after $10^4$ collisions in planet with a linear temperature gradient', fontsize=30, wrap=True)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.xticks(fontsize = 30) 
    plt.yticks(fontsize = 30) 
    plt.legend(prop={'size':18})
    plt.xlabel(r'$r/r_\chi$', fontsize=35)
    # plt.ylabel(r'$4 \pi  r^2 n_\chi(r)$', fontsize=45)
    plt.ylabel(r'Density', fontsize=35)
    plt.show()
    
    
    # plt.figure(300)
    
    # array_lum =np.linspace(0,2.5,100, endpoint=True)
    
    # plt.plot(array_lum, [item[0] for item in luminosity_r_bins], label='K = 0.2')
    # plt.suptitle(r'Radial distribution of the Luminosity by a DM particle', fontsize=30, wrap=True)
    # plt.xlabel(r'R (m)')
    # plt.ylabel(r'$\langle dL \rangle$')
    # plt.legend()
    
    
    # sb.lineplot(x=array,y=density_iso, color='xkcd:grey green', label='Isothermal')

    # sb.histplot(normed_position, color='xkcd:powder blue', stat='density', kde=True)
    # plt.xlim(0, 2.5)
    # plt.ylim(0, 1)
    # # plt.plot(time_dependent_velocity[int(len(simulation_time)/2):int(3*len(simulation_time)/4)], color='xkcd:dusty rose')

    # plt.xlabel(r't (s)')
    # plt.ylabel(r'v ($m.s^{-1}$)')
    # plt.title(r'Evolution of the velocity of a DM particle over 250 collisions')
    # plt.xlim(left=0, right=2.5)
    # plt.show()
    
    
    # print(f'\n time-dependent position list is {time_dependent_position}')
df = pd.DataFrame(data)
df.to_csv('BetterYoungJupiterValues2_020424.csv', index=False)

##Calculation of Monte Carlo variables##

def get_graphs(energy, position):
    sb.kdeplot(x=position, kind='kde', bw_adjust=2.85,)
    sb.histplot(x=position, stat='density', kde=True)
    plt.xlim(0, 3.5)
    plt.ylim(0)

    plt.xlabel(r'$r_\chi$ (m)')
    plt.ylabel(r'density distribution $n_\chi(r)$')
    plt.title(r'Radial distribution of the height of a DM particle over $10^4$ collisions')
    plt.show()
    return


       







