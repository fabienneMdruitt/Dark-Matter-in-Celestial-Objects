import numpy as np
import math

# constants
m_chi = 1
n=1
k_B = 1.380649 * pow(10,-23)# J/K 8.617333262 ev/K
m_N = 1

#functions

def magnitude(vector):
    mag = math.sqrt(sum(np.power(element, 2) for element in vector))
    return mag

## s is still calculated with the lab velocity vector which was the incoming velocity. This is because we assume that the collision was instantaneous
## and therefore s is a constant during the process. 
def transform_com_to_lab(outgoing_velocity_vector, lab_velocity_vector, u_vector):
    s = (m_chi*lab_velocity_vector+m_N*u_vector)/(m_chi+m_N)
    outgoing_lab = outgoing_velocity_vector + s
    return outgoing_lab

def transform_lab_to_com(velocity_vector, u_vector):
    
    s = (m_chi*velocity_vector+m_N*u_vector)/(m_chi+m_N)
    
    t = velocity_vector - s
    return t
def theta_phi_selection(interaction):
    
    if interaction == 'velocity-dependent':
        theta = np.random.uniform()*np.pi
    else:
        value_theta = np.linspace(0, 200, 100, True)
        theta_distribution = (1 -np.cos(value_theta))**n
        
        theta = theta_distribution[int(np.random.random()*(len(theta_distribution)-1))]
    
    phi = np.random.uniform(0, 2*np.pi)

    return theta, phi

def outgoing_velocity(incoming_velocity):
    theta, phi = theta_phi_selection('velocity-dependent')
    #calculating magnitude of velocity vector
    incoming_speed = np.sqrt(pow(incoming_velocity[0],2)+pow(incoming_velocity[1], 2)+pow(incoming_velocity[2],2))
    outgoing = np.array([incoming_speed*np.sin(theta)*np.cos(phi), incoming_speed*np.sin(theta)*np.sin(phi), incoming_speed*np.cos(theta)])

    
    return outgoing

def collision(dm_velocity, n_velocity):
    
    #transform incoming dm velocity to CoM frame
    t = transform_lab_to_com(dm_velocity, n_velocity)
    # print(magnitude(dm_velocity))
    t_outgoing = outgoing_velocity(t)
    
    #transform back to lab frame
    dm_outgoing = transform_com_to_lab(t_outgoing,dm_velocity,n_velocity)
    # print(magnitude(dm_outgoing))
    return dm_outgoing

# test = collision(np.array([-34.49601653566134, 28.279707505558786, 22.52555812735106]), np.array([37.845575543195362, 57.559275485959475, 24.15550264051815]))