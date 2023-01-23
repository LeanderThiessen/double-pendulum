import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


#set up system of 1st order diff. equations of form y'(x)=f(x)
def f(t, dt,r,m_1,m_2, l_1, l_2,g,phi_1,phi_2,w_1,w_2):
    u=m_2/(m_1+m_2)
    w_1_dot=1/(1-u*np.cos(phi_1-phi_2)*np.cos(phi_1-phi_2))*(-u*np.cos(phi_1-phi_2)*np.sin(phi_1-phi_2)*w_1*w_1  +  u*g/l_1*np.cos(phi_1-phi_2)*np.sin(phi_2) - u*l_2/l_1*w_2*w_2*np.sin(phi_1-phi_2) - g/l_1*np.sin(phi_1))-r*w_1
    w_2_dot=1/(1-u*np.cos(phi_1-phi_2)*np.cos(phi_1-phi_2))*( u*np.cos(phi_1-phi_2)*np.sin(phi_1-phi_2)*w_2*w_2  +  g/l_2*np.cos(phi_1-phi_2)*np.sin(phi_1)   + l_1/l_2*w_1*w_1*np.sin(phi_1-phi_2)   - g/l_2*np.sin(phi_2))-r*w_2

    return [w_1,w_2,w_1_dot,w_2_dot]


#deletes every nth element of "array"
def delete_n(array,n):
    array = np.delete(array, np.arange(0, array.size, n))
    return array

#initialize animation
def init():
    result = []
    for l in range(L):
        point_1[l].set_data([], [])
        point_2[l].set_data([], [])
        point_3[l].set_data([], [])
        result.append(point_1[l])
        result.append(point_2[l])
        result.append(point_3[l])
    return result

#this function is applied for every frame of the animation
def animate(i,n):
    for l in range(L):
        t=np.linspace(0,1,2)
        x = t*l_1*np.sin(position_1[l][i])             #regular pendulum
        y = -t*l_1*np.cos(position_1[l][i])            #regular pendulum
        #x = np.sin(velocity_1[l][i])                            #phase space
        #y = np.sin(velocity_2[l][i])                            #phase space

        point_1[l].set_data(x, y)         #coordinates of first pendulum

        x = l_1*np.sin(position_1[l][i])+t*l_2*np.sin(position_2[l][i])     #regular pendulum
        y = -l_1*np.cos(position_1[l][i])-t*l_2*np.cos(position_2[l][i])    #regular pendulum
        #x = np.sin(velocity_1[l][:i])                                                #phase space
        #y = np.sin(velocity_2[l][:i])                                                #phase space

        point_2[l].set_data(x, y)         #coordinates of second pendulum

        x = l_1 * np.sin(position_1[l][:i]) +  l_2 * np.sin(position_2[l][:i])     #regular pendulum
        y = -l_1 * np.cos(position_1[l][:i]) -  l_2 * np.cos(position_2[l][:i])    #regular pendulum
        point_3[l].set_data(x, y)  #trace of second pendulum
    result = []
    for l in range(L):
        result.append(point_1[l])
        result.append(point_2[l])
        result.append(point_3[l])
    return result


#Runge Kutta algorithm for one time step
#function time_step takes in variables at time t and returns variables at time t+dt
def time_step(l, j, t, dt, r ,m_1 ,m_2 ,l_1, l_2, g, phi_1, phi_2, w_1, w_2):
    k_1 = f(t, dt, r,m_1 ,m_2 ,l_1, l_2, g , phi_1, phi_2, w_1, w_2)
    k_2 = f(t + dt / 2, dt, r ,m_1 ,m_2 ,l_1, l_2, g, phi_1 + dt / 2 * k_1[0], phi_2 + dt / 2 * k_1[1], w_1 + dt / 2 * k_1[2], w_2 + dt / 2 * k_1[3])
    k_3 = f(t + dt / 2, dt, r ,m_1 ,m_2 ,l_1, l_2, g, phi_1 + dt / 2 * k_2[0], phi_2 + dt / 2 * k_2[1], w_1 + dt / 2 * k_2[2], w_2 + dt / 2 * k_2[3])
    k_4 = f(t + dt, dt, r,m_1 ,m_2 ,l_1, l_2, g, phi_1 + dt * k_3[0], phi_2 + dt * k_3[1], w_1 + dt * k_3[2], w_2 + dt * k_3[3])

    phi_1 = phi_1 + dt * (1 / 6 * k_1[0] + 1 / 3 * k_2[0] + 1 / 3 * k_3[0] + 1 / 6 * k_4[0])
    phi_2 = phi_2 + dt * (1 / 6 * k_1[1] + 1 / 3 * k_2[1] + 1 / 3 * k_3[1] + 1 / 6 * k_4[1])
    w_1 = w_1 + dt * (1 / 6 * k_1[2] + 1 / 3 * k_2[2] + 1 / 3 * k_3[2] + 1 / 6 * k_4[2])
    w_2 = w_2 + dt * (1 / 6 * k_1[3] + 1 / 3 * k_2[3] + 1 / 3 * k_3[3] + 1 / 6 * k_4[3])

    position_1[l][j] = phi_1
    position_2[l][j] = phi_2
    velocity_1[l][j] = w_1
    velocity_2[l][j] = w_2
    return [phi_1,phi_2,w_1,w_2]


#Paramters
dt=0.001                    # time step
l_1=1                       # length of pendulum 1
l_2=1                       # length of pendulum 2
m_1=1                       # mass of pendulum 1
m_2=1                       # mass of pendulum 2
g=9.81                      # g=10
r=0                         # friction
#Initial conditions
t_in = 0                    # initial time
t_fin = 30                  # final time
w_1_in = 0                  # angular velocity of 1st
w_2_in = 0                  # angular velocity of 2nd

N = int((t_fin - t_in) / dt) + 1  # number of calculated steps

phi_1_in=np.pi              # angle of first pendulum
phi_2_in_string=[np.pi-0.000001]
#,np.pi-0.00000155,np.pi-0.0000016,np.pi-0.0000011,np.pi-0.00000125,np.pi-0.0000017,np.pi-0.00000145,np.pi-0.0000015
L=len(phi_2_in_string)

#store positions and velocities of both points in dictionary where each entry is a separate pendulum
position_1 = {}
velocity_1 = {}
position_2 = {}
velocity_2 = {}
for index in range(L):
    position_1[index]=[]
    velocity_1[index]=[]
    position_2[index]=[]
    velocity_2[index]=[]


l=0
count=0
for phi_2_in in phi_2_in_string:

    position_1[l]=np.zeros(N)
    position_2[l]=np.zeros(N)
    velocity_1[l]=np.zeros(N)
    velocity_2[l]=np.zeros(N)

    #set initial conditions
    phi_1=phi_1_in
    phi_2=phi_2_in
    w_1=w_1_in
    w_2=w_2_in
    t=t_in
    j=0

    while t<t_fin:
        updated_values=time_step(l,j, t, dt, r ,m_1 ,m_2 ,l_1, l_2 ,g, phi_1, phi_2, w_1, w_2)
        phi_1=updated_values[0]
        phi_2=updated_values[1]
        w_1=updated_values[2]
        w_2=updated_values[3]

        j += 1
        t += dt
        count+=1
        print("{:.2f}%".format((count/N)/L*100), end="\r", flush=True)

    #delete elements and only animate a subset (set speed of animation)
    position_1[l] = delete_n(position_1[l],2)
    position_2[l] = delete_n(position_2[l],2)
    position_1[l] = delete_n(position_1[l],2)
    position_2[l] = delete_n(position_2[l],2)



    l+=1

#execute animation
fig = plt.figure()
ax = plt.axes(xlim=(-2, 2), ylim=(-2, 2))

point_1 = {}
point_2 = {}
point_3 = {}
for l in range(L):
    point_1[l], = ax.plot([], [], '-o',color="red")
    point_2[l], = ax.plot([], [], '-o',color="red")
    point_3[l], = ax.plot([], [], '-', lw=1,alpha=0.2)
#color="{}".format((l+1)/(2*L))

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=N, fargs=[[i for i in range(N)]], interval=dt, blit=True)
#anim.save('pendulum.mp4', fps=30,bitrate=-1, extra_args=['-vcodec', 'libx264'])
plt.gca().set_aspect('equal', adjustable='box') #make sure plot is a square and stays that way
plt.show()
