import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from time import time

def f(t,u_1,u_2):                   #set up system of diff. equations of form y'(x)=f(x)
    return [u_2,-g/l*np.sin(u_1)-k*u_2]

def init():                         #prepare animation
    line.set_data([], [])
    phase.set_data([], [])
    return line,phase

def animate(i):
    t=np.linspace(0,1,2)
    x = t*np.sin(position[i])
    y = -t*np.cos(position[i])
    line.set_data(x, y)         #line is the data for pendulum motion

    y = [2*(y-p_avg)/(p_max-p_min) for y in position[:i]]
    x = [2*(x-v_avg)/(v_max-v_min) for x in velocity[:i]]
    phase.set_data(x,y)         #phase is the data for phase diagram in background

    return line,phase



dt=0.005                    #time step
l=0.5                       #length of pendulum
k=0.1                      #friction (0 for no friction)
g=9.81
t_in=0                      #initial time
t_fin=200                   #final time
u_1_in=np.pi-0.001        #initial position angle
u_2_in=0                    #initial angular velocity


position=[]     #position of endpoint for each time step given as angle to negative y-axis
velocity=[]     #time derivative of position

u_1=u_1_in      #initial conditions
u_2=u_2_in
t=t_in

while t<t_fin:          #Runge Kutta algorithm
    k_1 = f(t, u_1, u_2)
    k_2 = f(t + dt/2, u_1 + dt/2*k_1[0], u_2 + dt/2*k_1[1])
    k_3 = f(t + dt/2, u_1 + dt/2*k_2[0], u_2 + dt/2*k_2[1])
    k_4 = f(t + dt, u_1 + dt*k_3[0], u_2 + dt*k_3[1])

    u_1 = u_1 + dt*(1/6*k_1[0] + 1/3*k_2[0] + 1/3*k_3[0] + 1/6*k_4[0])
    u_2 = u_2 + dt*(1/6*k_1[1] + 1/3*k_2[1] + 1/3*k_3[1] + 1/6*k_4[1])

    position.append(u_1)
    velocity.append(u_2)
    t+=dt

p_min=np.max(position)  #values help for scaling the graph properly
p_max=np.min(position)
p_avg=(p_max+p_min)/2
v_min=np.max(velocity)
v_max=np.min(velocity)
v_avg=(v_max+v_min)/2

fig = plt.figure()      #execute animation
ax = plt.axes(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))
line, = ax.plot([], [], 'o-', lw=2)
phase,= ax.plot([], [], '-', color="red",lw=0.5,alpha=0.5)
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(position), interval=dt*5, blit=True)


plt.gca().set_aspect('equal', adjustable='box') #make sure plot is a square and stays that way
plt.show()

