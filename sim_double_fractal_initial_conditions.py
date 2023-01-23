import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import animation


#converts coordinates x,y into pendulum position phi_1,phi_2, such that the second pendulum is at x,y
def conv_angles(x,y):
    d=np.sqrt(x*x+y*y)

    if x==0 and y==0:
        return [0,np.pi]
    theta=np.arccos(y / d)
    alpha=np.arccos(d / 2)


    if x > 0 and y <= 0:
        return [theta + alpha, theta - alpha]

    if x > 0 and y > 0:
        return [theta  + alpha, theta  - alpha]

    if x <= 0 and y > 0:
        return [(theta  + alpha), (theta  - alpha)]

    if x <= 0 and y <= 0:
        return [(theta + alpha), (theta - alpha)]

#specify system of 1st order D.eq f(y)=y'
def f(t,phi_1,phi_2,w_1,w_2):                   #set up system of diff. equations of form y'(x)=f(x)
    u=m_2/(m_1+m_2)

    w_1_dot=1/(1-u*np.cos(phi_1-phi_2)*np.cos(phi_1-phi_2))*(-u*np.cos(phi_1-phi_2)*np.sin(phi_1-phi_2)*w_1*w_1  +  u*g/l_1*np.cos(phi_1-phi_2)*np.sin(phi_2) - u*l_2/l_1*w_2*w_2*np.sin(phi_1-phi_2) - g/l_1*np.sin(phi_1))
    w_2_dot=1/(1-u*np.cos(phi_1-phi_2)*np.cos(phi_1-phi_2))*( u*np.cos(phi_1-phi_2)*np.sin(phi_1-phi_2)*w_2*w_2  +  g/l_2*np.cos(phi_1-phi_2)*np.sin(phi_1)   + l_1/l_2*w_1*w_1*np.sin(phi_1-phi_2)   - g/l_2*np.sin(phi_2))

    return [w_1,w_2,w_1_dot,w_2_dot]

#deletes every second element of "array"
def delete_half(array):
    array = np.delete(array, np.arange(0, array.size, 2))
    return array

#initialize animation plot
def init():                        #prepare animation
    point_1.set_data([], [])
    point_2.set_data([], [])


    return point_1,point_2

#animation function is executed for every frame (i) in the animation
def animate(i):                     #this function is applied for every frame of the animation
    t=np.linspace(0,1,2)
    x = t*l_1*np.sin(position_1[i])
    y = -t*l_1*np.cos(position_1[i])
    point_1.set_data(x, y)         #line is the data for pendulum motion

    x = l_1*np.sin(position_1[i])+t*l_2*np.sin(position_2[i])
    y = -l_1*np.cos(position_1[i])-t*l_2*np.cos(position_2[i])
    point_2.set_data(x, y)         #line is the data for pendulum motion

    #x = l_1 * np.sin(position_1[:i]) +  l_2 * np.sin(position_2[:i])
    #y = -l_1 * np.cos(position_1[:i]) -  l_2 * np.cos(position_2[:i])
    #point_3.set_data(x, y)  # line is the data for pendulum motion

    return point_1,point_2

#Runge Kutta algorithm
def time_step(j, t, phi_1, phi_2, w_1, w_2):
    k_1 = f(t, phi_1, phi_2, w_1, w_2)
    k_2 = f(t + dt / 2, phi_1 + dt / 2 * k_1[0], phi_2 + dt / 2 * k_1[1], w_1 + dt / 2 * k_1[2], w_2 + dt / 2 * k_1[3])
    k_3 = f(t + dt / 2, phi_1 + dt / 2 * k_2[0], phi_2 + dt / 2 * k_2[1], w_1 + dt / 2 * k_2[2], w_2 + dt / 2 * k_2[3])
    k_4 = f(t + dt, phi_1 + dt * k_3[0], phi_2 + dt * k_3[1], w_1 + dt * k_3[2], w_2 + dt * k_3[3])

    phi_1 = phi_1 + dt * (1 / 6 * k_1[0] + 1 / 3 * k_2[0] + 1 / 3 * k_3[0] + 1 / 6 * k_4[0])
    phi_2 = phi_2 + dt * (1 / 6 * k_1[1] + 1 / 3 * k_2[1] + 1 / 3 * k_3[1] + 1 / 6 * k_4[1])
    w_1 = w_1 + dt * (1 / 6 * k_1[2] + 1 / 3 * k_2[2] + 1 / 3 * k_3[2] + 1 / 6 * k_4[2])
    w_2 = w_2 + dt * (1 / 6 * k_1[3] + 1 / 3 * k_2[3] + 1 / 3 * k_3[3] + 1 / 6 * k_4[3])

    position_1[j] = phi_1
    position_2[j] = phi_2
    velocity_1[j] = w_1
    velocity_2[j] = w_2
    return [phi_1,phi_2,w_1,w_2]

#Paramters
dt=0.001                    #time step
l_1=1                       #length of pendulum 1
l_2=1                       #length of pendulum 2
m_1=1                       #mass of pendulum 1
m_2=1                       #mass of pendulum 2
g=9.81
delta=0.01

t_in = 0  # initial time
t_fin = 10  # final time

w_1_in = 0  # angular velocity of 1st
w_2_in = 0  # angular velocity of 2nd

N = int((t_fin - t_in) / dt) + 1  # number of calculated steps

#L=dimension of grid; each grid point is the initial position of a simulation run (number of grid points = (2*L)^2)
L=4
result=np.zeros((2*L+1,2*L+1))

start_time = time.time()
count=0
for n in range(-L,L+1):
    for m in range(-L,L+1):
        print("{:.2f}%".format(count/((2*L+1)*(2*L+1))*100), end="\r", flush=True)
        count+=1
        #print("{},{}".format(n,m))
        x_pos=n*(l_1+l_2)/L
        y_pos=m*(l_1+l_2)/L
        #print(x_pos,y_pos)
        if np.sqrt(x_pos*x_pos+y_pos*y_pos)>(l_1+l_2):
            result[(m+L),n+L]=-2
        else:
            phi_1_in,phi_2_in = conv_angles(x_pos,y_pos)



            position_1=np.zeros(N)
            position_2=np.zeros(N)
            velocity_1=np.zeros(N)
            velocity_2=np.zeros(N)

            #initial conditions
            phi_1=phi_1_in
            phi_2=phi_2_in
            w_1=w_1_in
            w_2=w_2_in
            t=t_in
            j=0

            while t<t_fin:
                updated_values=time_step(j, t, phi_1, phi_2, w_1, w_2)
                phi_1=updated_values[0]
                phi_2=updated_values[1]
                w_1=updated_values[2]
                w_2=updated_values[3]

                x = l_1 * np.sin(position_1[j]) + l_2 * np.sin(position_2[j])
                y = -l_1 * np.cos(position_1[j]) - l_2 * np.cos(position_2[j])

                if abs(x)<delta and abs(y)<delta:
                    #print(x,y,t)
                    #result[n+L,m+L]=t
                    break
                j += 1
                t += dt
            result[(m+L),n+L] = t
            #if x_pos==0 or y_pos==0:
            #    result[m + L, n + L]=0

end_time = time.time()

print("L={} Runtime= {}".format(L,end_time-start_time))

plt.imshow(result)
plt.colorbar()
plt.savefig("Plots/N={}.png".format(L),dpi=400)
plt.show()

#position_1 = delete_half(position_1)
#position_2 = delete_half(position_2)
#position_1 = delete_half(position_1)
#position_2 = delete_half(position_2)
#position_1 = delete_half(position_1)
#position_2 = delete_half(position_2)
#position_1 = delete_half(position_1)
#position_2 = delete_half(position_2)
#position_1 = delete_half(position_1)
#position_2 = delete_half(position_2)

#execute animation
#fig = plt.figure()
#ax = plt.axes(xlim=(-2, 2), ylim=(-2, 2))
#point_1, = ax.plot([], [], 'o-', lw=2, color="blue")
#point_2,= ax.plot([], [], 'o-', lw=2,color="red")
#point_3,= ax.plot([], [], '-', lw=1,alpha=0.4,color="red")

#anim = animation.FuncAnimation(fig, animate, init_func=init, frames=position_1.size, interval=dt, blit=True)
#plt.plot(l_1*np.sin(position_1),-l_1*np.cos(position_1))
#plt.plot(l_1*np.sin(position_1)+l_2*np.sin(position_2),-l_1*np.cos(position_1)-l_2*np.cos(position_2))


#plt.gca().set_aspect('equal', adjustable='box') #make sure plot is a square and stays that way
#plt.show()

