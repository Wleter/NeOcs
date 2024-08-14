from matplotlib import animation
import numpy as np
import matplotlib.pyplot as plt

def wave_animation(prefix):
    path = "../data"
    
    wave = np.load(f'{path}/{prefix}_wave_animation.npy')
    r = np.load(f'{path}/{prefix}_wave_animation_x_grid.npy')
    theta = np.load(f'{path}/{prefix}_wave_animation_y_grid.npy')

    theta = np.concatenate(([0], theta))
    wave = np.concatenate((wave[:, 0:1], wave), axis=1)

    theta = np.concatenate((theta, np.flip(-theta)))
    wave = np.concatenate((wave, np.flip(wave, axis=1)), axis=1)

    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))

    def animate(i):
        ax.clear()
        ax.contourf(theta, r, wave[:, :, i], cmap='hot', levels=50)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(False)
    anim = animation.FuncAnimation(fig, animate, interval=60, frames=wave.shape[2], blit=False)
    return anim

def angular_animation(prefix): 
    path = "../data/"

    wave = np.load(f'{path}/{prefix}_angular_animation.npy')
    l = np.load(f'{path}/{prefix}_angular_animation_angular_momentum_grid.npy')

    fig, ax = plt.subplots()

    def animate(i):
        ax.clear()
        ax.plot(l, wave[:, i])
        ax.set_xlabel('Angular momentum j')
        ax.set_ylabel('Wave function density')

    anim = animation.FuncAnimation(fig, animate, interval=60, frames=wave.shape[1], blit=False)
    return anim