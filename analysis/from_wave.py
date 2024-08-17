from matplotlib import animation
import numpy as np
import matplotlib.pyplot as plt

ps = 41341.374

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

def loss_plot(prefix): 
    path = "../data/"

    xpi = np.loadtxt(f'{path}/{prefix}_xpi.dat', skiprows=1, delimiter="\t")
    bsigma = np.loadtxt(f'{path}/{prefix}_bsigma.dat', skiprows=1, delimiter="\t")

    fig, ax = plt.subplots()
    ax.grid()
    ax.tick_params(which='both', direction="in")
    
    ax.plot(xpi[:, 0] / ps, 100 * xpi[:, 1], label="XPi")
    ax.plot(bsigma[:, 0] / ps, 100 * bsigma[:, 1], label="A + B")
    ax.set_xlabel('Time [ps]')
    ax.set_ylabel('Loss [%]')
    ax.legend()

    return fig, ax