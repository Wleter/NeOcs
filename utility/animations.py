from dataclasses import dataclass
from matplotlib import animation
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import roots_legendre
from .plotting import wave_into_polar

@dataclass
class Animator:
    path: str
    file_prefix: str

    save_path: str
    save_prefix: str

    def wave_2d_animation(self, fps: int = 30):
        wave = np.load(f'{self.path}/{self.file_prefix}_wave_animation.npy')
        r = np.load(f'{self.path}/{self.file_prefix}_wave_animation_x_grid.npy')
        polar = np.load(f'{self.path}/{self.file_prefix}_wave_animation_y_grid.npy')

        polar, wave = wave_into_polar(polar, wave)

        fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
        def animate(i: int):
            ax.clear()
            ax.contourf(polar, r, wave[:, :, i], cmap='hot', levels=50)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.grid(False)
            
            return []

        anim = animation.FuncAnimation(fig, animate, interval=60, frames=wave.shape[2], blit=False)

        filename = f"{self.save_path}/{self.save_prefix}_wave_animation.gif"
        anim.save(filename, writer="pillow", fps=fps)
        print("saved animation as", filename)
    
    def angular_animation(self, fps: int = 30):
        wave = np.load(f'{self.path}/{self.file_prefix}_angular_animation.npy')
        l = np.load(f'{self.path}/{self.file_prefix}_angular_animation_angular_momentum_grid.npy')

        fig, ax = plt.subplots()

        def animate(i):
            ax.clear()
            ax.plot(l, wave[:, i])

            ax.set_xlabel('Angular momentum j')
            ax.set_ylabel('Wave function density')

            return []

        anim = animation.FuncAnimation(fig, animate, interval=60, frames=wave.shape[1], blit=False)
        filename = f"{self.save_path}/{self.save_prefix}_angular_animation.gif"
        anim.save(filename, writer="pillow", fps=fps)
        print("saved animation as", filename)

    def polar_animation(self, fps: int = 30): 
        wave = np.load(f'{self.path}/{self.file_prefix}_polar_animation.npy')
        l = np.load(f'{self.path}/{self.file_prefix}_polar_animation_theta_grid.npy')

        fig, ax = plt.subplots()

        def animate(i):
            ax.clear()
            ax.plot(l, wave[:, i])

            ax.set_xlabel('Angle [rad]')
            ax.set_ylabel('Wave function density')

            return []

        anim = animation.FuncAnimation(fig, animate, interval=60, frames=wave.shape[1], blit=False)
        filename = f"{self.save_path}/{self.save_prefix}_polar_animation.gif"
        anim.save(filename, writer="pillow", fps=fps)
        print("saved animation as", filename)

    def omega_animation(self, fps: int = 30):
        wave = np.load(f'{self.path}/{self.file_prefix}_omega_animation.npy')
        omega = np.load(f'{self.path}/{self.file_prefix}_omega_animation_omega_grid.npy')

        fig, ax = plt.subplots()

        def animate(i):
            ax.clear()
            ax.plot(omega, wave[:, i])

            ax.set_xlabel(r'Angular momentum projection $\Omega$')
            ax.set_ylabel('Wave function density')
            ax.set_ylim(0, 1)

            return []

        anim = animation.FuncAnimation(fig, animate, interval=60, frames=wave.shape[1], blit=False)
        filename = f"{self.save_path}/{self.save_prefix}_omega_animation.gif"
        anim.save(filename, writer="pillow", fps=fps)
        print("saved animation as", filename)

    def distance_animation(self, fps: int = 30): 
        wave = np.load(f'{self.path}/{self.file_prefix}_distance_animation.npy')
        distance = np.load(f'{self.path}/{self.file_prefix}_distance_animation_r_grid.npy')

        fig, ax = plt.subplots()

        def animate(i):
            ax.clear()
            ax.plot(distance, wave[:, i])

            ax.set_xlabel('Distance r [bohr]')
            ax.set_ylabel('Wave function density')

            return []

        anim = animation.FuncAnimation(fig, animate, interval=60, frames=wave.shape[1], blit=False)
        filename = f"{self.save_path}/{self.save_prefix}_distance_animation.gif"
        anim.save(filename, writer="pillow", fps=fps)
        print("saved animation as", filename)

def alignement(prefix): 
    wave = np.load(f'{path}/{prefix}_polar_animation.npy')
    l = np.load(f'{path}/{prefix}_polar_animation_theta_grid.npy')
    time = np.load(f'{path}/{prefix}_polar_animation_time.npy') / ps

    points, weights = roots_legendre(l.shape[0])
    weights = np.flip(weights)
    points = np.flip(points)
    
    align = (points ** 2) @ wave
    
    fig, ax = plt.subplots()
    ax.grid()
    ax.tick_params(which='both', direction="in")
    
    ax.plot(time, align)
    ax.set_xlabel('time [ps]')
    ax.set_ylabel('$\\left<\\cos^2(\\theta)\\right>$')

    return fig, ax

def alignements(prefix, js): 
    fig, ax = plt.subplots()
    ax.grid()
    ax.tick_params(which='both', direction="in")
    ax.set_xlabel('time [ps]')
    ax.set_ylabel('$\\left<\\cos^2(\\theta)\\right>$')

    for j in js:
        time, alignment = alignment_from_wave(path, f"{prefix}_{j}_0")
        alignment /= (2 * j + 1)

        for omega in range(1, j+1):
            _, align = alignment_from_wave(path, f"{prefix}_{j}_{omega}")
            alignment += 2 / (2 * j + 1) * align
            
        ax.plot(time, alignment, label = f"$j = {j}$")

    return fig, ax

def alignments_with_distance(prefix, js):
    fig, ax = alignements(prefix ,js)

    ax2 = ax.twinx()
    ax2.set_ylabel("distance [bohr]")
    
    for j in js:
        time, distances = distance_from_wave(path, f"{prefix}_{j}_0")
        distances /= (2 * j + 1)

        for omega in range(1, j+1):
            _, distance = distance_from_wave(path, f"{prefix}_{j}_{omega}")
            distances += 2 / (2 * j + 1) * distance
            
        ax2.plot(time, distances, alpha=0.3)

    return fig, ax, ax2

def distance_from_wave(path, prefix):
    wave = np.load(f'{path}/{prefix}_distance_animation.npy')
    r = np.load(f'{path}/{prefix}_distance_animation_r_grid.npy')
    time = np.load(f'{path}/{prefix}_distance_animation_time.npy') / ps
    
    distances = r @ wave
    return time, distances

def show_potential(prefix: str):
    r = np.load(f'{path}/{prefix}_potential_r.npy')
    theta = np.load(f'{path}/{prefix}_potential_theta.npy')
    potential_array = np.load(f'{path}/{prefix}_potential.npy')
    
    fig, ax = plt.subplots()
    CS = ax.contourf(r, theta, potential_array, levels=50)
    cbar = fig.colorbar(CS)
    cbar.ax.set_ylabel("Energy [cm$^{-1}$]")
    ax.set_xlabel("R [bohr]")
    ax.set_ylabel("$\\theta$ [rad]")

    return fig, ax

def alignement_mixed(prefix, js): 
    fig, ax = plt.subplots()
    ax.grid()
    ax.tick_params(which='both', direction="in")
    ax.set_xlabel('time [ps]')
    ax.set_ylabel('$\\left<\\cos^2(\\theta)\\right>$')

    for j in js:
        time, alignment = alignment_from_wave(path, f"{prefix}_{j}")
        ax.plot(time, alignment, label = f"$j = {j}$")

    return fig, ax

def alignement_mixed_phases(prefix, phases: list[str]): 
    fig, ax = plt.subplots()
    ax.grid()
    ax.tick_params(which='both', direction="in")
    ax.set_xlabel('time [ps]')
    ax.set_ylabel('$\\left<\\cos^2(\\theta)\\right>$')

    time, alignment = alignment_from_wave(path, f"{prefix}_0_0")
    ax.plot(time, alignment, label = f"$j = 0$")

    for phase in phases:
        time, alignment = alignment_from_wave(path, f"{prefix}_1_phase_{phase}")

        phase = phase.split("_")
        phase[0] = phase[0].replace("hpi", "pi/2")
        phase[1] = phase[1].replace("hpi", "pi/2")
        phase[0] = phase[0].replace("pi", "\\pi")
        phase[1] = phase[1].replace("pi", "\\pi")
        ax.plot(time, alignment, label = f"$j = 1, \\phi_1 = {phase[0]}$, " + "$\\phi_{-1} = $" + f"${phase[1]}$")

    return fig, ax

def alignment_from_wave(path, prefix):
    wave = np.load(f'{path}/{prefix}_polar_animation.npy')
    l = np.load(f'{path}/{prefix}_polar_animation_theta_grid.npy')
    time = np.load(f'{path}/{prefix}_polar_animation_time.npy') / ps

    points, weights = roots_legendre(l.shape[0])
    weights = np.flip(weights)
    points = np.flip(points)
    
    align = (points ** 2) @ wave

    return time, align

def loss_plot(prefix): 
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
