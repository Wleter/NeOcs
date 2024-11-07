from dataclasses import dataclass
from matplotlib import animation
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 16
from scipy.special import roots_legendre

from .units import PS
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

        r, polar, wave = wave_into_polar(r, polar, wave)

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
