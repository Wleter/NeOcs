from dataclasses import dataclass
from matplotlib import pyplot as plt
plt.rcParams['font.size'] = 16

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
from .my_types import FloatNDArray
from .potential import PotentialArray
from .units import PS
from scipy.special import roots_legendre

def plot() -> tuple[Figure, Axes] :
    fig, ax = plt.subplots()
    ax.grid()
    ax.tick_params(which='both', direction="in")

    return fig, ax

def into_polar(potential: PotentialArray, value_max: float) -> PotentialArray:
    n_polar = potential.polar.shape[0]

    values_mod = np.zeros((potential.radial.shape[0] + 1, 2 * n_polar))
    values_mod[1:, 0:n_polar] = potential.values
    values_mod[1:, n_polar::] = potential.values[:, ::-1]
    values_mod[0, :] = value_max

    radial_mod = np.zeros(potential.radial.shape[0] + 1)
    radial_mod[1:] = potential.radial

    polar_mod = np.zeros(2 * n_polar)
    polar_mod[0:n_polar] = potential.polar
    polar_mod[n_polar::] = 2 * np.pi - potential.polar[::-1]

    return PotentialArray(radial_mod, polar_mod, values_mod)

def wave_into_polar(radial: FloatNDArray, polar: FloatNDArray, values: FloatNDArray) -> tuple[FloatNDArray, FloatNDArray, FloatNDArray]:
    radial = np.concatenate(([0], radial))
    polar = np.concatenate(([0], polar))
    values = np.concatenate((values[:, 0:1], values), axis=1)
    values = np.concatenate((np.zeros_like(values[0:1, :]), values), axis=0)

    polar = np.concatenate((polar, np.flip(-polar)))
    values = np.concatenate((values, np.flip(values, axis=1)), axis=1)

    return radial, polar, values

def loss_plot(path: str, prefix: str): 
    xpi = np.loadtxt(f'{path}/{prefix}_xpi.dat', skiprows=1, delimiter="\t")
    bsigma = np.loadtxt(f'{path}/{prefix}_bsigma.dat', skiprows=1, delimiter="\t")

    fig, ax = plt.subplots()
    ax.grid()
    ax.tick_params(which='both', direction="in")
    
    ax.plot(xpi[:, 0] / PS, 100 * xpi[:, 1], label="PI")
    ax.plot(bsigma[:, 0] / PS, 100 * bsigma[:, 1], label="DI")
    ax.set_xlabel('Time [ps]')
    ax.set_ylabel('Loss [%]')
    ax.legend()

    return fig, ax

def with_distance(path, file_prefix: str, fig_ax: tuple[Figure, Axes]) -> Axes:
    wave = np.load(f'{path}/{file_prefix}_distance_animation.npy')
    r = np.load(f'{path}/{file_prefix}_distance_animation_r_grid.npy')
    time = np.load(f'{path}/{file_prefix}_distance_animation_time.npy') / PS
    
    distance = r @ wave

    ax2: Axes = fig_ax[1].twinx() # type: ignore
    ax2.plot(time, distance, alpha = 0.4)
    ax2.set_ylabel("Distance [bohr]")

    return ax2

@dataclass
class LastState:
    path: str
    file_prefix: str

    def wave_2d(self) -> tuple[Figure, Axes]:
        wave = np.load(f'{self.path}/{self.file_prefix}_wave_animation.npy')
        r = np.load(f'{self.path}/{self.file_prefix}_wave_animation_x_grid.npy')
        polar = np.load(f'{self.path}/{self.file_prefix}_wave_animation_y_grid.npy')

        r, polar, wave = wave_into_polar(r, polar, wave)

        fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))
        ax.contourf(polar, r, wave[:, :, -1], cmap='hot', levels=50)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(False)

        return fig, ax
    
    def angular(self) -> tuple[Figure, Axes]:
        wave = np.load(f'{self.path}/{self.file_prefix}_angular_animation.npy')
        l = np.load(f'{self.path}/{self.file_prefix}_angular_animation_angular_momentum_grid.npy')

        fig, ax = plot()

        ax.plot(l, wave[:, -1])

        ax.set_xlabel('Angular momentum j')
        ax.set_ylabel('Wave function density')

        return fig, ax

    def polar(self) -> tuple[Figure, Axes]: 
        wave = np.load(f'{self.path}/{self.file_prefix}_polar_animation.npy')
        l = np.load(f'{self.path}/{self.file_prefix}_polar_animation_theta_grid.npy')

        fig, ax = plot()

        ax.plot(l, wave[:, -1])

        ax.set_xlabel('Angle [rad]')
        ax.set_ylabel('Wave function density')

        return fig, ax

    def omega(self) -> tuple[Figure, Axes]:
        wave = np.load(f'{self.path}/{self.file_prefix}_omega_animation.npy')
        omega = np.load(f'{self.path}/{self.file_prefix}_omega_animation_omega_grid.npy')

        fig, ax = plot()

        ax.plot(omega, wave[:, -1])

        ax.set_xlabel(r'Angular momentum projection $\Omega$')
        ax.set_ylabel('Wave function density')
        ax.set_ylim(0, 1)

        return fig, ax

    def distance(self) -> tuple[Figure, Axes]: 
        wave = np.load(f'{self.path}/{self.file_prefix}_distance_animation.npy')
        distance = np.load(f'{self.path}/{self.file_prefix}_distance_animation_r_grid.npy')

        fig, ax = plot()

        ax.plot(distance, wave[:, -1])

        ax.set_xlabel('Distance r [bohr]')
        ax.set_ylabel('Wave function density')

        return fig, ax

@dataclass
class AlignmentPlot:
    path: str

    def single(self, file_prefix: str) -> tuple[FloatNDArray, FloatNDArray]:
        wave = np.load(f'{self.path}/{file_prefix}_polar_animation.npy')
        l = np.load(f'{self.path}/{file_prefix}_polar_animation_theta_grid.npy')
        time = np.load(f'{self.path}/{file_prefix}_polar_animation_time.npy') / PS

        points, weights = roots_legendre(l.shape[0])
        weights = np.flip(weights)
        points = np.flip(points)
        
        align = (points ** 2) @ wave

        return time, align
    
    def plot_single(self, file_prefix: str) -> tuple[Figure, Axes]:
        time, align = self.single(file_prefix)

        fig, ax = plot()
        ax.plot(time, align)

        ax.set_xlabel('time [ps]')
        ax.set_ylabel(r'$\left< \cos^2(\theta) \right>$')

        return fig, ax
    
    def single_j(self, file_prefix: str, j: int) -> tuple[FloatNDArray, FloatNDArray]:
        time, alignment = self.single(f"{file_prefix}_{j}_0")
        alignment /= (2 * j + 1)

        for omega in range(1, j+1):
            _, align = self.single(f"{file_prefix}_{j}_{omega}")
            alignment += 2 / (2 * j + 1) * align

        return time, alignment
    
    def plot_series(self, *args: tuple[FloatNDArray, FloatNDArray]):
        fig, ax = plot()
        for time, align in args: 
            ax.plot(time, align)

        ax.set_xlabel('time [ps]')
        ax.set_ylabel(r'$\left< \cos^2(\theta) \right>$')

        return fig, ax
    
    def with_distance(self, file_prefix: str, fig_ax: tuple[Figure, Axes]) -> Axes:
        wave = np.load(f'{self.path}/{file_prefix}_distance_animation.npy')
        r = np.load(f'{self.path}/{file_prefix}_distance_animation_r_grid.npy')
        time = np.load(f'{self.path}/{file_prefix}_distance_animation_time.npy') / PS
        
        distance = r @ wave

        ax2: Axes = fig_ax[1].twinx() # type: ignore
        ax2.plot(time, distance, alpha = 0.4)
        ax2.set_ylabel("Distance [bohr]")

        return ax2
    