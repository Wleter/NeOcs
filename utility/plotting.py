from dataclasses import dataclass
from matplotlib import pyplot as plt
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

def wave_into_polar(polar: FloatNDArray, values: FloatNDArray) -> tuple[FloatNDArray, FloatNDArray]:
    polar = np.concatenate(([0], polar))
    values = np.concatenate((values[:, 0:1], values), axis=1)

    polar = np.concatenate((polar, np.flip(-polar)))
    values = np.concatenate((values, np.flip(values, axis=1)), axis=1)

    return polar, values

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
        wave = np.load(f'{self.path}/{file_prefix}_0_0_distance_animation.npy')
        r = np.load(f'{self.path}/{file_prefix}_0_0_distance_animation_r_grid.npy')
        time = np.load(f'{self.path}/{file_prefix}_0_0_distance_animation_time.npy') / PS
        
        distance = r @ wave

        ax2: Axes = fig_ax[1].twinx() # type: ignore
        ax2.plot(time, distance)
        ax2.set_ylabel("Distance [bohr]")

        return ax2
    