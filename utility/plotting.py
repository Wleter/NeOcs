from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
from .my_types import FloatNDArray
from .potential import PotentialArray

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