from dataclasses import dataclass
from enum import Flag, auto
from typing import Protocol
import split_op as split
import numpy as np
from scipy.special import roots_legendre, lpmv
from numpy.typing import NDArray
from tqdm import tqdm
import os

data_path = "data/"

CM_INV = 4.55633e-6
KELVIN = 3.1668105e-6
U = 1822.88839
ANGS = 1.88973
DEB = 0.39343
EV = 0.03675
PM = 0.0188973
GHZ = 1.51983e-7
KCAL_MOL = 0.043 * EV
PS = 41341.374

Floating = NDArray[np.floating] | float

class AnimationConfig(Flag):
    No = 0
    Angular = 1
    Polar = 2
    Distance = 4
    Wave = 8
    Momentum = 16
    AngProjection = 32
    All = 63

def centrifugal(r_points, j_tot: int, omega: int, mass_u: float):
    return (j_tot * (j_tot + 1) - 2 * omega * omega) / (2 * mass_u * U * np.power(r_points, 2))
