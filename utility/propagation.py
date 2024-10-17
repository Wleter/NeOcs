from enum import Flag
from my_types import Floating
from units import U

DATA_PATH = "../data/"

class AnimationConfig(Flag):
    No = 0
    Angular = 1
    Polar = 2
    Distance = 4
    Wave = 8
    Momentum = 16
    AngProjection = 32
    All = 63

def centrifugal(r_points: Floating, j_tot: int, omega: int, mass_u: float) -> Floating:
    return (j_tot * (j_tot + 1) - 2 * omega * omega) / (2 * mass_u * U * r_points**2)
