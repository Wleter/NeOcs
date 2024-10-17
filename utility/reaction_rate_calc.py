from dataclasses import dataclass
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import shutil
import os
from .units import KELVIN
from .my_types import FloatNDArray

class Ionization(Enum):
    Penning = 1
    Dissociative = 2

    def reaction_name(self) -> str:
        return "PI" if self.value == 1 else "DI"

@dataclass
class Reaction:
    ionization_type: Ionization
    partial_cross_sections: FloatNDArray
    cross_section: float
    reaction_rate: float

@dataclass
class Losses:
    j_tot_values: FloatNDArray
    dissociative_losses: FloatNDArray
    penning_losses: FloatNDArray

    j_init: int
    omega_init: int | None

    def __init__(self, path: str, filename: str, j_init: int, omega_init: int | None):
        values = np.loadtxt(path + filename, skiprows=1)

        self.j_tot_values = values[:, 0]
        self.dissociative_losses = values[:, 1]
        self.penning_losses = values[:, 2]

        self.j_init = j_init
        self.omega_init = omega_init

    def get_reaction(self, ionization_type: Ionization, energy: float = 3700 * KELVIN, mass: float = 27535.2484) -> Reaction:
        match ionization_type:
            case Ionization.Penning:
                losses = self.penning_losses
            case Ionization.Dissociative:
                losses = self.dissociative_losses
        
        g_factor = 1 if self.omega_init == 0 or self.omega_init is None else 2
        
        losses_effective = (2 * self.j_tot_values + 1) * losses
        partial_cross_sections = np.pi / (2 * energy * mass) * losses_effective * g_factor / (2 * self.j_init + 1)
        cross_section: float = np.trapezoid(partial_cross_sections, self.j_tot_values) # type: ignore

        conversion = (5.29177210903)**3 * 10**(-10) / 2.4188843265
        reaction_rate = np.sqrt((2 * energy / mass)) * cross_section * conversion

        return Reaction(ionization_type, partial_cross_sections, cross_section, reaction_rate)

def get_reaction_rate_dependence(prefix, changing_parameter, is_energy_parameter = False, is_mass_scaling = False, identity = 1.0):
    energy_kelvin = 3700
    energy = energy_kelvin * 3.1668105e-6
    path = "../data/"

    xpi0 = []
    bsigma0 = []

    xpi10 = []
    xpi11 = []

    bsigma10 = []
    bsigma11 = []

    for parameter in changing_parameter:
        if is_energy_parameter:
            energy = parameter * KELVIN
        if is_mass_scaling:
            mass = parameter * 27535.2484

        if parameter == identity and (os.path.exists(f"{path}/{prefix}_{parameter}_0_0.dat") == False
                                or os.path.exists(f"{path}/{prefix}_{parameter}_1_0.dat") == False
                                or os.path.exists(f"{path}/{prefix}_{parameter}_1_1.dat") == False):
            print(f"Copied data from 3700 energy calculation to {prefix}_{parameter}")
            shutil.copy(f"{path}/losses_3700_0_0.dat", f"{path}/{prefix}_{parameter}_0_0.dat")
            shutil.copy(f"{path}/losses_3700_1_0.dat", f"{path}/{prefix}_{parameter}_1_0.dat")
            shutil.copy(f"{path}/losses_3700_1_1.dat", f"{path}/{prefix}_{parameter}_1_1.dat")

        losses = read_losses(f"{path}/{prefix}_{parameter}_0_0.dat")
        Js = losses[:, 0]
        BSigma_losses = losses[:, 1]
        XPi_losses = losses[:, 2]
        xpi0.append(get_reaction_rate(mass, energy, 0, 0, Js, XPi_losses))
        bsigma0.append(get_reaction_rate(mass, energy, 0, 0, Js, BSigma_losses))

        losses = read_losses(f"{path}/{prefix}_{parameter}_1_0.dat")
        Js = losses[:, 0]
        BSigma_losses = losses[:, 1]
        XPi_losses = losses[:, 2]
        xpi10.append(get_reaction_rate(mass, energy, 1, 0, Js, XPi_losses))
        bsigma10.append(get_reaction_rate(mass, energy, 1, 0, Js, BSigma_losses))

        losses = read_losses(f"{path}/{prefix}_{parameter}_1_1.dat")
        Js = losses[:, 0]
        BSigma_losses = losses[:, 1]
        XPi_losses = losses[:, 2]
        xpi11.append(get_reaction_rate(mass, energy, 1, 1, Js, XPi_losses))
        bsigma11.append(get_reaction_rate(mass, energy, 1, 1, Js, BSigma_losses))

    xpi0 = np.array(xpi0)
    xpi10 = np.array(xpi10)
    xpi11 = np.array(xpi11)
    bsigma0 = np.array(bsigma0)
    bsigma10 = np.array(bsigma10)
    bsigma11 = np.array(bsigma11)

    return xpi0, xpi10, xpi11, bsigma0, bsigma10, bsigma11

def plot_reaction_rate_dependence_0(xlabel, parameters, xpi0, bsigma0, position = None):
    fig, ax = utility.plot()
    ax.plot(parameters, xpi0, "o-", label=r"$X \Pi$, j=0")
    ax.plot(parameters, bsigma0, "o-", label=r"$A + B$, j=0")

    if position is not None:
        ax.legend(loc=position)
    else:
        ax.legend()

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Reaction rate [cm$^3$/s]")

    return fig, ax

def plot_reaction_rate_dependence_1(xlabel, parameters, xpi10, xpi11, bsigma10, bsigma11, position = None):
    fig, ax = utility.plot()
    ax.plot(parameters, bsigma10 + bsigma11, "o-", label=r"$A + B, j=1$", color="blue")
    ax.plot(parameters, bsigma10, "o-", label=r"$A + B, j=1, \Omega=0$", color="steelblue")
    ax.plot(parameters, bsigma11, "o-", label=r"$A + B, j=1, |\Omega|=1$", color="deepskyblue")

    ax.plot(parameters, xpi10 + xpi11, "o-", label=r"$X \Pi, j=1$", color = "red")
    ax.plot(parameters, xpi10, "o-", label=r"$X \Pi, j=1, \Omega=0$", color="darkorange")
    ax.plot(parameters, xpi11, "o-", label=r"$X \Pi, j=1, |\Omega|=1$", color="orange")

    if position is not None:
        ax.legend(loc=position)
    else:
        ax.legend()
        
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Reaction rate [cm$^3$/s]")

    return fig, ax

def plot_ratio_dependence(xlabel, parameters, xpi0, xpi10, xpi11, bsigma0, bsigma10, bsigma11, sub_ratios = True, position = None):
    fig, ax = utility.plot()
    ax.plot(parameters, (bsigma10 + bsigma11) / bsigma0, "o-", label="$A + B$", color="blue")
    if sub_ratios:
        ax.plot(parameters, bsigma10 / bsigma0, "o-", label=r"$A + B, \Omega=0$", color="steelblue")
        ax.plot(parameters, bsigma11 / bsigma0, "o-", label=r"$A + B, |\Omega|=1$", color="deepskyblue")

    ax.plot(parameters, (xpi10 + xpi11) / xpi0, "o-", label=r"$X \Pi$", color = "red")
    if sub_ratios:
        ax.plot(parameters, xpi10 / xpi0, "o-", label=r"$X \Pi, \Omega=0$", color="darkorange")
        ax.plot(parameters, xpi11 / xpi0, "o-", label=r"$X \Pi, |\Omega|=1$", color="orange")

    if position is not None:
        ax.legend(loc=position)
    else:
        ax.legend()

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Reaction rate ratio")
    return fig, ax