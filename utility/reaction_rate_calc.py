from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np

from .plotting import plot
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
        
        match self.omega_init:
            case 0:
                g_factor = 1
            case None:
                g_factor = 2 * self.j_init + 1
            case _:
                g_factor = 2
        
        losses_effective = (2 * self.j_tot_values + 1) * losses
        partial_cross_sections = np.pi / (2 * energy * mass) * losses_effective * g_factor / (2 * self.j_init + 1)
        cross_section: float = np.trapezoid(partial_cross_sections, self.j_tot_values) # type: ignore

        conversion = (5.29177210903)**3 * 10**(-10) / 2.4188843265
        reaction_rate = np.sqrt((2 * energy / mass)) * cross_section * conversion

        return Reaction(ionization_type, partial_cross_sections, cross_section, reaction_rate)

class ParamType(Enum):
    EnergyKelvin = 0
    MassScaling = 1
    Other = 2

@dataclass
class ReactionDependence:
    parameters: FloatNDArray
    path: str
    ionization_type: Ionization
    param_type: ParamType = ParamType.Other

    unchanged_value: Optional[float] = None
    unchanged_filename_prefix = "losses_3700"

    def get_reaction_rates(self, file_pattern: Callable[[float], str], j_init: int, omega_init: int) -> FloatNDArray:
        rates = []
        for parameter in self.parameters:

            if parameter != self.unchanged_value:
                filename = file_pattern(parameter)
                losses = Losses(self.path, filename, j_init, omega_init)
            else:
                losses = Losses(self.path, f"{self.unchanged_filename_prefix}_{j_init}_{omega_init}.dat", j_init, omega_init)


            match self.param_type:
                case ParamType.EnergyKelvin:
                    reaction = losses.get_reaction(self.ionization_type, energy = parameter * KELVIN)
                case ParamType.MassScaling:
                    reaction = losses.get_reaction(self.ionization_type, mass = 27535.2484 * parameter)
                case ParamType.Other:
                    reaction = losses.get_reaction(self.ionization_type)
            rates.append(reaction.reaction_rate)

        return np.array(rates)
    
    def plot_ratios(self, file_pattern: Callable[[float, int, int], str], fig_ax: Optional[tuple[Figure, Axes]] = None) -> tuple[Figure, Axes]:
        if fig_ax is None:
            fig, ax = plot()
            ax.set_ylabel("Reaction rate ratio")
        else:
            fig, ax = fig_ax

        r00 = self.get_reaction_rates(lambda x: file_pattern(x, 0, 0), 0, 0)
        r10 = self.get_reaction_rates(lambda x: file_pattern(x, 1, 0), 1, 0)
        r11 = self.get_reaction_rates(lambda x: file_pattern(x, 1, 1), 1, 1)

        match self.ionization_type:
            case Ionization.Penning:
                ax.plot(self.parameters, (r10 + r11) / r00, "o-", label = "PI", color = "red")
                ax.plot(self.parameters, r10 / r00, "o-", label = r"PI, $\Omega=0$", color = "darkorange")
                ax.plot(self.parameters, r11 / r00, "o-", label = r"PI, $|\Omega|=1$", color = "orange")
            case Ionization.Dissociative:
                ax.plot(self.parameters, (r10 + r11) / r00, "o-", label = "DI", color = "blue")
                ax.plot(self.parameters, r10 / r00, "o-", label = r"DI, $\Omega=0$", color = "steelblue")
                ax.plot(self.parameters, r11 / r00, "o-", label = r"DI, $|\Omega|=1$", color = "deepskyblue")
                
        ax.legend()
        return fig, ax
    
    def plot_j_0(self, file_pattern: Callable[[float], str], fig_ax: Optional[tuple[Figure, Axes]] = None) -> tuple[Figure, Axes]:
        if fig_ax is None:
            fig, ax = plot()
            ax.set_ylabel("Reaction rate [cm$^3$/s]")
        else:
            fig, ax = fig_ax

        r00 = self.get_reaction_rates(file_pattern, 0, 0)

        match self.ionization_type:
            case Ionization.Penning:
                ax.plot(self.parameters, r00, "o-", label = "PI", color = "red")
            case Ionization.Dissociative:
                ax.plot(self.parameters, r00, "o-", label = "DI", color = "blue")

        ax.legend()
        return fig, ax
    
    def plot_j_1(self, file_pattern: Callable[[float, int, int], str], fig_ax: Optional[tuple[Figure, Axes]] = None) -> tuple[Figure, Axes]:
        if fig_ax is None:
            fig, ax = plot()
            ax.set_ylabel("Reaction rate [cm$^3$/s]")
        else:
            fig, ax = fig_ax

        r10 = self.get_reaction_rates(lambda x: file_pattern(x, 1, 0), 1, 0)
        r11 = self.get_reaction_rates(lambda x: file_pattern(x, 1, 1), 1, 1)

        match self.ionization_type:
            case Ionization.Penning:
                ax.plot(self.parameters, r11 + r10, "o-", label = r"PI, $j=1$", color = "red")
                ax.plot(self.parameters, r10, "o-", label = r"PI, $j=1, \Omega=0$", color = "darkorange")
                ax.plot(self.parameters, r11, "o-", label = r"PI, $j=1, |\Omega|=1$", color = "orange")
            case Ionization.Dissociative:
                ax.plot(self.parameters, r11 + r10, "o-", label = r"DI, $j=1$", color = "blue")
                ax.plot(self.parameters, r10, "o-", label = r"DI, $j=1, \Omega=0$", color = "steelblue")
                ax.plot(self.parameters, r11, "o-", label = r"DI, $j=1, |\Omega|=1$", color = "deepskyblue")

        ax.legend()
        return fig, ax
