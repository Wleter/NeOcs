import numpy as np
import matplotlib.pyplot as plt
import shutil
import os
import utility

def read_losses(filename):
    return np.loadtxt(filename, skiprows=1)

def calc_partial_cross_sections(mass, energy, j_init, omega_init, Js, losses):
    losses_eff = np.array(list(map(lambda J, P: (2*J + 1) * P, Js, losses)))

    return np.pi / (2 * energy * mass) * losses_eff * (omega_init + 1) / (2 * j_init + 1)

def calc_cross_section(Js, partial_cross_sections):
    return np.trapezoid(partial_cross_sections, Js)

def calc_reaction_rate(mass, energy, cross_section):
    convertion = (5.29177210903)**3 * 10**(-10) / 2.4188843265
    return np.sqrt((2 * energy / mass)) * cross_section * convertion

def get_reaction_rate(mass, energy, j_init, omega_init, Js, losses):
    partial_cross_sections = calc_partial_cross_sections(mass, energy, j_init, omega_init, Js, losses)
    cross_section = calc_cross_section(Js, partial_cross_sections)

    return calc_reaction_rate(mass, energy, cross_section)

def get_reaction_rate_dependence(prefix, changing_parameter, is_energy_parameter = False, is_mass_factor_parameter = False, identity = 1.0):
    mass = 27535.24841189485
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
            energy = parameter * 3.1668105e-6
        if is_mass_factor_parameter:
            mass = parameter * 27535.24841189485

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