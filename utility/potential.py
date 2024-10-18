from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np
import scipy.interpolate as interp
from split_op import Grid
import os
from .my_types import Floating, FloatNDArray
from .units import ANGS, KCAL_MOL, DEB
from scipy.interpolate import CubicSpline

@dataclass
class PotentialArray:
    radial: FloatNDArray
    polar: FloatNDArray
    values: FloatNDArray
    """
    Values of the potential with shape V[radial, polar]
    """

def load_from_file(path: str, filename: str) -> PotentialArray:
    polar = []
    radial = []
    V_values = []
    
    i = 0
    V_theta = []

    with open(path + filename) as file:
        contents = file.readlines()
        for line in contents:
            if line.startswith("# theta ="):
                i += 1
                polar.append(float(line.split("=")[1]) * np.pi / 180)
                if i > 1:
                    V_values.append(V_theta)

                V_theta = []
                continue

            if line.startswith("#"):
                continue

            lineSplitted = line.split()
            if i == 1:
                radial.append(float(lineSplitted[0]))

            V_theta.append(float(lineSplitted[1]))
        V_values.append(V_theta)
    
    return PotentialArray(np.array(radial), np.array(polar), np.array(V_values).T)

class PotentialRetriever:
    def __init__(self, r_grid: Grid, polar_grid: Grid, path: str):
        self._path = path
        self._radial = np.array(r_grid.points())
        self._polar = np.array(polar_grid.points())

    def load_interpolated(self, filename: str, is_gamma: bool) -> PotentialArray:
        grids_filepath = f"{self._path}{filename}_grid.npy"
        potential_filepath = f"{self._path}{filename}.npy"

        if os.path.isfile(grids_filepath) and os.path.isfile(potential_filepath):
            grids: FloatNDArray = np.load(grids_filepath)

            if self._same_grids(grids):
                return PotentialArray(self._radial, self._polar, np.load(potential_filepath))
        
        return self.reload_interpolated(filename, is_gamma)

    def _same_grids(self, retrived_grids: FloatNDArray) -> bool:
        return retrived_grids[0] != self._radial[0] or retrived_grids[1] != self._radial[-1] or retrived_grids[2] != len(self._radial) \
            or retrived_grids[3] != self._polar[0] or retrived_grids[4] != self._polar[-1] or retrived_grids[5] != len(self._polar) 

    def reload_interpolated(self, filename: str, is_gamma: bool) -> PotentialArray:
        potential_data = load_from_file(self._path, filename)

        radial_ext0, values_ext0 = self._extrapolate_to_zero(potential_data)
        radial_ext_inf, values_ext_inf = self._extrapolate_to_inf(potential_data)

        radial_ext = np.concat((radial_ext0, potential_data.radial, radial_ext_inf))
        values_ext = np.concat((values_ext0, potential_data.values, values_ext_inf), axis=0)

        if is_gamma:
            values_ext = np.power(values_ext, 1/16)

        values_inv = values_ext[::-1, :]
        radial_inv = (1 / radial_ext)[::-1]

        interpolation_polar = np.zeros((radial_ext.shape[0], self._polar.shape[0]))
        for i in range(radial_ext.shape[0]):
            interpolation = CubicSpline(potential_data.polar, values_inv[i, :], bc_type="clamped")

            interpolation_polar[i, :] = interpolation(self._polar)

        values_grid = np.zeros((self._radial.shape[0], self._polar.shape[0]))
        for i in range(self._polar.shape[0]):
            interpolation = CubicSpline(radial_inv, interpolation_polar[:, i])

            values_grid[:, i] = interpolation(1 / self._radial)

        values_grid = values_grid[::-1, :]

        if is_gamma:
            values_grid = np.power(values_grid, 16)

        np.save(self._path + filename.split(".")[0] + ".npy", values_grid)
        np.save(self._path + filename.split(".")[0] + "_grid.npy", [
            self._radial[0], self._radial[-1], self._radial.shape[0],
            self._polar[0], self._polar[-1], self._polar.shape[0],
        ])

        return PotentialArray(self._radial, self._polar, values_grid)
    
    def get_extended(self, filename: str) -> PotentialArray:
        potential_data = load_from_file(self._path, filename)

        radial_ext0, values_ext0 = self._extrapolate_to_zero(potential_data)
        radial_ext_inf, values_ext_inf = self._extrapolate_to_inf(potential_data)

        radial_ext = np.concat((radial_ext0, potential_data.radial, radial_ext_inf))
        values_ext = np.concat((values_ext0, potential_data.values, values_ext_inf), axis=0)

        return PotentialArray(radial_ext, potential_data.polar, values_ext)


    def _extrapolate_to_zero(self, potential_data: PotentialArray, max_val: float = 10000) -> tuple[FloatNDArray, FloatNDArray]:
        dr = potential_data.radial[1] - potential_data.radial[0]

        radial_ext = []
        values = []
        for angle_i in range(len(potential_data.polar)):
            values_ratio = potential_data.values[0, angle_i] / potential_data.values[1, angle_i]
            values_ratio *= 1.005
        
            radial_ext = [potential_data.radial[0] - dr]
            values_ext = [potential_data.values[0, angle_i] * values_ratio]

            dumping = 0
            while radial_ext[-1] > dr:
                if values_ext[-1] >= max_val:
                    values_ext[-1] = max_val + 1000 * np.sqrt(dumping)
                    values_ratio = 1
                    dumping += 1
                else:
                    values_ratio *= 1.005
            
                radial_ext.append(radial_ext[-1] - dr)
                values_ext.append(values_ext[-1] * values_ratio)

            radial_ext.reverse()
            values_ext.reverse()
            values.append(values_ext)

        return np.array(radial_ext), np.array(values).T

    def _extrapolate_to_inf(self, potential_data: PotentialArray) -> tuple[FloatNDArray, FloatNDArray]:
        dr = potential_data.radial[-1] - potential_data.radial[-2]

        radial_ext = [potential_data.radial[-1] + dr]
        while radial_ext[-1] < self._radial[-1]:
            radial_ext.append(radial_ext[-1] + dr)
        
        values_ext = np.zeros((len(radial_ext), len(potential_data.polar)))

        return np.array(radial_ext), values_ext
    
def positions_from_center_mass(oc_bond: float, cs_bond: float) -> tuple[float, float, float]:
    mass_O = 15.999
    mass_C = 12.011
    mass_S = 32.06

    center_mass_from_O = (mass_O * oc_bond - mass_S * cs_bond) / (mass_O + mass_C + mass_S)

    return (oc_bond - center_mass_from_O, -center_mass_from_O, -cs_bond - center_mass_from_O)

def _get_lennard_jones(eps_a, eps_b, half_r_a, half_r_b) -> tuple[float, float]:
    repulsive = np.sqrt(eps_a * eps_b) * (half_r_a + half_r_b) ** 12
    attractive = 2 * np.sqrt(eps_a * eps_b) * (half_r_a + half_r_b) ** 6

    return (repulsive, attractive)

class ForceField:
    def __init__(self, oc_bond: float, cs_bond: float, 
                 rep_o: float, rep_c: float, rep_s: float, 
                 attr_o: float, attr_c: float, attr_s: float, 
                 alpha: float
    ):
        self.r_o, self.r_c, self.r_s = positions_from_center_mass(oc_bond, cs_bond)
        self.rep_o = rep_o
        self.rep_c = rep_c
        self.rep_s = rep_s
        self.attr_o = attr_o
        self.attr_c = attr_c
        self.attr_s = attr_s
        self.alpha = alpha

    @staticmethod
    def theoretical() -> "ForceField":
        oc_bond = 1.5710 * ANGS
        cs_bond = 1.1526 * ANGS
        
        eps_ne = -9.4352 * KCAL_MOL
        half_r_ne = 1.7727 * ANGS
        rep_o, attr_o = _get_lennard_jones(-0.18 * KCAL_MOL, eps_ne, 1.79 * ANGS, half_r_ne)
        rep_c, attr_c = _get_lennard_jones(-0.18 * KCAL_MOL, eps_ne, 1.87 * ANGS, half_r_ne)
        rep_s, attr_s = _get_lennard_jones(-0.45 * KCAL_MOL, eps_ne, 2 * ANGS, half_r_ne)

        alpha = 0.7 * DEB * 27.8 * ANGS ** 3

        return ForceField(oc_bond, cs_bond,
                         rep_o, rep_c, rep_s,
                         attr_o, attr_c, attr_s,
                         alpha)

    @staticmethod
    def fitted() -> "ForceField":
        return ForceField(1.68136775, 2.76987627, 
                          1.42041335e+07, 9.92743969e+07, 1.45642651e+08, 
                          1.19155634e-02, 2.64214008e+02, 2.51866931e+02,
                          93.1398194)

    def value(self, r: Floating, theta: Floating) -> Floating:
        r_o = np.sqrt(r ** 2 + self.r_o ** 2 - 2 * r * self.r_o * np.cos(theta))
        r_c = np.sqrt(r ** 2 + self.r_c ** 2 - 2 * r * self.r_c * np.cos(theta))
        r_s = np.sqrt(r ** 2 + self.r_s ** 2 - 2 * r * self.r_s * np.cos(theta))
    
        pot_o = self.rep_o / r_o ** 12 - self.attr_o / r_o ** 6
        pot_c = self.rep_c / r_c ** 12 - self.attr_c / r_c ** 6
        pot_s = self.rep_s / r_s ** 12 - self.attr_s / r_s ** 6

        cos_theta_shifted = (r * np.cos(theta) - self.r_c) / r_c

        pot_dip = -self.alpha * (1 + 3 * cos_theta_shifted ** 2) / (2 * r_c ** 6)

        return pot_o + pot_c + pot_s + pot_dip
