import signal
import numpy as np
import scipy.interpolate as interp
import numpy.polynomial.legendre as leg
from split_op import Grid
import os

cm_inv = 4.5563352812e-6

def load_potential(path: str, filename: str, r_grid: Grid, polar_grid: Grid, kx, ky, is_gamma: bool, force_reload = False):
    if force_reload:
        return save_potential(path, filename, r_grid, polar_grid, kx, ky, is_gamma)

    grids_filepath = f"{path}{filename}_grid.npy"
    potential_filepath = f"{path}{filename}.npy"
    if os.path.isfile(grids_filepath) and os.path.isfile(potential_filepath):
        grids = np.load(grids_filepath)
        if same_grids(r_grid, polar_grid, grids):
            return np.load(potential_filepath)
        
    return save_potential(path, filename, r_grid, polar_grid, kx, ky, is_gamma)

def same_grids(r_grid: Grid, polar_grid: Grid, retrived_grids) -> bool:
    r_points = r_grid.points()
    polar_points = polar_grid.points()

    return retrived_grids[0] != r_points[0] or retrived_grids[1] != r_points[-1] or retrived_grids[2] != len(r_points) \
        or retrived_grids[3] != polar_points[0] or retrived_grids[4] != polar_points[-1] or retrived_grids[5] != len(polar_points) 

def save_potential(path: str, filename: str, r_grid: Grid, polar_grid: Grid, kx, ky, is_gamma: bool):
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    r_grid = np.array(r_grid.points()) # type: ignore
    polar_grid = np.array(polar_grid.points()) # type: ignore

    return potential_to_npy(path, filename, r_grid, polar_grid, lambda x: np.sqrt(x), lambda x: x**2, kx=kx, ky=ky) \
        if is_gamma else \
            potential_to_npy(path, filename, r_grid, polar_grid, kx=kx, ky=ky)

def potential_to_npy(path, filename, r_grid, theta_grid, transformation = None, inverse_transformation = None, kx=3, ky=3):
    thetas, rs, V_values = load_from_file(path, filename)

    rs_full = rs

    for i in range(len(thetas)):
        rs_extended_0, V_extended_0 = extrapolate_to_zero(rs, V_values[i])
        rs_extended_infty, V_extended_infty = extrapolate_to_infinity(rs, V_values[i], r_grid[-1])
        rs_full = rs_extended_0 + rs + rs_extended_infty
        V_values[i] = V_extended_0 + V_values[i] + V_extended_infty

    V_inv = (np.array(V_values).T)[::-1, :]
    rs_full_inv = (1 / np.array(rs_full))[::-1]

    interpolation = interp.RectBivariateSpline(rs_full_inv, thetas, V_inv if transformation == None else transformation(V_inv), kx=kx, ky=ky)

    V_grid_inv = interpolation((1 / r_grid)[::-1], theta_grid)
    V_grid = V_grid_inv[::-1, :]

    if transformation != None and inverse_transformation == None:
        raise Exception("inverse_transformation must be specified if transformation is specified")
    
    if inverse_transformation != None:
        V_grid = inverse_transformation(V_grid)
    
    np.save(path + filename.split(".")[0] + ".npy", V_grid)
    # save start, end and number of points in grids
    np.save(path + filename.split(".")[0] + "_grid.npy", [r_grid[0], r_grid[-1], r_grid.shape[0], theta_grid[0], theta_grid[-1], theta_grid.shape[0]])

    return V_grid

def load_from_file(path, filename):
    thetas = []
    rs = []
    V_values = []
    
    i = 0
    V_theta = []

    with open(path + filename) as file:
        contents = file.readlines()
        for line in contents:
            if line.startswith("# theta ="):
                i += 1
                thetas.append(float(line.split("=")[1]) * np.pi / 180)

                if i > 1:
                    V_values.append(V_theta)

                V_theta = []
                continue
            if line.startswith("#"):
                continue

            lineSplitted = line.split()
            if i == 1:
                rs.append(float(lineSplitted[0]))

            V_theta.append(float(lineSplitted[1]))
        V_values.append(V_theta)
    
    return thetas, rs, V_values

def extrapolate_to_zero(rs, V_values, max_val=10000):
    dr = rs[1] - rs[0]
    dVRatio = V_values[0] / V_values[1]
    dVRatio *= 1.005

    rs_extended = [rs[0] - dr]
    V_extended = [V_values[0] * dVRatio]

    dumping = 0
    while rs_extended[-1] > dr:
        if V_extended[-1] >= max_val:
            V_extended[-1] = max_val + 1000 * np.sqrt(dumping)
            dVRatio = 1
            dumping += 1
        else:
            dVRatio *= 1.005

        rs_extended.append(rs_extended[-1] - dr)
        V_extended.append(V_extended[-1] * dVRatio)

    rs_extended.reverse()
    V_extended.reverse()

    return rs_extended, V_extended

def extrapolate_to_infinity(rs, V_values, max_distance = 60):
    dr = rs[-1] - rs[-2]
    r_extended = [rs[-1] + dr]
    V_extended = [0.0]
    while(r_extended[-1] < max_distance):
        r_extended.append(r_extended[-1] + dr)
        V_extended.append(0.0)

    return r_extended, V_extended


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.special import roots_legendre
    import utility
    from potential import *
    from split_op import Grid
    from potential import cm_inv

    r_grid = Grid.linear_continuos("r", 50 / 1024, 50, 300, 0)
    r_points = r_grid.points()

    x, weights = roots_legendre(100)
    polar_points = np.flip(np.arccos(x))
    weights = np.flip(weights)

    polar_grid = Grid.custom("polar", polar_points, weights, 1)

    path = "potentials/"
    V_grid = load_potential(path, "potential.dat", r_grid, polar_grid, kx=5, ky=5, is_gamma=False, force_reload=True)