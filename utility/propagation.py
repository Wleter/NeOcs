from dataclasses import dataclass
from enum import Flag
from typing import Callable, Optional
from .my_types import Floating
from .units import KELVIN, U
from .potential import GammaRetriever, ForceField, Potential
import split_op as split
from scipy.special import roots_legendre, lpmv
import numpy as np

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

class CumulativeLosses:
    def __init__(self, j_init: int, energy_kelvin: float = 3700) -> None:
        self.bsigma_losses = []
        self.xpi_losses = []

        self.j_totals = [int(j_init + np.ceil(i * 5.5 * np.sqrt(energy_kelvin / 2000))) for i in range(50)]
    
    def extract_loss(self, propagation: split.Propagation) -> None:
        losses = propagation.get_losses()

        self.bsigma_losses.append(losses[0])
        self.xpi_losses.append(losses[1])

    def save_losses(self, filename: str) -> None:
        combined = np.vstack((self.j_totals, self.bsigma_losses, self.xpi_losses)).transpose()
        np.savetxt(f"{DATA_PATH}/{filename}.dat", combined, delimiter=" ", header="j_total\tbsigma_loss\txpi_loss")

@dataclass
class AtomsParams:
    """
        Ne OCS Default values.
    """
    rot_const: float = 9.243165268327e-7

    wave_r0: float = 20
    wave_r_sigma: float = 0.6

    mass_u: float = 15.1052848671
    energy_kelvin: float = 3700

@dataclass
class SpinOne:
    population_1: float
    phase_proj_1: float
    population_minus_1: float
    phase_proj_minus_1: float

@dataclass
class PropagationConfig:
    j_init: int
    omega_init: int | SpinOne
    j_tot: int

    time_step: float = 200
    steps_no: int = 150

    r_start: float = 5 + 25 / 512
    r_end: float = 30
    r_no: int = 512

    polar_no: int = 128
    coriolis_omega_max: int = 0

    im_time = False

    animation: AnimationConfig = AnimationConfig.No
    frames: int = 60

@dataclass
class Transform:
    t: Callable[[Floating, Floating], Floating]

@dataclass
class PotentialProvider:
    potential: Potential = ForceField.fitted()
    potentials_path: str = "../potentials/"

    transform_potential: Optional[Transform] | float = None
    transform_gamma: Optional[Transform] | float = None

class Propagation:
    def __init__(self, save_prefix: str, j_init: int, omega_init: int | SpinOne, j_tot: int):
        self.save_prefix = save_prefix
        self.params = AtomsParams()
        self.config = PropagationConfig(j_init, omega_init, j_tot)
        self.provider = PotentialProvider()

    @staticmethod
    def new(save_prefix: str, params: AtomsParams, config: PropagationConfig, potential_provider: PotentialProvider) -> "Propagation":
        propagation = Propagation(save_prefix, 0, 0, 0)
        propagation.params = params
        propagation.config = config
        propagation.provider = potential_provider

        return propagation

    def into_split(self) -> split.Propagation:
        assert self.config.coriolis_omega_max >= 0

        if self.config.coriolis_omega_max == 0:
            return self._prepare_no_coriolis()
        else:
            return self._prepare_with_coriolis()

    def _prepare_no_coriolis(self) -> split.Propagation:
        assert type(self.config.omega_init) == int, f"wrong type of omega_init {type(self.config.omega_init)}"

        omega_init: int = self.config.omega_init # type: ignore
        j_init = self.config.j_init
        j_tot = self.config.j_tot

        assert j_init >= omega_init
        assert j_tot >= omega_init

        ############# grids
        time_grid = split.TimeGrid(self.config.time_step, self.config.steps_no, im_time = self.config.im_time)
        r_grid = split.Grid.linear_continuos("r", self.config.r_start, self.config.r_end, self.config.r_no, 0)
        
        polar_points, weights = roots_legendre(self.config.polar_no)
        polar_points = np.flip(np.arccos(polar_points))
        weights = np.flip(weights)
        
        polar_grid = split.Grid.custom("theta", polar_points, weights, 1)

        ########## initial wave function creation
        r_points = np.array(r_grid.points())
        momentum = np.sqrt(2 * self.params.mass_u * U * self.params.energy_kelvin * KELVIN)

        wave_r_init = np.array([
            split.gaussian_distribution(r_points[i], self.params.wave_r0, self.params.wave_r_sigma, momentum) for i in range(self.config.r_no)
        ])
        wave_polar_init = lpmv(omega_init, j_init, np.cos(polar_points))

        wave_init = np.outer(wave_r_init, wave_polar_init)
        wave_function = split.WaveFunction(wave_init.flatten(), [r_grid, polar_grid])

        ########## operator creation
        r_mesh, polar_mesh = np.meshgrid(r_points, polar_points, indexing="ij")

        potential = self.provider.potential.value(r_mesh, polar_mesh)
        match self.provider.transform_potential:
            case float():
                potential *= self.provider.transform_potential
            case Transform():
                potential *= self.provider.transform_potential.t(r_mesh, polar_mesh)
            case None:
                pass
            case _:
                raise Exception(f"wrong type of transform potential {type(self.provider.transform_potential)}")

        centrifugal_potential = centrifugal(r_points, j_tot, omega_init, self.params.mass_u)
        centrifugal_potential = np.broadcast_to(np.expand_dims(centrifugal_potential, 1), (self.config.r_no, self.config.polar_no))

        gamma_retriever = GammaRetriever(r_grid, polar_grid, self.provider.potentials_path)
        
        bsigma_gamma = gamma_retriever.load_interpolated("BSigma_gamma.dat").values
        api_gamma = gamma_retriever.load_interpolated("APi_gamma.dat").values
        xpi_gamma = gamma_retriever.load_interpolated("XPi_gamma.dat").values

        match self.provider.transform_gamma:
            case float():
                bsigma_gamma *= self.provider.transform_gamma
                api_gamma *= self.provider.transform_gamma
                xpi_gamma *= self.provider.transform_gamma
            case Transform():
                bsigma_gamma *= self.provider.transform_gamma.t(r_mesh, polar_mesh)
                api_gamma *= self.provider.transform_gamma.t(r_mesh, polar_mesh)
                xpi_gamma *= self.provider.transform_gamma.t(r_mesh, polar_mesh)
            case None:
                pass
            case _:
                raise Exception(f"wrong type of transform potential {type(self.provider.transform_gamma)}")

        potential = potential + centrifugal_potential + complex(0, -0.5) * (bsigma_gamma + api_gamma)
        potential_with_bsigma_prop = split.complex_n_dim_into_propagator(potential.shape, potential.flatten(), time_grid)

        loss_checker = split.LossChecker.new_with_saver("bsigma", self.config.frames, f"../data/{self.save_prefix}_bsigma", time_grid) \
            if self.config.animation else split.LossChecker("bsigma")
        potential_with_bsigma_prop.set_loss_checked(loss_checker)

        xpi_gamma = complex(0, -0.5) * xpi_gamma
        xpi_gamma_prop = split.complex_n_dim_into_propagator(xpi_gamma.shape, xpi_gamma.flatten(), time_grid)
    
        loss_checker = split.LossChecker.new_with_saver("xpi", self.config.frames, f"../data/{self.save_prefix}_xpi", time_grid) \
            if self.config.animation else split.LossChecker("xpi")
        xpi_gamma_prop.set_loss_checked(loss_checker)

        leak_control = split.LeakControl(split.LossChecker("leak control"))
        dumping_border = split.BorderDumping(1., 3., r_grid)

        angular_transformation = split.associated_legendre_transformation(polar_grid, omega_init)

        shape, angular_kinetic_op = split.rotational_hamiltonian(r_grid, polar_grid, self.params.mass_u, self.params.rot_const)
        angular_prop = split.n_dim_into_propagator(shape, angular_kinetic_op, time_grid)

        fft_transformation = split.FFTTransformation(r_grid, "r momentum")

        kinetic_op = split.kinetic_hamiltonian(r_grid, self.params.mass_u, self.params.energy_kelvin)
        kinetic_prop = split.one_dim_into_propagator(kinetic_op, r_grid, time_grid, step = "full")

        ########## operation stack creation
        operation_stack = split.OperationStack()

        if self.config.im_time:
            leak_control.add_operation(operation_stack)

        potential_with_bsigma_prop.add_operation(operation_stack)
        xpi_gamma_prop.add_operation(operation_stack)

        if AnimationConfig.Wave in self.config.animation:
            wave_saver = split.WaveFunctionSaver(f"{DATA_PATH}/{self.save_prefix}_wave_animation", time_grid, r_grid, polar_grid, self.config.frames)
            wave_saver.add_operation(operation_stack)

        if AnimationConfig.Distance in self.config.animation:
            distance_saver = split.StateSaver(f"{DATA_PATH}/{self.save_prefix}_distance_animation", time_grid, r_grid, self.config.frames)
            distance_saver.add_operation(operation_stack)

        if AnimationConfig.Polar in self.config.animation:
            polar_saver = split.StateSaver(f"{DATA_PATH}/{self.save_prefix}_polar_animation", time_grid, polar_grid, self.config.frames)
            polar_saver.add_operation(operation_stack)

        if not self.config.im_time:
            dumping_border.add_operation(operation_stack)
            leak_control.add_operation(operation_stack)

        angular_transformation.add_operation(operation_stack, True)
        if AnimationConfig.Angular in self.config.animation:
            angular_grid = angular_transformation.transformed_grid()
            wave_legendre_saver = split.StateSaver(f"{DATA_PATH}/{self.save_prefix}_angular_animation", time_grid, angular_grid, self.config.frames)
            wave_legendre_saver.add_operation(operation_stack)

        angular_prop.add_operation(operation_stack)

        fft_transformation.add_operation(operation_stack, True)
        if AnimationConfig.Momentum in self.config.animation: 
            fft_grid = fft_transformation.transformed_grid()
            wave_legendre_saver = split.StateSaver(f"{DATA_PATH}/{self.save_prefix}_momentum_animation", time_grid, fft_grid, self.config.frames)
            wave_legendre_saver.add_operation(operation_stack)

        kinetic_prop.add_operation(operation_stack)

        ######### propagation creation

        propagation = split.Propagation()
        propagation.set_wave_function(wave_function)
        propagation.set_time_grid(time_grid)
        propagation.set_operation_stack(operation_stack)

        return propagation
    
    def _prepare_with_coriolis(self) -> split.Propagation:
        omega_init = self.config.omega_init
        j_init = self.config.j_init
        j_tot = self.config.j_tot

        match omega_init:
            case int():
                assert j_init >= omega_init
                assert j_tot >= omega_init
            case SpinOne():
                assert j_tot >= 1
                assert j_init >= 1

        ############# grids
        time_grid = split.TimeGrid(self.config.time_step, self.config.steps_no, im_time = self.config.im_time)
        r_grid = split.Grid.linear_continuos("r", self.config.r_start, self.config.r_end, self.config.r_no, 0)
        
        polar_points, weights = roots_legendre(self.config.polar_no)
        polar_points = np.flip(np.arccos(polar_points))
        weights = np.flip(weights)
        
        polar_grid = split.Grid.custom("theta", polar_points, weights, 1)

        omega_max = self.config.coriolis_omega_max
        omega_no = 2 * self.config.coriolis_omega_max + 1
        omega_shift = self.config.coriolis_omega_max
        omega_dim_nr = 2

        omega_grid = split.Grid.linear_countable("omega", -omega_max, omega_max, omega_no, omega_dim_nr)

        ########## initial wave function creation
        r_points = np.array(r_grid.points())
        momentum = np.sqrt(2 * self.params.mass_u * U * self.params.energy_kelvin * KELVIN)

        wave_r_init = np.array([
            split.gaussian_distribution(r_points[i], self.params.wave_r0, self.params.wave_r_sigma, momentum) for i in range(self.config.r_no)
        ])

        wave_full_init = np.zeros((self.config.r_no, self.config.polar_no, omega_no), dtype=complex)
        match omega_init:
            case int():
                wave_polar_init = lpmv(omega_init, j_init, np.cos(polar_points))
                wave_init = np.outer(wave_r_init, wave_polar_init)
                wave_full_init[:, :, omega_shift+omega_init] = wave_init
            case SpinOne():
                population_zero = 1 - omega_init.population_1 - omega_init.population_minus_1
                wave_polar_init = np.sqrt(3 / 2 * population_zero) * lpmv(0, 1, np.cos(polar_points))
                wave_init = np.outer(wave_r_init, wave_polar_init)
                wave_full_init[:, :, omega_shift] = wave_init

                wave_polar_init = np.sqrt(3 / 4 * omega_init.population_1) * lpmv(1, 1, np.cos(polar_points))
                wave_init = np.outer(wave_r_init, wave_polar_init)
                wave_full_init[:, :, omega_shift+1] = wave_init * np.exp(1j * omega_init.phase_proj_1)

                wave_polar_init = np.sqrt(3 * omega_init.population_minus_1) * lpmv(-1, 1, np.cos(polar_points))
                wave_init = np.outer(wave_r_init, wave_polar_init)
                wave_full_init[:, :, omega_shift-1] = wave_init * np.exp(1j * omega_init.phase_proj_minus_1)

        wave_function = split.WaveFunction(wave_full_init.flatten(), [r_grid, polar_grid, omega_grid])

        ########## operator creation
        r_mesh, polar_mesh = np.meshgrid(r_points, polar_points, indexing="ij")

        potential = self.provider.potential.value(r_mesh, polar_mesh)
        match self.provider.transform_potential:
            case float():
                potential *= self.provider.transform_potential
            case Transform():
                potential *= self.provider.transform_potential.t(r_mesh, polar_mesh)
            case None:
                pass
            case _:
                raise Exception(f"wrong type of transform potential {type(self.provider.transform_potential)}")

        gamma_retriever = GammaRetriever(r_grid, polar_grid, self.provider.potentials_path)
        
        bsigma_gamma = gamma_retriever.load_interpolated("BSigma_gamma.dat").values
        api_gamma = gamma_retriever.load_interpolated("APi_gamma.dat").values
        xpi_gamma = gamma_retriever.load_interpolated("XPi_gamma.dat").values

        match self.provider.transform_gamma:
            case float():
                bsigma_gamma *= self.provider.transform_gamma
                api_gamma *= self.provider.transform_gamma
                xpi_gamma *= self.provider.transform_gamma
            case Transform():
                bsigma_gamma *= self.provider.transform_gamma.t(r_mesh, polar_mesh)
                api_gamma *= self.provider.transform_gamma.t(r_mesh, polar_mesh)
                xpi_gamma *= self.provider.transform_gamma.t(r_mesh, polar_mesh)
            case None:
                pass
            case _:
                raise Exception(f"wrong type of transform potential {type(self.provider.transform_gamma)}")

        potential_ab = np.zeros_like(wave_full_init)
        potential_xpi = np.zeros_like(wave_full_init)

        for omega in range(-omega_max, omega_max + 1):
            centrifugal_potential = centrifugal(r_points, j_tot, omega, self.params.mass_u)
            centrifugal_potential = np.broadcast_to(np.expand_dims(centrifugal_potential, 1), (self.config.r_no, self.config.polar_no))
            
            potential_ab[:, :, omega_shift+omega] = potential + centrifugal_potential + complex(0, -0.5) * (bsigma_gamma + api_gamma)
            potential_xpi[:, :, omega_shift+omega] = complex(0, -0.5) * xpi_gamma

        potential_with_bsigma_prop = split.complex_n_dim_into_propagator(potential_ab.shape, potential_ab.flatten(), time_grid)

        loss_checker = split.LossChecker.new_with_saver("bsigma", self.config.frames, f"../data/{self.save_prefix}_bsigma", time_grid) \
            if self.config.animation else split.LossChecker("bsigma")
        potential_with_bsigma_prop.set_loss_checked(loss_checker)

        xpi_gamma_prop = split.complex_n_dim_into_propagator(potential_xpi.shape, potential_xpi.flatten(), time_grid)
        
        loss_checker = split.LossChecker.new_with_saver("xpi", self.config.frames, f"../data/{self.save_prefix}_xpi", time_grid) \
            if self.config.animation else split.LossChecker("xpi")
        xpi_gamma_prop.set_loss_checked(loss_checker)

        leak_control = split.LeakControl(split.LossChecker("leak control"))
        dumping_border = split.BorderDumping(1., 3., r_grid)

        angular_transformation = split.associated_legendre_transformations(polar_grid, omega_grid)

        shape, angular_kinetic_op = split.rotational_hamiltonian(r_grid, polar_grid, self.params.mass_u, self.params.rot_const)
        angular_op = np.broadcast_to(np.expand_dims(np.array(angular_kinetic_op).reshape(shape), 2), (self.config.r_no, self.config.polar_no, omega_no))

        angular_prop = split.n_dim_into_propagator(angular_op.shape, angular_op.flatten(), time_grid)

        j_grid = angular_transformation.transformed_grid()
        coriolis = split.NonDiagPropagator.get_coriolis(r_grid, j_grid, omega_grid, self.params.mass_u, j_tot, time_grid, step="half")

        fft_transformation = split.FFTTransformation(r_grid, "r momentum")

        kinetic_op = split.kinetic_hamiltonian(r_grid, self.params.mass_u, self.params.energy_kelvin)
        kinetic_prop = split.one_dim_into_propagator(kinetic_op, r_grid, time_grid, step = "full")

        ########## operation stack creation
        operation_stack = split.OperationStack()

        if self.config.im_time:
            leak_control.add_operation(operation_stack)

        potential_with_bsigma_prop.add_operation(operation_stack)
        xpi_gamma_prop.add_operation(operation_stack)

        if AnimationConfig.Distance in self.config.animation:
            distance_saver = split.StateSaver(f"{DATA_PATH}/{self.save_prefix}_distance_animation", time_grid, r_grid, self.config.frames)
            distance_saver.add_operation(operation_stack)

        if AnimationConfig.Polar in self.config.animation:
            polar_saver = split.StateSaver(f"{DATA_PATH}/{self.save_prefix}_polar_animation", time_grid, polar_grid, self.config.frames)
            polar_saver.add_operation(operation_stack)

        if AnimationConfig.AngProjection in self.config.animation:
            omega_saver = split.StateSaver(f"{DATA_PATH}/{self.save_prefix}_omega_animation", time_grid, omega_grid, self.config.frames)
            omega_saver.add_operation(operation_stack)

        if not self.config.im_time:
            dumping_border.add_operation(operation_stack)
            leak_control.add_operation(operation_stack)

        angular_transformation.add_operation(operation_stack, True)
        if AnimationConfig.Angular in self.config.animation:
            angular_grid = angular_transformation.transformed_grid()
            wave_legendre_saver = split.StateSaver(f"{DATA_PATH}/{self.save_prefix}_angular_animation", time_grid, angular_grid, self.config.frames)
            wave_legendre_saver.add_operation(operation_stack)

        angular_prop.add_operation(operation_stack)
        coriolis.add_operation(operation_stack)

        fft_transformation.add_operation(operation_stack, True)
        if AnimationConfig.Momentum in self.config.animation: 
            fft_grid = fft_transformation.transformed_grid()
            wave_legendre_saver = split.StateSaver(f"{DATA_PATH}/{self.save_prefix}_momentum_animation", time_grid, fft_grid, self.config.frames)
            wave_legendre_saver.add_operation(operation_stack)

        kinetic_prop.add_operation(operation_stack)
        
        ######### propagation creation

        propagation = split.Propagation()
        propagation.set_wave_function(wave_function)
        propagation.set_time_grid(time_grid)
        propagation.set_operation_stack(operation_stack)

        return propagation
