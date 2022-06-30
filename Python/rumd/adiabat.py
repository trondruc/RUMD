#!/usr/bin/python3
# -*- coding: utf-8 -*-
r""" Generate a configurational adiabat

This module use RUMD and the fourth order Runge-Kutta (RK4) integration method
(as a default) to generate a configuration adiabat.
Below example is an example of generating an adiabat of a Lennard-Jones model.

Example
-------

Computing adiabat of the Lennard-Jones model::

    # Setup a RUMD simulation object
    import rumd
    from rumd.Simulation import Simulation
    sim = Simulation('start.xyz.gz')
    itg = rumd.IntegratorNVT(targetTemperature=1.0, timeStep=0.005)
    sim.SetIntegrator(itg)
    pot = rumd.Pot_LJ_12_6(cutoff_method=rumd.ShiftedForce)
    pot.SetParams(i=0, j=0, Epsilon=1.0, Sigma=1.0, Rcut=2.5)
    sim.AddPotential(pot)
    sim.SetOutputScheduling("energies", "linear", interval=8, precision=16)

    # Compute configurational adiabat
    from numpy import arange
    from rumd.adiabat import compute_adiabat
    sim.adiabat_density = arange(1.0, 1.3, 0.05)
    sim.adiabat_temperature = 2.0
    sim.adiabat_set_temperature = itg.SetTargetTemperature
    sim.adiabat_num_iter = 2**18
    sim.adiabat_dense_input = arange(0.95, 1.35, 0.001)
    compute_adiabat(sim)

Usage details
-------------

The function for computing configurational is named compute_adiabat(sim).
The input is a RUMD simulation object with some extra attributes. These are:

    - adiabat_density [**mandatory**]: A list or a float.
    - adiabat_temperature [**mandatory**]: A list or a float.
    - adiabat_set_temperature [**mandatory**]: Function to set temperature that expects a float as input.
    - adiabat_num_iter [**mandatory**]: Steps for the production run.
    - adiabat_num_iter_equbriliation [optional]: Steps for equbriliation run (default: num_iter).
    - adiabat_stepper [optional]: Stepper function (default: stepper_rk4)
    - adiabat_dense_input [optional]: A list (default: None)
    - adiabat_save_trajectories_on_adiabat [optional]: boolean (default: True)
    - adiabat_save_trajectories_not_on_adiabat [optional]: boolean (default: False)

The numerical integration is done in steps of `h` along
either `t=log(rho)` estimating `y=log(T)` or
along `t=log(T)` estimating `y=log(rho)`
where `rho` is density and `T` is temperature.
The requied slopes, `f(t,y)=dy/dt`, for the numerical integration is computed
in from virial-energy fluctuations in a RUMD simulation.

If sim object the contains a list of densities then the integration
is done along densities, and vise versa if it contains a list of temperatures.
Data of the computed adiabat is saved to the plain text file adiabat.dat.
It can be read using the read_adiabat() function.

A dense output using a cubic spline is computed if a dense_input is attributed
to the sim object. The data of the dense output is save to dense_adiabat.dat.

If adiabat_save_trajectories_on_adiabat=True then directories
with simulations on the adiabat is keept.
All trajectories are saved if the adiabat_save_trajectories_not_on_adiabat=True
is also set to True. In some cases this avoids reruns for fast execution,
but may take up diskspace with uninteresting data.
By default, the trajectories on the adiabat are saved, but not trajectories
not on the adiabat.
The directories are named f'{density:.9f}_{temperature:.9f}'.



List of methods (stepper functions)
-----------------------------------

The below methods are implimented in the stepper_method_name functions.
See documentation for the stepper functions for more details on the specific
methods.

   * stepper_euler: Forward Euler integration (1st order).
   * stepper_euler_err: Forward Euler. Error estimate using Huen's prediction.
   * stepper_midpoint: Explicit midpoint method aka modified Euler (2nd order).
   * stepper_heun: Heun's method (2nd order)
   * stepper_ralston: Ralston's method (2nd order)
   * stepper_bogacki_shampine: Bogacki & Shampine's method (3rd order).
   * stepper_bogacki_shampine_4_order: Using the 4th order prediction of the above.
   * stepper_rk4: The standard Runge-Kutta method (RK4) (4th order).  **DEFAULT METHOD**
   * stepper_rk4_38rule: RK4 with the 3/8 rule (4th order)
   * stepper_rk4_double_step: RK4 with double step (5th order).
   * stepper_fehlberg (*in testing*): Fehlberg's method aka RKF45 (5th order)
   * stepper_heun_predictor_corrector (*in testing*): Predictor corrector alg.

A stepper function can be attaced to the sim object::

    from adiabat import stepper_rk4
    sim.adiabat_stepper = stepper_rk4

Using the read_adiabat functions
--------------------------------
The read_adiabat() function can be used the plot information along the adiabat.
The following snipped show how plot information along the adiabat::

    from rumd.adiabat import read_adiabat
    
    # Print column names
    print('Column names in adiabat.dat: ',
          read_adiabat(return_column_names=True))
    print('Column name in dense_adiabat.dat: ',
          read_adiabat(filename='dense_adiabat.dat', return_column_names=True))

    # Plot result
    rho = read_adiabat('density')
    T = read_adiabat('temperature')
    gamma = read_adiabat('gamma')
    error_gamma = read_adiabat('error_gamma')

    rho_dense = read_adiabat('input', 'dense_adiabat.dat')
    T_dense = read_adiabat('output', 'dense_adiabat.dat')
    gamma_dense = read_adiabat('f', 'dense_adiabat.dat')
    curvature_dense = read_adiabat('g', 'dense_adiabat.dat')

    import matplotlib.pyplot as plt
    plt.figure('rhoT')
    plt.plot(rho, T, '.')
    plt.plot(rho_dense, T_dense, '-', label='Dense output')
    plt.xlabel(r'Density, $\rho$')
    plt.ylabel(r'Temperature, $T$')
    plt.legend()
    plt.savefig('rhoT.png')
    plt.show()

    plt.figure('gamma')
    from numpy import gradient, log
    plt.errorbar(rho, gamma, yerr=error_gamma, fmt='.')
    plt.plot(rho, gradient(log(T), log(rho)), 'x',
             label='Numerical difference')
    plt.plot(rho_dense, gamma_dense, '-')
    plt.xlabel(r'Density, $\rho$')
    plt.ylabel(r'Slope, $\gamma$')
    plt.legend()
    plt.savefig('gamma.png')
    plt.show()

    plt.figure('curvature')
    plt.plot(rho, gradient(gamma, log(rho)), 'x')
    plt.plot(rho_dense, curvature_dense, '-', label='Dense output')
    plt.xlabel(r'Density, $\rho$')
    plt.ylabel(r'Curvature, $d\gamma/d\ln\rho$')
    plt.legend()
    plt.savefig('curvature.png')
    plt.show()


Error analysis
--------------

Most of integration methods gives an error estimated (error_y) by computing
the difference from a prediction usually from a estimate of different order.
As an example, the secound order Heun method uses the (forward) Euler
prediction for an error estimate.
The RK4 method (default) uses an estimate using the k3 midtpoint slope.
The third order Bogacki_Shampine method uses a fourth order estimate, that
comes almost free (if all trajectory files are save to the disk).

There are two sources of error to the error on the value:

    1) Statistical error from the estimated slopes (i.e. gamma).
    2) Truncation of higher-order terms in the numerical integration.

The statistical error (1) can be estimated independently
as the error on the slope (gamma) times the step length.
The error on the slope is given as error_gamma and is computed by assuming that
the blocks of the RUMD simulations are independent.

The error can be reduced in three different ways:

    A) Making longer simulations for each slope integratoin (`num_iter`).
    B) Picking a smaller step-size 'h' along the integration variable.
    C) Using a higher-order method (RK4 is recomended).

The example below show an exampe of an error analysis in the last part::

    # Plot error analysis on each step
    from rumd.adiabat import read_adiabat
    from numpy import array, absolute
    plt.figure('Error analysis')
    rho = read_adiabat('density')
    gamma_error = read_adiabat('error_gamma')
    h = read_adiabat('h')
    statistical_error_on_y = array(gamma_error)*array(h)
    plt.plot(rho, statistical_error_on_y, label='Statistical error')
    error_y = read_adiabat('error_y')
    plt.plot(rho, absolute(error_y), label='Total error')
    plt.legend()
    plt.show()

    """


def get_index_last_completed(directory='./TrajectoryFiles/'):
    """ Return the number of compleeted blocks of a RUMD simulation """
    from os.path import join
    filename = join(directory, 'LastComplete_energies.txt')
    with open(filename, 'rt') as file:
        index_last_complete = int(file.readline().split()[0])
    return index_last_complete


def read_energies(energy_variable='pe', directory='./TrajectoryFiles/',
                  print_column_names=False):
    """ Returns a list of energies of the energies{i:04d}.dat.gz files  """
    from os.path import join
    from gzip import open as gzip_open

    have_printed_names = False
    enr_list = []  # List of energies in column named energy_variable
    index_last_complete = get_index_last_completed(directory)
    filename = join(directory, 'LastComplete_energies.txt')
    with open(filename, 'rt') as file:
        index_last_complete = int(file.readline().split()[0])
    # Read data in all the energy files
    for i in range(index_last_complete+1):
        filename = join(directory, f'energies{i:04d}.dat.gz')
        with gzip_open(filename, 'rt') as file:
            # Read the header of the file to get the column of interest
            header = file.readline().split()[1:]
            for section in header:
                variable_name = section.split('=')[0]
                values = section.split('=')[1]
                if variable_name == 'columns':
                    column_names = values.split(',')
                    if(print_column_names and not have_printed_names):
                        print(column_names)
                        have_printed_names = True
            column_of_interest = -1
            for index, name in enumerate(column_names):
                if name == energy_variable:
                    column_of_interest = index
            if column_of_interest == -1:
                print(f'Columns in energy file: {column_names}')
                print(f'Warning: column named \
                      {energy_variable} was not found.')
            # Read data in column of interest and append to list
            for line in file.read().splitlines():
                enr_list.append(float(line.split()[column_of_interest]))
    return enr_list


def slope(t,
          y,
          sim,
          is_on_adiabat=True):
    ''' Make a RUMD simulation and
    return the scaling exponent gamma and other thermodynamic data '''
    # import rumd
    from os import mkdir
    from os import chdir
    from os import getcwd
    from os.path import isdir
    from shutil import rmtree
    from math import sqrt
    from numpy import exp

    if sim.adiabat_integrate_along_density:
        rho = exp(float(t))
        T = exp(float(y))
    else:
        T = exp(float(t))
        rho = exp(float(y))

    V = sim.GetVolume()
    N = sim.GetNumberOfParticles()

    # Change to a new simulation directory, if it is not already done
    root_directory = getcwd()
    simulation_directory = f'{rho:.9f}_{T:.9f}'
    this_is_a_new_simulation = True
    if not isdir(simulation_directory):
        print(f'Perform simulation with rho={rho:.9f} and T={T:.9f}')
        mkdir(simulation_directory)
        chdir(simulation_directory)

        # Scale simulation box
        sim.ScaleSystem(pow(N/V/rho, 1./3))

        # Set target temperature of thermostat
        sim.adiabat_set_temperature(T)

        # Run simulation
        sim.Run(sim.adiabat_num_iter_equilibration, suppressAllOutput=True)
        sim.Run(sim.adiabat_num_iter)
        sim.sample.TerminateOutputManagers()
    else:
        print(f'Reuse simulation with rho={rho:.9f} and T={T:.9f}')
        chdir(simulation_directory)
        this_is_a_new_simulation = False

    from numpy import mean, cov, array, std
    pots = read_energies('pe')
    virials = read_energies('W')
    pot = mean(pots)
    virial = mean(virials)

    def virial_energy_analysis(energies, virials):
        covariance_matrix = cov(array([energies, virials]))
        duu = covariance_matrix[0, 0]
        dww = covariance_matrix[1, 1]
        duw = covariance_matrix[0, 1]
        corr_coef = duw / sqrt(duu*dww)
        gamma = duw / duu
        excess_heat_capacity = duu/T/T*N
        return gamma, corr_coef, excess_heat_capacity, duu, dww, duw

    gamma, corr_coef, excess_heat_capacity, duu, dww, duw =\
        virial_energy_analysis(pots, virials)

    # Error analysis on gamma
    number_of_blocks = get_index_last_completed()
    block_size = int(len(pots)/number_of_blocks)
    gammas, corr_coefs, excess_heat_capacities, uss_list, wss_list = \
        [], [], [], [], []
    for i in range(number_of_blocks):
        first_index = i*block_size
        last_index = (i+1)*block_size
        uss = pots[first_index:last_index]
        wss = virials[first_index:last_index]
        out = virial_energy_analysis(uss, wss)
        gammas.append(out[0])
        corr_coefs.append(out[1])
        excess_heat_capacities.append(out[2])
        uss_list.append(uss)
        wss_list.append(uss)
    error_gamma = std(gammas)/sqrt(number_of_blocks)
    error_corr_coef = std(corr_coefs)/sqrt(number_of_blocks)
    error_u = std(uss_list)/sqrt(number_of_blocks)
    error_w = std(wss_list)/sqrt(number_of_blocks)
    error_excess_heat_capacity =\
        std(excess_heat_capacities)/sqrt(number_of_blocks)

    # Move to root directory and clean up
    chdir(root_directory)
    save_trajectory = (is_on_adiabat and
                       sim.adiabat_save_trajectories_on_adiabat) or (
                           sim.adiabat_save_trajectories_not_on_adiabat)
    if this_is_a_new_simulation and not save_trajectory:
        rmtree(simulation_directory)

    if sim.adiabat_integrate_along_density:
        output_slope = gamma
    else:
        output_slope = 1.0/gamma

    return (output_slope, rho, T, gamma, corr_coef, pot, virial,
            excess_heat_capacity, duu, dww, duw,
            error_gamma, error_corr_coef,
            error_u, error_w, error_excess_heat_capacity)


def stepper_rk4(t, y, h, sim,
                return_extra_comments=False):
    """ Fourth order Runge-Kutta integrator (the default integrator) """
    if return_extra_comments:
        return ',k1,k2,k3,k4,error_y'
    simulation_data = slope(t, y, sim, True)
    k1 = simulation_data[0]
    simulation_data2 = slope(t+h/2.0, y+h*k1/2.0, sim, False)
    k2 = simulation_data2[0]
    simulation_data3 = slope(t+h/2.0, y+h*k2/2.0, sim, False)
    k3 = simulation_data3[0]
    simulation_data4 = slope(t+h, y+h*k3, sim, False)
    k4 = simulation_data4[0]
    y = y+h*(k1+2.0*k2+2.0*k3+k4)/6.0
    error_y = h*k3-h*(k1+2.0*k2+2.0*k3+k4)/6.0
    return y, (*simulation_data, k1, k2, k3, k4, error_y)


def stepper_euler(t, y, h, sim,
                  return_extra_comments=False):
    """ Euler integration. """
    if return_extra_comments:
        return ',k1'
    simulation_data = slope(t, y, sim, True)
    k1 = simulation_data[0]
    y = y+h*k1
    return y, (*simulation_data, k1)


def stepper_euler_err(t, y, h, sim,
                      return_extra_comments=False):
    """ Forward Euler integration with an error estimate using the
    secound order Heun's estimate. Produce the same estimate as
    the `Euler_stepper()` function. It is almost as fast, but have needes
    to make one additional simulation for a given adiabat. """
    if return_extra_comments:
        return ',k1,error_y'
    simulation_data = slope(t, y, sim, True)
    k1 = simulation_data[0]
    simulation_data1 = slope(t + h, y+h*k1, sim, True)
    k2 = simulation_data1[0]
    y = y+h*k1
    y2 = y+h*(k1+k2)/2
    error_y = y2 - y
    return y, (*simulation_data, k1, error_y)


def stepper_midpoint(t, y, h, sim,
                     return_extra_comments=False):
    """ The (explicit) midpoint method (2nd order). """
    if return_extra_comments:
        return ',k1,k2,error_y'
    simulation_data = slope(t, y, sim, True)
    k1 = simulation_data[0]
    simulation_data2 = slope(t+h/2, y+h*k1/2, sim, False)
    k2 = simulation_data2[0]
    y = y+h*k2
    error_y = h*k1-h*k2  # Euler method (1st order), minus Midpoint (2nd order)
    return y, (*simulation_data, k1, k2, error_y)


def stepper_heun(t, y, h, sim,
                 return_extra_comments=False):
    """ Heun's method (2nd order).
    $$ y_1 = y_0 + h(k_1+k_2)/2 $$
    where
    $$ k_1 = f(t_0, y_0) $$
    $$ k_2 = f(t_0+h, y_0+hk_1) $$
    """
    if return_extra_comments:
        return ',k1,k2,error_y'
    simulation_data = slope(t, y, sim, True)
    k1 = simulation_data[0]
    simulation_data2 = slope(t+h, y+h*k1, sim, False)
    k2 = simulation_data2[0]
    y = y+h*0.5*(k1+k2)
    error_y = h*k1-h*0.5*(k1+k2)  # Euler method minus Midpoint
    return y, (*simulation_data, k1, k2, error_y)


def stepper_ralston(t, y, h, sim,
                    return_extra_comments=False):
    """ Raslson's method (2nd order) """
    if return_extra_comments:
        return ',k1,k2,error_y'
    simulation_data = slope(t, y, sim, True)
    k1 = simulation_data[0]
    simulation_data2 = slope(t+2.0*h/3.0, y+h*2.0*k1/3.0, sim, True)
    k2 = simulation_data2[0]
    y = y+0.25*h*k1+0.75*h*k2
    error_y = h*k1-0.25*h*k1+0.75*h*k2
    return y, (*simulation_data, k1, k2, error_y)


def stepper_bogacki_shampine(t, y, h, sim,
                             return_extra_comments=False):
    """ Bogacki–Shampine's third order method with
    with an embedded fourth order estimate use for error analysis. """
    if return_extra_comments:
        return ',k1,k2,k3,k4,error_y'
    simulation_data = slope(t, y, sim, True)
    k1 = simulation_data[0]
    simulation_data2 = slope(t+h/2.0, y+h*k1/2.0, sim, False)
    k2 = simulation_data2[0]
    simulation_data3 = slope(t+3.0*h/4.0, y+3.0*h*k2/4.0, sim, False)
    k3 = simulation_data3[0]
    y3 = y + h*(2*k1+3*k2+4*k3)/9
    simulation_data4 = slope(t+h, y+h*(2*k1+3*k2+4*k3)/9, sim, True)
    k4 = simulation_data4[0]
    y4 = y+h*(7*k1+6*k2+8*k3+3*k4)/24.0
    error_y = y3-y4
    return y3, (*simulation_data, k1, k2, k3, k4, error_y)


def stepper_bogacki_shampine_4_order(t, y, h, sim,
                                     return_extra_comments=False):
    """ Bogacki–Shampine's fourth order method with
    with an embedded third order estimate used for error analysis. """
    if return_extra_comments:
        return ',k1,k2,k3,k4,error_y'
    simulation_data = slope(t, y, sim, True)
    k1 = simulation_data[0]
    simulation_data2 = slope(t+h/2.0, y+h*k1/2.0, sim, False)
    k2 = simulation_data2[0]
    simulation_data3 = slope(t+3.0*h/4.0, y+3.0*h*k2/4.0, sim, False)
    k3 = simulation_data3[0]
    y3 = y + h*(2*k1+3*k2+4*k3)/9
    simulation_data4 = slope(t+h, y+h*(2*k1+3*k2+4*k3)/9, sim, False)
    k4 = simulation_data4[0]
    y4 = y+h*(7*k1+6*k2+8*k3+3*k4)/24.0
    error_y = y3-y4
    return y3, (*simulation_data, k1, k2, k3, k4, error_y)


def stepper_rk4_38rule(t, y, h, sim,
                       return_extra_comments=False):
    """ Fourth order Runge-Kutta integrator with the 3/8 rule."""
    if return_extra_comments:
        return ',k1,k2,k3,k4'
    simulation_data = slope(t, y, sim, True)
    k1 = simulation_data[0]
    simulation_data2 = slope(t+h/3.0, y+h*k1/3.0, sim, False)
    k2 = simulation_data2[0]
    simulation_data3 = slope(t+2.0*h/3.0, y-h*k1/3.0+h*k2, sim, False)
    k3 = simulation_data3[0]
    simulation_data4 = slope(t+h, y+h*(k1-k2+k3), sim, False)
    k4 = simulation_data4[0]
    y = y+h*(k1+3.0*k2+3.0*k3+k4)/8.0
    return y, (*simulation_data, k1, k2, k3, k4)


def stepper_rk4_double_step(t, y, h, sim,
                            return_extra_comments=False):
    """ Double stepping of the fourth order Runge-Kutta integrator RK4. """
    if return_extra_comments:
        return ',k1,k2,k3,k4,kA2,kA3,kA4,kB1,kB2,kB3,kB4,error_y'
    simulation_data = slope(t, y, sim, True)
    k1 = simulation_data[0]
    simulation_data2 = slope(t+h/2.0, y+h*k1/2.0, sim, False)
    k2 = simulation_data2[0]
    simulation_data3 = slope(t+h/2.0, y+h*k2/2.0, sim, False)
    k3 = simulation_data3[0]
    simulation_data4 = slope(t+h, y+h*k3, sim, False)
    k4 = simulation_data4[0]
    y1 = y+h*(k1+2.0*k2+2.0*k3+k4)/6.0
    # The first half step, A
    hh = h/2
    simulation_data5 = slope(t+hh/2.0, y+hh*k1/2.0, sim, False)
    kA2 = simulation_data5[0]
    simulation_data6 = slope(t+hh/2.0, y+hh*kA2/2.0, sim, False)
    kA3 = simulation_data6[0]
    simulation_data7 = slope(t+hh, y+hh*kA3, sim, False)
    kA4 = simulation_data7[0]
    yy = y+hh*(k1+2.0*kA2+2.0*kA3+kA4)/6.0
    # The secound half part, B
    tt = t + hh
    simulation_data8 = slope(tt, yy, sim, False)
    kB1 = simulation_data8[0]
    simulation_data9 = slope(tt+hh/2.0, yy+hh*kB1/2.0, sim, False)
    kB2 = simulation_data9[0]
    simulation_data10 = slope(tt+hh/2.0, yy+hh*kB2/2.0, sim, False)
    kB3 = simulation_data10[0]
    simulation_data11 = slope(tt+hh, yy+hh*kB3, sim, False)
    kB4 = simulation_data11[0]
    y = yy + hh*(kB1+2.0*kB2+2.0*kB3+kB4)/6.0
    error_y = y1-y
    return y, (*simulation_data, k1, k2, k3, k4,
               kA2, kA3, kA4, kB1, kB2, kB3, kB4, error_y)


def stepper_fehlberg(t, y, h, sim,
                     return_extra_comments=False):
    """ This stepper function is in testing mode """
    if return_extra_comments:
        return ',k1,k2,k3,k4,k5,k6,error_y'
    simulation_data = slope(t, y, sim, True)
    k1 = simulation_data[0]
    simulation_data2 = slope(t+h/4.0, y+h*k1/4.0, sim, False)
    k2 = simulation_data2[0]
    simulation_data3 = slope(t+3.0*h/8.0,
                             y+h*(3*k1+9*k2)/32, sim, False)
    k3 = simulation_data3[0]
    simulation_data4 = slope(t+12*h/13,
                             y+h*(1932*k1-7200*k2+7296*k3)/2197,
                             sim, False)
    k4 = simulation_data4[0]
    simulation_data5 = slope(t+h,
                             y+h*(439*k1/216-8*k2+3680*k3/513-845*k4/4104),
                             sim, False)
    k5 = simulation_data5[0]
    step_in_y = h*(-8*k1/27+2*k2-3544*k3/2565+1859*k4/4104-11*k5/40)
    simulation_data6 = slope(t+h/2,
                             y+step_in_y,
                             sim, False)
    k6 = simulation_data6[0]
    y = y+h*(16*k1/135+6656*k3/12825+28561*k4/56430-9*k5/50+2*k6/55)
    y2 = y+h*(25*k1/216+1408*k3/2565+2197*k4/4104-k5/5)
    error_y = y2-y
    return y, (*simulation_data, k1, k2, k3, k4, k5, k6, error_y)


def stepper_heun_predictor_corrector(t, y, h, sim,
                                     return_extra_comments=False):
    """ Heun's predictor–corrector method (2nd order).
    The sim object needs sim.adiabat_tolerence=1e-5 and
    sim.adiabat_min_max_evaluations=(2,10) attribute."""

    if return_extra_comments:
        return ',k1,k2,number_of_evaluations,change_of_y,error_y'
    simulation_data = slope(t, y, sim, True)
    k1 = simulation_data[0]
    y_old = y + h*k1
    change_of_y = sim.adiabat_tolerence + 1.0
    number_of_evaluations = 0
    while number_of_evaluations < sim.adiabat_min_max_evaluations[0] or \
            (change_of_y**2 > sim.adiabat_tolerence**2
             and number_of_evaluations < sim.adiabat_min_max_evaluations[1]):
        simulation_data2 = slope(t+h, y_old, sim, False)
        k2 = simulation_data2[0]
        y_new = y+h*(k1+k2)/2
        change_of_y = y_new-y_old
        y_old = y_new
        number_of_evaluations += 1
        from numpy import exp
        print(f'Evaluation {number_of_evaluations} at t = {t}:')
        print(f'            exp(t) = {exp(t)}')
        print(f'            y_new  = {y_new}')
        print(f'        exp(y_new) = {exp(y_new)}')
        print(f'       change_of_y = {change_of_y}')
        error_gamma = simulation_data2[11]
        statistical_error = error_gamma*h
        print(f' statistical_error = {statistical_error}')
        if statistical_error > sim.adiabat_tolerence:
            print('Warning: statistical_error > tolerence ')
            print(f'     {statistical_error} > {sim.tolerence}.')
    # error_y = ("Forward error"+"Backward error")/2
    error_y = h*(k1-k2)/2
    y = y_new
    return y, (*simulation_data, k1, k2,
               number_of_evaluations, change_of_y, error_y)


def driver(ts, y0, sim):
    """ Method to drive along an adiabat
    by calling a stepper function mulitple times """

    data_collection = []
    ys = [y0]
    extra_comments = sim.adiabat_stepper(0, 0, 0, sim,
                                         return_extra_comments=True)
    print('# \
columns=t,y,h,y_next,f,density,temperature,gamma,correlation_coefficient,u,w,\
excess_heat_capacity,duu,dww,duw,error_gamma,error_correlation_coefficient,\
error_u,error_w,error_excess_heat_capacity'
          + extra_comments, file=open('adiabat.dat', 'w'))
    for i, t in enumerate(ts):
        if i < len(ts)-1:
            h = ts[i+1]-t
            y, data = sim.adiabat_stepper(t, ys[-1], h, sim)
            print(t, ys[-1], h, y, *data, file=open('adiabat.dat', 'a'))
            data_collection.append([t, ys[-1], h, y, *data])
            ys.append(y)
    return data_collection


def compute_adiabat(sim):
    """ Compute adiabat and dense output """
    from numpy import log

    # Confirm that sim object has mandatory attributes
    if not (hasattr(sim, 'adiabat_density') and
            hasattr(sim, 'adiabat_temperature') and
            hasattr(sim, 'adiabat_set_temperature') and
            hasattr(sim, 'adiabat_num_iter')):
        print('Warning: The sim objects is missing attribute(s).')

    # Detect integration variable
    try:
        iter(sim.adiabat_density)
        adiabat_integrate_along_density = True
    except TypeError:
        adiabat_integrate_along_density = False
    if adiabat_integrate_along_density:
        ts = log(sim.adiabat_density)
        y0 = log(sim.adiabat_temperature)
    else:
        ts = log(sim.adiabat_temperature)
        y0 = log(sim.adiabat_density)
    sim.adiabat_integrate_along_density = adiabat_integrate_along_density

    # Set default values if not given on sim object
    if not hasattr(sim, 'adiabat_stepper'):
        sim.adiabat_stepper = stepper_rk4
    if not hasattr(sim, 'adiabat_num_iter_equilibration'):
        sim.adiabat_num_iter_equilibration = sim.adiabat_num_iter
    if not hasattr(sim, 'adiabat_save_trajectories_on_adiabat'):
        sim.adiabat_save_trajectories_on_adiabat = True
    if not hasattr(sim, 'adiabat_save_trajectories_not_on_adiabat'):
        sim.adiabat_save_trajectories_not_on_adiabat = False
    data = driver(ts, y0, sim)

    if not hasattr(sim, 'adiabat_dense_input'):
        sim.adiabat_dense_input = None
    if sim.adiabat_dense_input is not None:
        dense_adiabat = compute_dense_adiabat(sim.adiabat_dense_input)
        return data, dense_adiabat
    return data


def read_adiabat(variable='y', filename='adiabat.dat',
                 return_column_names=False, dtype=float):
    """ Returns a list of values from the adiabat.dat file  """

    output = []  # List of energies in column named energy_variable

    with open(filename, 'r') as file:
        header = file.readline().split()[1:]
        for section in header:
            variable_name = section.split('=')[0]
            values = section.split('=')[1]
            if variable_name == 'columns':
                column_names = values.split(',')
                if return_column_names:
                    return column_names
        column_of_interest = -1
        for index, name in enumerate(column_names):
            if name == variable:
                column_of_interest = index
        if column_of_interest == -1:
            print(f'Warning: variable {variable} not found in {filename}.')
            return None
        for line in file.read().splitlines():
            output.append(dtype(line.split()[column_of_interest]))
        return output


def compute_dense_adiabat(dense_input):
    """ Compute dense adiabat by a cubic spline """
    from numpy import log, exp
    print('# columns=input,output,t,y,f,g,\
i0,i1,Dt,Dy,tau,a,b,c,is_interpolation',
          file=open('dense_adiabat.dat', 'w+'))
    dense_adiabat = []
    t = read_adiabat('t')
    y = read_adiabat('y')
    f = read_adiabat('f')
    for tt in log(dense_input):
        # Find the inverval for this dense input tt
        i0 = 0
        interpolation = 0
        if tt < min(t):
            i0 = 0
        elif tt > max(t):
            i0 = len(t)-2
        else:
            interpolation = 1
            best_distance = (tt-t[0])**2+(tt-t[1])**2
            for i, _ in enumerate(t):
                if i < len(t)-1:
                    distance = (tt-t[i])**2+(tt-t[i+1])**2
                    if distance < best_distance:
                        best_distance = distance
                        i0 = i
        # Fit a third degree polynomial (cubic spline)
        i1 = i0+1
        Dt = t[i1]-t[i0]
        Dy = y[i1]-y[i0]
        tau = (tt-t[i0])/Dt
        f0_tilde = f[i0]*Dt/Dy
        f1_tilde = f[i1]*Dt/Dy
        a = f0_tilde + f1_tilde - 2.0
        b = 3.0 - 2.0*f0_tilde - f1_tilde
        c = f0_tilde
        ytau = y[i0] + (a*pow(tau, 3) + b*pow(tau, 2) + c*tau)*Dy
        ftau = (3.0*a*pow(tau, 2) + 2.0*b*tau + c)*Dy/Dt
        gtau = (6.0*a*tau+2.0*b)*Dy/Dt/Dt
        print(exp(tt), exp(ytau), tt, ytau, ftau, gtau, i0, i1, Dt, Dy,
              tau, a, b, c, interpolation,
              file=open('dense_adiabat.dat', 'a'))
        dense_adiabat.append([exp(tt), exp(ytau), tt, ytau, ftau, gtau,
                              i0, i1, Dt, Dy, tau, a, b, c, interpolation])
    return dense_adiabat


def example_lennard_jones():
    ''' Example of running this module by computinh
        a configurational adiabat of the Lennard-Jones model '''

    # Setup a RUMD simulation object
    import rumd
    from rumd.Simulation import Simulation
    sim = Simulation('start.xyz.gz')
    itg = rumd.IntegratorNVT(targetTemperature=1.0, timeStep=0.005)
    sim.SetIntegrator(itg)
    pot = rumd.Pot_LJ_12_6(cutoff_method=rumd.ShiftedForce)
    pot.SetParams(i=0, j=0, Epsilon=1.0, Sigma=1.0, Rcut=2.5)
    sim.AddPotential(pot)
    sim.SetOutputScheduling("energies", "linear", interval=8, precision=16)

    # Dress the sim object
    from numpy import arange
    sim.adiabat_set_temperature = itg.SetTargetTemperature
    sim.adiabat_num_iter = 2**18
    sim.adiabat_num_iter_equilibration = 2**16
    sim.adiabat_density = arange(1.0, 1.3, 0.05)
    sim.adiabat_temperature = 2.0
    sim.adiabat_dense_input = arange(0.95, 1.3, 0.001)
    sim.adiabat_stepper = stepper_rk4
    sim.adiabat_save_trajectories_on_adiabat = True
    sim.adiabat_save_trajectories_not_on_adiabat = True
    compute_adiabat(sim)

    # Print column names
    print('Column names in adiabat.dat: ',
          read_adiabat(return_column_names=True))
    print('Column name in dense_adiabat.dat: ',
          read_adiabat(filename='dense_adiabat.dat', return_column_names=True))

    # Plot result
    rho = read_adiabat('density')
    T = read_adiabat('temperature')
    gamma = read_adiabat('gamma')
    error_gamma = read_adiabat('error_gamma')

    rho_dense = read_adiabat('input', 'dense_adiabat.dat')
    T_dense = read_adiabat('output', 'dense_adiabat.dat')
    gamma_dense = read_adiabat('f', 'dense_adiabat.dat')
    curvature_dense = read_adiabat('g', 'dense_adiabat.dat')

    import matplotlib.pyplot as plt
    plt.figure('rhoT')
    plt.plot(rho, T, '.')
    plt.plot(rho_dense, T_dense, '-', label='Dense output')
    plt.xlabel(r'Density, $\rho$')
    plt.ylabel(r'Temperature, $T$')
    plt.legend()
    plt.savefig('rhoT.png')
    plt.show()

    plt.figure('gamma')
    from numpy import gradient
    from numpy import log
    plt.errorbar(rho, gamma, yerr=error_gamma, fmt='.')
    plt.plot(rho, gradient(log(T), log(rho)), 'x',
             label='Numerical difference')
    plt.plot(rho_dense, gamma_dense, '-')
    plt.xlabel(r'Density, $\rho$')
    plt.ylabel(r'Slope, $\gamma$')
    plt.legend()
    plt.savefig('gamma.png')
    plt.show()

    plt.figure('curvature')
    plt.plot(rho, gradient(gamma, log(rho)), 'x')
    plt.plot(rho_dense, curvature_dense, '-', label='Dense output')
    plt.xlabel(r'Density, $\rho$')
    plt.ylabel(r'Curvature, $d\gamma/d\ln\rho$')
    plt.legend()
    plt.savefig('curvature.png')
    plt.show()

    # Plot error analysis on each step
    plt.figure('Error analysis')
    from numpy import array
    from numpy import absolute
    gamma_error = read_adiabat('error_gamma')
    h = read_adiabat('h')
    statistical_error_on_y = array(gamma_error)*array(h)
    plt.plot(rho, statistical_error_on_y,
             label=r'Statistical error on $y$ from slope ($\gamma$)')
    error_y = read_adiabat('error_y')
    if error_y is not None:
        plt.plot(rho, absolute(error_y), label=r'Error estimate on $y$')
    plt.xlabel(r'Density, $\rho$')
    plt.xlabel(r'Error on $y=\log($T$)$')
    plt.legend()
    plt.savefig('error_analysis.png')
    plt.show()


if __name__ == '__main__':
    example_lennard_jones()
