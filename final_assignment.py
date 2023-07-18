# -*- coding: utf-8 -*-
"""
PHYS20161 Z0 Boson Assignment

This code will find the partial width, mass and lifetime of a Z0 boson, with
their respective uncertainties, by fitting data filtered from two different
experiments to a Breit-Wigner expression for the cross section of
electron-positron collisions resulting in electron-positron products.

Tomas Welsh 15/12/2021
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin
import scipy.constants as pc

GAMMA_EE = 83.91e-3 #in GeV

FILE_NAME_1 = 'z_boson_data_1.csv'
FILE_NAME_2 = 'z_boson_data_2.csv'


def read_data(file):
    """
    Reads in data given a file name and filters out non-numerical values and
    numerical values less than or equal to zero as well as checks data is in
    three columns. It also removes outliers that are anomalies to a general
    set of 1D data

    Parameters
    ----------
    file_name : string

    Returns
    -------
    data : np.array of floats
        Data should be in format [x, y, uncertainty on y]
    """
    try:
        data = np.genfromtxt(file, delimiter=',', comments='%')

    except IOError:
        print("Unfortunately, the file inputted could not be found.")
        sys.exit()

    try:
        for line in data:
            np.vstack((np.empty((0, 3)), line))
    except ValueError:
        print("Please input a file with 3 columns of the form: x, y, "
              "uncertainty on y.")
        sys.exit()

    indicies = np.unique(np.where(np.isnan(data))[0])
    data = np.delete(data, indicies, 0)
    indicies = np.unique(np.where(data <= 0)[0])
    data = np.delete(data, indicies, 0)

    mean = np.mean(data[:,1])
    standard_deviation = np.std(data[:,1])

    indicies = np.unique(np.where(np.abs(data[:,1] - mean) > 5 *
                                  standard_deviation)[0])
    data = np.delete(data, indicies, 0)

    return data


def bw_function(energy, mass, width):
    """
    Breit-Wigner expression to be fitted

    Parameters
    ----------
    energy : float in units of GeV

    mass : float in units of GeV/c^2

    width : float in units of GeV

    Returns
    -------
    cross_section : float in units of nano-barns

    """
    cross_section = (12 * np.pi * energy**2 * GAMMA_EE**2) / (mass**2 *
    ((energy**2 - mass**2)** 2 + (mass * width)**2)) * 0.3894e6

    return cross_section


def bw_inverse_function(mass, energy, cross_section):
    """
    A rearrangement of the Breit-Wigner expression that instead gives the
    width. The function may output nans due to square-rooting a negative value

    Parameters
    ----------
    energy : float in units of GeV

    mass : float in units of GeV/c^2

    cross_section : float in units of nano-barns

    Returns
    -------
    width : float in units of GeV

    """
    width_squared = ((12 * np.pi * energy**2 * GAMMA_EE**2 * 0.3894e6 -
             cross_section * mass**2 * (energy**2 - mass**2)**2) /
            (cross_section * mass**4))

    adjusted_width_sq = np.where(width_squared >= 0, width_squared, np.nan)
    #above is useful when applying function in starting point function as
    #mass is in an array
    width = np.sqrt(adjusted_width_sq)

    return width


def chi_square(observation, observation_uncertainty, prediction):
    """
    Returns the chi squared

    Parameters
    ----------
    observation : np.array of floats

    observation_uncertainty : np.array of floats

    prediction : np.array of floats

    Returns
    -------
    float

    """
    return np.sum((observation - prediction)**2 / observation_uncertainty**2)


def chi_square_array(data, mesh_1, mesh_2):
    """
    Return a np.array of chi-squared values that match indicies with two
    meshes of respective values

    Parameters
    ----------
    data : np.array of floats
        Data should be in format [x, y, uncertainty on y]
    mesh_1 : np.array of floats

    mesh_2 : np.array of floats

    Returns
    -------
    chi_sqaure_array : np.array of floats

    """
    chi_squares_array = np.empty((0, np.shape(mesh_1)[1]))

    for i in range(np.shape(mesh_1)[0]):
        temp = np.zeros(np.shape(mesh_1)[1])

        for j in range(np.shape(mesh_1)[1]):
            temp[j] = chi_square(data[:, 1], data[:, 2],
                       bw_function(data[:, 0], mesh_1[i,j], mesh_2[i,j]))

        chi_squares_array = np.vstack((chi_squares_array, temp))

    return chi_squares_array


def mesh_arrays(x_array, y_array):
    """
    Returns two meshed arrays of size len(y_array) by len(x_array)

    Parameters
    ----------
    x_array : np.array of floats

    y_array : np.array of floats

    Returns
    -------
    x_array_mesh : np.array of floats

    y_array_mesh : np.array of floats

    """
    x_array_mesh = np.empty((0, len(x_array)))

    for _ in enumerate(y_array):
        x_array_mesh = np.vstack((x_array_mesh, x_array))

    y_array_mesh = np.empty((0, len(y_array)))

    for _ in enumerate(x_array):
        y_array_mesh = np.vstack((y_array_mesh, y_array))

    y_array_mesh = np.transpose(y_array_mesh)

    return x_array_mesh, y_array_mesh


def shared_values(array_1, array_2):
    """
    Finds the shared values between two arrays

    Parameters
    ----------
    array_1 : array

    array_2 : array

    Returns
    -------
    same_values : array

    """
    same_values = np.append(array_1, array_2)
    unique_values, count = np.unique(same_values, return_counts=True)
    same_values = unique_values[count > 1]

    return same_values


def estimations(data):
    """
    Finds two mean values of two columns in a data set

    Parameters
    ----------
    data : np.array of floats
        Data must have at least two columns

    Returns
    -------
    Mean values in arrays

    """
    energy_1 = np.mean(data[0:np.shape(data)[0] // 2, 0])
    energy_2 = np.mean(data[np.shape(data)[0] // 2:np.shape(data)[0], 0])

    cross_section_1 = np.mean(data[0:np.shape(data)[0] // 2, 1])
    cross_section_2 = np.mean(data[np.shape(data)[0] // 2:
                                   np.shape(data)[0], 1])

    return [energy_1, energy_2], [cross_section_1, cross_section_2]


def starting_points(energies, cross_sections, mass_range):
    """
    Function that solves two simultaneous equations graphically for two
    unknowns with algebraic expressions that can give nan values

    Parameters
    ----------
    energies : np.array of floats
        Data should be in format [energy_1, energy_2]

    energies : np.array of floats
        Data should be in format [cross_section_1, cross_section_2]

    mass_range : np.array of floats
        Data that the function is applied to

    Returns
    -------
    intersection : np.array of floats
        Data should be in the form [unknown_1_value, unknown_2_value]

    data_1 : np.array of floats
        Data should be in format [mass, width] for all values that do not give
        a nan

    data_2 : np.array of floats
        Data should be in format [mass, width] for all values that do not give
        a nan
    """
    estimate_1 = bw_inverse_function(mass_range, energies[0],
                                     cross_sections[0])
    indicies_1 = np.where(estimate_1 >= 0)[0]
    original_estimate_1 = estimate_1
    estimate_1 = estimate_1[indicies_1]

    estimate_2 = bw_inverse_function(mass_range, energies[1],
                                     cross_sections[1])
    indicies_2 = np.where(estimate_2 >= 0)[0]
    original_estimate_2 = estimate_2
    estimate_2 = estimate_2[indicies_2]

    same_indicies = shared_values(indicies_1, indicies_2)

    intersection_index = \
    np.argwhere(np.diff(np.sign(original_estimate_1[same_indicies] -
    original_estimate_2[same_indicies]))).flatten()

    intersection = [mass_range[same_indicies[intersection_index]],
                (original_estimate_1[same_indicies[intersection_index]] +
                original_estimate_2[same_indicies[intersection_index]]) / 2]

    data_1 = np.hstack(((mass_range[indicies_1][:,None]),
                        estimate_1[:,None]))

    data_2 = np.hstack((((mass_range[indicies_2])[:,None]),
                        estimate_2[:,None]))

    return intersection, data_1, data_2

def find_minimums(data, initial_points, sigma_lim=3, removal_iterations=3):
    """
    Returns the minimum value of two coeffcients that minimises chi-squared
    and the minimum chi-squared value itself while removing outliers
    simultaneously

    Parameters
    ----------
    data : np.array of floats
        Data should be in format [x, y, uncertainty on y]
    initial_points : np_array of floats
        Data should be in format [x, y]
    sigma_lim : float, optional
        The multiple of the uncertainty such that if the residual is larger
        than the product, the data point will be removed. The default is 3 due
        to size of the data set.
    removal_iterations : int, optional
        The number of repeats in fitting the data, removing outliers and then
        redoing fit. The default is 3.

    Returns
    -------
    minimums : np.array of floats
        Array of the two values that give the minimum value of chi-squared.
    function_minimum : float
        The minimum chi-squared value.
    data : np.array of floats
        Data should be in format [x, y, uncertainty on y]
    """
    anomalies = np.zeros((0, 3))

    for _ in range(removal_iterations):

        minimums, function_minimum, _, _, _ = fmin(lambda coefficient:
                      chi_square(data[:, 1], data[:, 2],
                      bw_function(data[:, 0], coefficient[0],
                      coefficient[1])), initial_points, full_output=True,
                      disp=False)

        indicies = np.unique(np.where((np.abs(bw_function(data[:, 0],
        minimums[0], minimums[1]) - data[:, 1]) > sigma_lim * data[:, 2]))[0])
        anomalies_temp = data[indicies]
        anomalies = np.vstack((anomalies, anomalies_temp))
        data = np.delete(data, indicies, 0) #removes outliers that are
        #anomalies due to their residual being too big compared with the fit

    minimums, function_minimum, _, _, _ = fmin(lambda coefficient:
                       chi_square(data[:, 1], data[:, 2],
                       bw_function(data[:, 0], coefficient[0],
                       coefficient[1])), initial_points, full_output=True,
                       disp=False)

    return minimums, function_minimum, data, anomalies


def uncertainties(verticies):
    """
    Returns half the ranges in a 2D np.array

    Parameters
    ----------
    verticies : 2D np.array

    Returns
    -------
    half_coeff1_range : float

    half_coeff2_range : float

    """
    half_coeff1_range = (np.max(verticies[:,0]) - np.min(verticies[:,0])) / 2

    half_coeff2_range = (np.max(verticies[:,1]) - np.min(verticies[:,1])) / 2

    return half_coeff1_range, half_coeff2_range


def plot_starting_points(results, data_1, data_2):
    """
    Plots two lines and the intersection between them

    Parameters
    ----------
    results : np.array of floats
        Data should be in the format [x, y], where x and y specify the
        coordinates of the intersection point

    data_1 : np.array of floats
        Data should be in format [x, y] and is the first line to be plotted

    data_2 : np.array of floats
        Data should be in format [x, y] and is the second line to be plotted

    Returns
    -------
    None.

    """
    fig = plt.figure()
    starting_ax = fig.add_subplot(111)

    starting_ax.plot(data_1[:,0], data_1[:,1], color='green')

    starting_ax.plot(data_2[:,0], data_2[:,1], color='orange')

    intersection_point = starting_ax.scatter([results[0]], [results[1]],
                            color='r', marker='x', label='Intersection point')
    starting_ax.legend(handles=[intersection_point])

    starting_ax.set_title('Graphical method to estimate the mass and width of'
        ' the \n Z boson from given energy and cross-section using a \n '
        'rearrangement of the Breit-Wigner expression', fontname='serif')

    starting_ax.set_xlabel('Mass / GeV/c$^{2}$', fontname='serif')
    starting_ax.set_ylabel('Width / GeV', fontname='serif')

    plt.savefig('Starting points graph.png', dpi=300,
                bbox_inches="tight")
    plt.show()


def plot_fit_results(data, anomalies, result):
    """
    Plots a minimising chi-squared fit to a set of data

    Parameters
    ----------
    data : numpy array of floats
        Should be in format [x, y, y_uncertainty]
    result : array of floats
        Optimum values for two coefficients

    Returns
    -------
    None.

    """
    fig = plt.figure()

    fit_ax = fig.add_subplot(111)

    data_points = fit_ax.errorbar(data[:,0], data[:,1], yerr=data[:,2],
                fmt='o', color='blue', label='fitted data', markersize=4)
    anomaly_points = fit_ax.errorbar(anomalies[:,0], anomalies[:,1],
                yerr=anomalies[:,2], fmt='o', color='red', label='anomaly',
                markersize=4)
    fit_ax.plot(data[:,0], bw_function(data[:,0], result[0], result[1]),
            color='orange', linewidth=2.5)

    fit_ax.set_title('Plot of cross-section against energy for the Z boson in'
    '\n an electron - positron reaction with a line of best fit',
    fontname='serif')
    fit_ax.set_xlabel('Energy / GeV', fontname='serif')
    fit_ax.set_ylabel('Cross-section / nanobarns', fontname='serif')

    fit_ax.legend(handles=[data_points, anomaly_points], loc='upper left')

    fit_ax.text(92.8, 1.93, '$m_{Z}$'+f' = {result[0]:.4g}'+'\n $\\Gamma_{Z}$'
                + ' =' + f' {result[1]:.4g}', backgroundcolor='lightgrey')

    plt.savefig('Cross-section against energy fit.png', dpi=300,
                bbox_inches="tight")
    plt.show()


def plot_contour(x_array, y_array, data, minimums):
    """
    Plots contour with a red ring for the x-y-values of the minimum z point
    plus one and a cross for the minimum z point.

    Parameters
    ----------
    x_array : np.array of floats

    y_array : np.array of floats

    data : np.array of floats
        Should be in format [x, y, y_uncertainty]

    minimums : array of floats
        Should be in format [minimum_x, minimum_y, minimum_z], where x_minimum
        and y_minimum are the coordinates such that z is minimised

    Returns
    -------
    verticies_chi2_plus_1 : np.array of floats
        Data should be in the format of [x, y], which correspond to
        coordinates
    """
    x_values_mesh, y_values_mesh = mesh_arrays(x_array, y_array)

    fig = plt.figure()
    contour_ax = fig.add_subplot(111)

    chi2_contour = contour_ax.contour(x_values_mesh, y_values_mesh,
             chi_square_array(data, x_values_mesh, y_values_mesh),
             levels=[90, 93, 97, 103, 111, 120, 130, 137])

    contour_ax.set_title('Plot of chi-squared for varying mass and width for'
                         ' the Z boson', fontname='serif')
    contour_ax.set_xlabel('Mass / GeV/c$^{2}$')
    contour_ax.set_ylabel('Width / GeV')

    chi2_plus_1_contour = contour_ax.contour(x_values_mesh, y_values_mesh,
             chi_square_array(data, x_values_mesh, y_values_mesh),
             levels=[minimums[2] + 1], colors='r', linestyles='dashed')

    contour_ax.clabel(chi2_contour, inline=1, fontsize=10)
    contour_ax.clabel(chi2_plus_1_contour, inline=1, fontsize=10,
              fmt='$\\chi^{2}_{min} + 1$')

    verticies_chi2_plus_1 = \
    chi2_plus_1_contour.collections[0].get_paths()[0].vertices

    chi2_min_point = contour_ax.scatter([minimums[0]], [minimums[1]],
                     color='black', marker='x', label='$\\chi^{2}_{min}$' +
                     f' = {minimums[2]:.2f}')
    contour_ax.legend(handles=[chi2_min_point])

    plt.savefig('Chi-squared contour plot.png', dpi=300, bbox_inches="tight")

    plt.show()

    return verticies_chi2_plus_1

def main():
    """
    Main code for programme. Reads data then performs a minimised chi
    squared fit for two parameters with 1 sigma estimations from calculated
    starting points. Plots the graphical method to determine the starting
    points, the result and a contour plot of chi-squared
    """
    data_1 = read_data(FILE_NAME_1)
    data_2 = read_data(FILE_NAME_2)
    data = np.vstack((data_1, data_2))
    data = data[np.argsort(data[:, 0])]

    energies, cross_sections = estimations(data)

    mass_range = np.arange(80, 100, 0.01)
    initial_points, line1_array, line2_array = starting_points(energies,
                                    cross_sections, mass_range)

    plot_starting_points(initial_points, line1_array, line2_array)

    [boson_mass, boson_width], chi2_min, data, anomalies = find_minimums(data,
    initial_points)
    reduced_chi_square = chi2_min / (np.shape(data)[0] - 2)

    boson_lifetime = pc.hbar / (boson_width * pc.eV * 1e9)

    plot_fit_results(data, anomalies, [boson_mass, boson_width])

    mass_values = np.linspace(91.12, 91.24, 36)
    width_values = np.linspace(2.42, 2.6, 36)
    verticies = plot_contour(mass_values, width_values, data, [boson_mass,
                                                    boson_width, chi2_min])

    boson_mass_uncert, boson_width_uncert = uncertainties(verticies)

    boson_lifetime_uncert = boson_lifetime * boson_width_uncert / boson_width

    print(f'Results: \nReduced chi squared: {reduced_chi_square:.3f} \n'
          f'Z-boson mass: {boson_mass:.4g} ± {boson_mass_uncert:.2f}'
          f' GeV/c\u00b2\n'
          f'Z-boson width: {boson_width:.4g} ± {boson_width_uncert:.3f} GeV\n'
          f'Z-boson lifetime: ({boson_lifetime:.3g} ± '
          f'{boson_lifetime_uncert:.1g}) s')


if __name__ == "__main__":
    main()
