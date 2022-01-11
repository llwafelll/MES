import numpy as np
from numpy.polynomial.legendre import leggauss
import itertools as it
from pcs_iterators import *
from constants import *

class Element4pc_2D:

    # === VARTIABLES ===
    # Gauss integration variables
    pcs, ws = leggauss(2)
    Npcs = 4 # Number of integration points

    # Generate indices for ksi and eta e.g. ksi = 0, 1, 1, 0
    ksi_order = list(gen_ksi_order(2))
    eta_order = list(gen_eta_order(2))

    # Create object that will return elements from given array in ksi or eta order
    get_values_in_ksi_order_for = leg_iterator(ksi_order)
    get_values_in_eta_order_for = leg_iterator(eta_order)
    
    # Create lists containing integration points and wages in right eta_order
    # for each axis
    pcs_ksi_order = list(get_values_in_ksi_order_for(pcs))
    ws_ksi_order =  list(get_values_in_ksi_order_for(ws))
    pcs_eta_order = list(get_values_in_eta_order_for(pcs))
    ws_eta_order =  list(get_values_in_eta_order_for(ws))

    pcs_orders = pcs_ksi_order, pcs_eta_order
    ws_orders = ws_ksi_order, ws_eta_order


    # === SHAPE FUNCTIONS ===
    # The shape function represented by lambda expression
    # this calculation should be performed only once
    N_functions = [
        lambda ksi, eta: 1/4 * (1 - ksi) * (1 - eta),
        lambda ksi, eta: 1/4 * (1 + ksi) * (1 - eta),
        lambda ksi, eta: 1/4 * (1 + ksi) * (1 + eta),
        lambda ksi, eta: 1/4 * (1 - ksi) * (1 + eta),
    ]

    # Calculate all shape functions for each integration point
    N_matrix = np.empty((4, 4))
    for j, (ksi, eta) in enumerate(zip(*pcs_orders)):
        for i, f in enumerate(N_functions):
            N_matrix[i, j] = f(ksi, eta)
        
    

    # === SHAPE FUNCTIONS DERIVATIVES===
    # Derivatives represented by lambda expressions
    # this calculation should be performed only once
    dNdeta_lambdas = [
        lambda ksi: - 1/4 * (1 - ksi),
        lambda ksi: - 1/4 * (1 + ksi),
        lambda ksi: 1/4 * (1 + ksi),
        lambda ksi: 1/4 * (1 - ksi),
    ]

    # Calculate all dNdeta derivatives for each integration point
    dNdeta = np.empty((4, 4))
    for row, pc in enumerate(pcs_ksi_order):
        for col, dNdeta_func in enumerate(dNdeta_lambdas):
            dNdeta[row, col] = dNdeta_func(pc)

    # Derivatives represented by lambda expressions
    # this calculation should be performed only once
    dNdksi_lambdas = [
        lambda eta: - 1/4 * (1 - eta),
        lambda eta: 1/4 * (1 - eta),
        lambda eta: 1/4 * (1 + eta),
        lambda eta: - 1/4 * (1 + eta),
    ]

    # Calculate all dNdksi derivatives for each integration point
    dNdksi = np.empty((4, 4))
    for row, pc in enumerate(pcs_eta_order):
        for col, dNdksi_func in enumerate(dNdksi_lambdas):
            dNdksi[row, col] = dNdksi_func(pc)


    N_surf = np.zeros((4, 2, 4))
    pos = -1, 1, 1, -1
    Nx = 0, 1, 3, 0
    Ny = 3, 2, 2, 1
    for i in range(2):
        for pc in range(2):
            N_surf[i, pc, Nx[i]] = N_functions[Nx[i]](pos[i], pcs[pc])
            N_surf[i, pc, Ny[i]] = N_functions[Ny[i]](pos[i], pcs[pc])
    for i in range(2, 4):
        for pc in range(2):
            N_surf[i, pc, Nx[i]] = N_functions[Nx[i]](pcs[pc], pos[i])
            N_surf[i, pc, Ny[i]] = N_functions[Ny[i]](pcs[pc], pos[i])

    print()

    def __init__(self, element_size: tuple[float] = None) -> None:
        pass
        

class Element9pc_2D:
    '''This is Element9p_2D but the name is Element4p_2D to avoid replacement
    in each occurence'''

    # === VARTIABLES ===
    # Gauss integration variables
    n = 3
    pcs, ws = leggauss(n)
    Npcs = n*n # Number of integration points

    # Generate indices for ksi and eta e.g. ksi = 0, 1, 1, 0
    # ksi_order = list(gen_ksi_order(n))
    # eta_order = list(gen_eta_order(n))
    ksi_order = [0, 2, 2, 0, 1, 2, 1, 0, 1]
    eta_order = [0, 0, 2, 2, 0, 1, 2, 1, 1]

    # Create object that will return elements from given array in ksi or eta order
    get_values_in_ksi_order_for = leg_iterator(ksi_order)
    get_values_in_eta_order_for = leg_iterator(eta_order)
    
    # Create lists containing integration points and wages in right eta_order
    # for each axis
    pcs_ksi_order = list(get_values_in_ksi_order_for(pcs))
    ws_ksi_order =  list(get_values_in_ksi_order_for(ws))
    pcs_eta_order = list(get_values_in_eta_order_for(pcs))
    ws_eta_order =  list(get_values_in_eta_order_for(ws))

    pcs_orders = pcs_ksi_order, pcs_eta_order
    ws_orders = ws_ksi_order, ws_eta_order


    # === SHAPE FUNCTIONS ===
    # The shape function represented by lambda expression
    # this calculation should be performed only once
    N_functions = [
        lambda ksi, eta: 1/4 * (1 - ksi) * (1 - eta),
        lambda ksi, eta: 1/4 * (1 + ksi) * (1 - eta),
        lambda ksi, eta: 1/4 * (1 + ksi) * (1 + eta),
        lambda ksi, eta: 1/4 * (1 - ksi) * (1 + eta),
    ]

    # Calculate all shape functions for each integration point
    N_matrix = np.empty((Npcs, 4))
    for i, f in enumerate(N_functions):
        for j, (ksi, eta) in enumerate(zip(*pcs_orders)):
            N_matrix[j, i] = f(ksi, eta)
        
    

    # === SHAPE FUNCTIONS DERIVATIVES===
    # Derivatives represented by lambda expressions
    # this calculation should be performed only once
    dNdeta_lambdas = [
        lambda ksi: - 1/4 * (1 - ksi),
        lambda ksi: - 1/4 * (1 + ksi),
        lambda ksi: 1/4 * (1 + ksi),
        lambda ksi: 1/4 * (1 - ksi),
    ]

    # Calculate all dNdeta derivatives for each integration point
    dNdeta = np.empty((Npcs, 4))
    for row, pc in enumerate(pcs_ksi_order):
        for col, dNdeta_func in enumerate(dNdeta_lambdas):
            dNdeta[row, col] = dNdeta_func(pc)

    # Derivatives represented by lambda expressions
    # this calculation should be performed only once
    dNdksi_lambdas = [
        lambda eta: - 1/4 * (1 - eta),
        lambda eta: 1/4 * (1 - eta),
        lambda eta: 1/4 * (1 + eta),
        lambda eta: - 1/4 * (1 + eta),
    ]

    # Calculate all dNdksi derivatives for each integration point
    dNdksi = np.empty((Npcs, 4))
    for row, pc in enumerate(pcs_eta_order):
        for col, dNdksi_func in enumerate(dNdksi_lambdas):
            dNdksi[row, col] = dNdksi_func(pc)


    N_surf = np.zeros((4, n, 4))
    pos = -1, 1, 1, -1
    Nx = 0, 1, 3, 0
    Ny = 3, 2, 2, 1
    for i in range(2):
        for pc in range(3):
            N_surf[i, pc, Nx[i]] = N_functions[Nx[i]](pos[i], pcs[pc])
            N_surf[i, pc, Ny[i]] = N_functions[Ny[i]](pos[i], pcs[pc])
    for i in range(2, 4):
        for pc in range(3):
            N_surf[i, pc, Nx[i]] = N_functions[Nx[i]](pcs[pc], pos[i])
            N_surf[i, pc, Ny[i]] = N_functions[Ny[i]](pcs[pc], pos[i])

    print()

if __name__ == "__main__":
    e1 =  Element4pc_2D()
    # e1.show_results()
