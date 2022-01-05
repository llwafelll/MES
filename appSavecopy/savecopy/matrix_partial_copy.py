import numpy as np
from numpy.polynomial.legendre import leggauss
import itertools as it
from pcs_iterators import *
from constants import *

class Element4p_2D:

    # Static fields
    pcs, ws = L = _L = leggauss(2) # _L == [[x1, x2], [w1, w2]]
    L = np.hstack((L[0], L[0][::-1]))

    Npcs = 4
    J = np.empty((2, 2))
    Jinv = np.empty((2, 2))

    ksi_order = list(gen_ksi_order(2))
    eta_order = list(gen_eta_order(2))

    get_values_in_ksi_order_for = leg_iterator(ksi_order)
    get_values_in_eta_order_for = leg_iterator(eta_order)
    
    pcs_ksi_order = list(get_values_in_ksi_order_for(pcs))
    ws_ksi_order =  list(get_values_in_ksi_order_for(ws))
    pcs_eta_order = list(get_values_in_eta_order_for(pcs))
    ws_eta_order =  list(get_values_in_eta_order_for(ws))

    pcs_orders = pcs_ksi_order, pcs_eta_order
    ws_orders = ws_ksi_order, ws_eta_order
    
    # The shape function represented by lambda expression
    # this calculation should be performed only once
    N_functions = [
        lambda ksi, eta: 1/4 * (1 - ksi) * (1 - eta),
        lambda ksi, eta: 1/4 * (1 + ksi) * (1 - eta),
        lambda ksi, eta: 1/4 * (1 + ksi) * (1 + eta),
        lambda ksi, eta: 1/4 * (1 - ksi) * (1 + eta),
    ]

    N_matrix = np.empty((4, 4))
    for j, (ksi, eta) in enumerate(zip(*pcs_orders)):
        for i, f in enumerate(N_functions):
            N_matrix[i, j] = f(ksi, eta)
        
    
    pre_C_matrix = rho * C_p * N_matrix @ N_matrix.T
    print()


    # Derivatives represented by lambda expressions
    # this calculation should be performed only once
    d_eta_lambdas = [
        lambda ksi: - 1/4 * (1 - ksi),
        lambda ksi: - 1/4 * (1 + ksi),
        lambda ksi: 1/4 * (1 + ksi),
        lambda ksi: 1/4 * (1 - ksi),
    ]

    dNdeta = np.empty((4, 4))
    print("dNdeta")
    for row, pc in enumerate(pcs_ksi_order):
        for col, dNdeta_func in enumerate(d_eta_lambdas):
            print(pc, end=" | ")
            dNdeta[row, col] = dNdeta_func(pc)
        print()

    # Derivatives represented by lambda expressions
    # this calculation should be performed only once
    d_ksi_lambdas = [
        lambda eta: - 1/4 * (1 - eta),
        lambda eta: 1/4 * (1 - eta),
        lambda eta: 1/4 * (1 + eta),
        lambda eta: - 1/4 * (1 + eta),
    ]

    dNdksi = np.empty((4, 4))
    for row, pc in enumerate(pcs_eta_order):
        for col, dNdksi_func in enumerate(d_ksi_lambdas):
            dNdksi[row, col] = dNdksi_func(pc)

    _Hbc = np.zeros((4, 4, 4))

    # P vectors
    _Pvector = np.zeros((4, 4))

    _i = 0, 1, 3, 0 # first shape function indices
    _j = 3, 2, 2, 1 # second shape function indices
    _pos = -1, 1, 1, -1 # ksi and eta values
    rpcs = pcs[::-1]
    rws = ws[::-1]

    # left and right edges
    for c, ((i, j), pos) in enumerate(zip(zip(_i[:2], _j[:2]), _pos[:2])):
        Na = np.zeros((2, 4))
        for pc in range(2):
            Na[pc, i] = N_functions[i](pos, rpcs[pc])
            Na[pc, j] = N_functions[j](pos, rpcs[pc])
            _Hbc[c] += rws[pc] * Na[pc][:, np.newaxis] * Na[pc]
            _Pvector[c] += rws[pc] * Na[pc] * 1200

    # up and down edges
    for c, ((i, j), pos) in enumerate(zip(zip(_i[2:], _j[2:]), _pos[2:])):
        Na = np.zeros((2, 4))
        for pc in range(2):
            Na[pc, i] = N_functions[i](rpcs[pc], pos)
            Na[pc, j] = N_functions[j](rpcs[pc], pos)
            _Hbc[c + 2] += rws[pc] * Na[pc][:, np.newaxis] * Na[pc]
            _Pvector[c + 2] += rws[pc] * Na[pc] * 1200
    
    print()

    def __init__(self, element_size: tuple[float] = None) -> None:
        # row = pc, column = N1, N2, N3, N4
        # self.part_N_by_eta = np.empty((4, 4))
        # self.part_N_by_ksi = np.empty((4, 4))

        # FIXME: This is wrong delete that after fixing
        self.part_N_by_x = np.empty((4, 4))
        self.part_N_by_y = np.empty((4, 4))

        # Shape functions values on the edge of a element (for each edge)
        # calculated for integration points
        #     | N1 | N2 | N3 | N4 |
        # pc1 |    |    |    |    |
        # pc2 |    |    |    |    |
        self.left_edge_N = np.zeros((2, 4))
        self.right_edge_N = np.zeros((2, 4))
        self.up_edge_N = np.zeros((2, 4))
        self.bottom_edge_N = np.zeros((2, 4))
        self.edge_N = [self.left_edge_N, self.right_edge_N,
                       self.up_edge_N, self.bottom_edge_N]
        
        # FIXME: Probably no longer neded:
        # self._Hpc1 = np.zeros((4, 4))
        # self._Hpc2 = np.zeros((4, 4))
        # self._Hpc3 = np.zeros((4, 4))
        # self._Hpc4 = np.zeros((4, 4))
        # self._H = [self._Hpc1, self._Hpc2, self._Hpc3, self._Hpc4]

        # H value calculated for each edge.
        self._H_left = np.zeros((4, 4))
        self._H_right = np.zeros((4, 4))
        self._H_up = np.zeros((4, 4))
        self._H_bottom = np.zeros((4, 4))
        # self._Hbc = [self._H_left, self._H_right,
        #              self._H_up, self._H_bottom]
        
        self._Hbc = np.zeros((4, 4, 4))

        # P vectors
        self._Pvector = np.zeros((4, 4))

        # Initialize part_N_by_eta and part_N_by_ksi (line 44, 45)
        # for row, pc in enumerate(Element4p_2D.L):
        #     self._calc_derivatives_local_coordinates(row, pc)

        # FIXME: This for loop and L assignment can be completely removed
        # and nothing changes?
        # Change oreder of pc's of legendre polynomial
        # Element4p_2D.L = np.array([Element4p_2D.L[0],
        #                           Element4p_2D.L[3],
        #                           Element4p_2D.L[1],
        #                           Element4p_2D.L[2]])

        # for row, pc in enumerate(Element4p_2D.L):
        #     for i in range(4):
        #         self.part_N_by_ksi[row, i] = self.d_ksi_lambdas[i](pc)
        
        # Helper variables
        pcs = self._L[0][::-1]
        ws = self._L[1][::-1]
        self.w = ws
        # for i in range(4):
        #     self._H[i] = self.get_part_N_x()[i][:, np.newaxis] \
        #                   * self.get_part_N_x()[i] \
        #                   + self.get_part_N_y()[i][:, np.newaxis] \
        #                   * self.get_part_N_y()[i]
                          

        # Initialize _H_left[, right, up, down] (so update _Hbc list as well)
        # _Hbc is claculated based on shape functions N_i and represend H matrix
        # on a boundary
        # TODO: consider moving this part to helper method
        print()

        # for i in range(2):
        #     self.left_edge_N[i][0] = self.N_functions[0](-1, pcs[i])
        #     self.left_edge_N[i][3] = self.N_functions[3](-1, pcs[i])
        #     self._Hbc[0] += \
        #         ws[i] * self.left_edge_N[i][:, np.newaxis] * self.left_edge_N[i]
        #     self._Pvector[0] += \
        #         ws[i] * self.left_edge_N[i] * 1200

        #     self.right_edge_N[i][1] = self.N_functions[1](1, pcs[i])
        #     self.right_edge_N[i][2] = self.N_functions[2](1, pcs[i])
        #     self._Hbc[1] += \
        #         ws[i] * self.right_edge_N[i][:, np.newaxis] * self.right_edge_N[i]
        #     self._Pvector[1] += \
        #         ws[i] * self.right_edge_N[i] * 1200

        #     self.up_edge_N[i][3] = self.N_functions[3](pcs[i], 1)
        #     self.up_edge_N[i][2] = self.N_functions[2](pcs[i], 1)
        #     self._Hbc[2] += \
        #         ws[i] * self.up_edge_N[i][:, np.newaxis] * self.up_edge_N[i]
        #     self._Pvector[2] += \
        #         ws[i] * self.up_edge_N[i] * 1200

        #     self.bottom_edge_N[i][0] = self.N_functions[0](pcs[i], -1)
        #     self.bottom_edge_N[i][1] = self.N_functions[1](pcs[i], -1)
        #     self._Hbc[3] += \
        #         ws[i] * self.bottom_edge_N[i][:, np.newaxis] * self.bottom_edge_N[i]
        #     self._Pvector[3] += \
        #         ws[i] * self.bottom_edge_N[i] * 1200
        
        for _H in self._Hbc:
            print(_H)
            print()

    # def _calc_derivatives_local_coordinates(self, row, pc):
    #     for i in range(4):
    #         self.part_N_by_eta[row, i] = self.d_eta_lambdas[i](pc)
    #         self.part_N_by_ksi[row, i] = self.d_ksi_lambdas[i](pc)
        
    #     # FIXME: This is very ugly solution
    #     self.part_N_by_ksi[[0, 1, 2, 3]] = self.part_N_by_ksi[[0, 3, 2, 1]]
    
    def calc_derivatives_global_coordinates(self, i, j, grid):
        grid.jakobian(i, j, self.J, self.Jinv, self, grid)
        w = np.array(list(zip(self.part_N_by_ksi[j], self.part_N_by_eta[j])))
        for k in range(4):
            self.part_N_by_x[j, k], self.part_N_by_y[j, k] = self.Jinv@w[k]
    
    # def get_part_N_by_eta(self) -> np.ndarray:
    #     return self.part_N_by_eta
    
    # def get_part_N_by_ksi(self) -> np.ndarray:
    #     return self.part_N_by_ksi

    def get_part_N_x(self) -> np.ndarray:
        return self.part_N_by_x
    
    def get_part_N_y(self) -> np.ndarray:
        return self.part_N_by_y
    

if __name__ == "__main__":
    e1 =  Element4p_2D()
    # e1.show_results()

# L = leggauss(2)
# _L = np.array([L[0][0], L[0][0], L[0][1], L[0][1]])
# # _L = np.hstack((L[0], L[0][::1]))[np.newaxis]

# result = np.empty((4, 4))
# for i, obj in enumerate(result):
#     for j, _ in enumerate(obj):
#         result[i, j] = der[j](_L[i])

# L = leggauss(2)
# # _L = np.array([L[0][0], L[0][0], L[0][1], L[0][1]])
# _L = np.hstack((L[0], L[0][::1]))[np.newaxis]
# result = np.empty((4, 4))
# for i, obj in enumerate(result):
#     for j, _ in enumerate(obj):
#         result[i, j] = der[j](_L[0, i])


# print(result)