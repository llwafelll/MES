import numpy as np
from numpy.polynomial.legendre import leggauss
import itertools as it

ALPHA = 25
class Element4p_2D:

    # Static fields
    L = _L = leggauss(2) # _L == [[x1, x2], [w1, w2]]
    L = np.hstack((L[0], L[0][::-1]))
    # [-0.57735027,  0.57735027,  0.57735027, -0.57735027]
    Npcs = 4
    J = np.empty((2, 2))
    Jinv = np.empty((2, 2))

    # The shape function represented by lambda expression
    # this calculation should be performed only once
    N = [
        lambda ksi, eta: 1/4 * (1 - ksi) * (1 - eta),
        lambda ksi, eta: 1/4 * (1 + ksi) * (1 - eta),
        lambda ksi, eta: 1/4 * (1 + ksi) * (1 + eta),
        lambda ksi, eta: 1/4 * (1 - ksi) * (1 + eta),
    ]

    # Derivatives represented by lambda expressions
    # this calculation should be performed only once
    d_eta_lambdas = [
        lambda ksi: - 1/4 * (1 - ksi),
        lambda ksi: - 1/4 * (1 + ksi),
        lambda ksi: 1/4 * (1 + ksi),
        lambda ksi: 1/4 * (1 - ksi),
    ]

    # Derivatives represented by lambda expressions
    # this calculation should be performed only once
    d_ksi_lambdas = [
        lambda eta: - 1/4 * (1 - eta),
        lambda eta: 1/4 * (1 - eta),
        lambda eta: 1/4 * (1 + eta),
        lambda eta: - 1/4 * (1 + eta),
    ]

    def __init__(self, element_size: tuple[float] = None) -> None:
        self.part_N_by_eta = np.empty((4, 4))
        self.part_N_by_ksi = np.empty((4, 4))

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
        
        self._Hpc1 = np.zeros((4, 4))
        self._Hpc2 = np.zeros((4, 4))
        self._Hpc3 = np.zeros((4, 4))
        self._Hpc4 = np.zeros((4, 4))
        self._H = [self._Hpc1, self._Hpc2, self._Hpc3, self._Hpc4]

        # H value calculated for each edge.
        self._H_left = np.zeros((4, 4))
        self._H_right = np.zeros((4, 4))
        self._H_up = np.zeros((4, 4))
        self._H_bottom = np.zeros((4, 4))
        self._Hbc = [self._H_left, self._H_right,
                     self._H_up, self._H_bottom]

        # Initialize part_N_by_eta and part_N_by_ksi (line 44, 45)
        for row, pc in enumerate(Element4p_2D.L):
            self._calc_derivatives_local_coordinates(row, pc)

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
        pc = self._L[0][::-1]
        w = self._L[1][::-1]
        # for i in range(4):
        #     self._H[i] = self.get_part_N_x()[i][:, np.newaxis] \
        #                   * self.get_part_N_x()[i] \
        #                   + self.get_part_N_y()[i][:, np.newaxis] \
        #                   * self.get_part_N_y()[i]
                          

        # Initialize _H_left[, right, up, down] (so update _Hbc list as well)
        # _Hbc is claculated based on shape functions N_i and represend H matrix
        # on a boundary
        # TODO: consider moving this part to helper method
        for i in range(2):
            self.left_edge_N[i][0] = self.N[0](-1, pc[i])
            self.left_edge_N[i][3] = self.N[3](-1, pc[i])
            self._H_left += \
                w[i] * self.left_edge_N[i][:, np.newaxis] * self.left_edge_N[i]

            self.right_edge_N[i][1] = self.N[1](1, pc[i])
            self.right_edge_N[i][2] = self.N[2](1, pc[i])
            self._H_right += \
                w[i] * self.right_edge_N[i][:, np.newaxis] * self.right_edge_N[i]

            self.up_edge_N[i][3] = self.N[3](pc[i], 1)
            self.up_edge_N[i][2] = self.N[2](pc[i], 1)
            self._H_up += \
                w[i] * self.up_edge_N[i][:, np.newaxis] * self.up_edge_N[i]

            self.bottom_edge_N[i][0] = self.N[0](pc[i], -1)
            self.bottom_edge_N[i][1] = self.N[1](pc[i], -1)
            self._H_bottom += \
                w[i] * self.bottom_edge_N[i][:, np.newaxis] * self.bottom_edge_N[i]
        
        # for _H in self._Hbc:
        #     print(_H)
        #     print()

    def show_results(self):
        print("part_N_by_eta:")
        print(self.part_N_by_eta)
        print("\npart_N_by_ksi:")
        print(self.part_N_by_ksi)
        print("\npart_N_by_x:")
        print(self.part_N_by_x)
        print("\npart_N_by_y:")
        print(self.part_N_by_y)
        print()

    def _calc_derivatives_local_coordinates(self, row, pc):
        for i in range(4):
            self.part_N_by_eta[row, i] = self.d_eta_lambdas[i](pc)
            self.part_N_by_ksi[row, i] = self.d_ksi_lambdas[i](pc)
    
    def calc_derivatives_global_coordinates(self, i, j, grid):
        grid.jakobian(i, j, self.J, self.Jinv, self, grid)
        w = np.array(list(zip(self.part_N_by_ksi[j], self.part_N_by_eta[j])))
        for k in range(4):
            self.part_N_by_x[j, k], self.part_N_by_y[j, k] = self.Jinv@w[k]
    
    def get_part_N_by_eta(self) -> np.ndarray:
        return self.part_N_by_eta
    
    def get_part_N_by_ksi(self) -> np.ndarray:
        return self.part_N_by_ksi

    def get_part_N_x(self) -> np.ndarray:
        return self.part_N_by_x
    
    def get_part_N_y(self) -> np.ndarray:
        return self.part_N_by_y
    

if __name__ == "__main__":
    e1 =  Element4p_2D()
    e1.show_results()

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