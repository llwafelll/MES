import numpy as np
from numpy.polynomial.legendre import leggauss
import itertools as it

class Element4p_2D:
    L = leggauss(2)
    L = np.hstack((L[0], L[0][::-1]))
    # [-0.57735027,  0.57735027,  0.57735027, -0.57735027]
    N = 4
    J = np.empty((2, 2))
    Jinv = np.empty((2, 2))

    eta_lambdas = [
        lambda ksi: - 1/4 * (1 - ksi),
        lambda ksi: - 1/4 * (1 + ksi),
        lambda ksi: 1/4 * (1 + ksi),
        lambda ksi: 1/4 * (1 - ksi),
    ]

    ksi_lambdas = [
        lambda eta: - 1/4 * (1 - eta),
        lambda eta: 1/4 * (1 - eta),
        lambda eta: 1/4 * (1 + eta),
        lambda eta: - 1/4 * (1 + eta),
    ]

    def __init__(self) -> None:
        self.part_N_by_eta = np.empty((4, 4))
        self.part_N_by_ksi = np.empty((4, 4))

        self.part_N_by_x = np.empty((4, 4))
        self.part_N_by_y = np.empty((4, 4))

        for row, pc in enumerate(Element4p_2D.L):
            self._calc_derivatives_local_coordinates(row, pc)

        Element4p_2D.L = np.array([Element4p_2D.L[0],
                                  Element4p_2D.L[3],
                                  Element4p_2D.L[1],
                                  Element4p_2D.L[2]])
        for row, pc in enumerate(Element4p_2D.L):
            for i in range(4):
                self.part_N_by_ksi[row, i] = self.ksi_lambdas[i](pc)

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
            self.part_N_by_eta[row, i] = self.eta_lambdas[i](pc)
            self.part_N_by_ksi[row, i] = self.ksi_lambdas[i](pc)
    
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