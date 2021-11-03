import numpy as np
from numpy.polynomial.legendre import leggauss
import itertools as it

class Element4p_2D:
    L = leggauss(2)
    L = np.hstack((L[0], L[0][::-1]))
    # [-0.57735027,  0.57735027,  0.57735027, -0.57735027]
    N = 4

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

        for row, pc in enumerate(Element4p_2D.L):
            for i in range(4):
                self.part_N_by_eta[row, i] = self.eta_lambdas[i](pc)

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
    
    def get_part_N_by_eta(self) -> np.ndarray:
        return self.part_N_by_eta
    
    def get_part_N_by_ksi(self) -> np.ndarray:
        return self.part_N_by_ksi

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