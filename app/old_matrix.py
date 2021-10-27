import numpy as np
from numpy.polynomial.legendre import leggauss
import itertools as it

part_N_by_eta = np.empty((4, 4))
part_N_by_ksi = np.empty((4, 4))

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

L = leggauss(2)
L = np.hstack((L[0], L[0][::-1]))
# [-0.57735027,  0.57735027,  0.57735027, -0.57735027]

for row, pc in enumerate(L):
    for i, eta in enumerate(part_N_by_eta):
        part_N_by_eta[row, i] = eta_lambdas[i](pc)

for row, pc in enumerate(L):
    for i, eta in enumerate(part_N_by_ksi):
        part_N_by_ksi[row, i] = ksi_lambdas[i](pc)

print("part_N_by_eta:")
print(part_N_by_eta)
print("\npart_N_by_ksi:")
print(part_N_by_ksi)

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