from matrix_partial_copy import Element4p_2D
import numpy as np
from numpy.linalg import det, inv
import itertools as it 

e1 =  Element4p_2D()
part_N_by_eta = e1.get_part_N_by_eta()
part_N_by_ksi = e1.get_part_N_by_ksi()

X = np.array([0, .025, .025, 0])
Y = np.array([0, 0, .025, .025])

x = np.sum(part_N_by_ksi[0] * X)
y = np.sum(part_N_by_eta[0] * Y)

# creatge 2x2 matrix with diagonal created based on the provided list
print("\nMacierz z x, y na przekatnej")
print(M := np.diag((x, y)))

print("\nMarcierz Jakobiego:")
print(J := inv(M))

print("\n1 / det(J) = ")
print(1 / det(J))
print()


