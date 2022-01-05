from typing import Generator
import numpy as np
from numpy.polynomial import legendre
from numpy.polynomial.legendre import leggauss
import itertools as it

class Integral:
    N = 3
    L = leggauss(N)
    s = lambda q: (q[1] - q[0])/2 # subtract
    a = lambda q: (q[1] + q[0])/2 # add

    def __init__(self, function, dim: int) -> None:
        self.f = function
        self.dim: int = dim
    
    def calculate(self, interval: tuple = (-1, 1)) -> float:
        # Stores the result
        res: float = 0

        # x - stores x_i points from Gauss integration and their coresponding wages w_i
        x, w = Integral.L

        # Generates indices for each nested loop e.g. [[0, 1, 2], [0, 1, 2]]
        g = ([i for i in range(Integral.N)] for _ in range(self.dim))
        # g = ([i for i in range(self.dim)] for _ in range(Integral.N))

        # Helper variables to integrate at any interval
        p1 = Integral.s(interval)
        p2 = Integral.a(interval)

        # Lambda expression for creating generator to get value of
        # w (wage) or x at given index
        Vi = lambda v: (v[i] for i in indices)

        # Helper lambda to transform v (probably x) to fit given interval
        arg = lambda v: p1 * v + p2

        # Integrate
        # it.product creates cartesian product from indices
        for indices in it.product(*g):
            # p1 * [get value of function for x_ij suited to interval] *
            # * [get product of all wages that are asociated with x_ij]
            temp = self.f(*[arg(v) for v in Vi(x)]) * np.prod(list(Vi(w)))
            res += temp

        return p1**self.dim * res

        # if self.dim == 2:
        #     xs = Integral.L[0]

        #     for j, wy in enumerate(Integral.L[1]):
        #         for i, wx in enumerate(Integral.L[1]):
        #             res += self.f(xs[i], xs[j]) * wx * wy
        
        
        # elif self.dim == 1:
        #     res: float = 0
        #     s = lambda i: (i[1] - i[0])/2 # subtract
        #     a = lambda i: (i[1] + i[0])/2 # add

        #     for x, w in np.column_stack(Integral.L):
        #         arg = s(interval) * x + a(interval)

        #         res +=  s(interval) * self.f(arg) * w
                
    
    @classmethod
    def set_number_of_points(cls, value):
        cls.N = value
        cls.L = leggauss(value)
        
            
# f1 = lambda x: 5*x**2 + 3*x + 6
# f2 = lambda x, y: 5*x**2*y**2 + 3*x*y + 6
# f3 = lambda x, y, z: 5*x**2 * y**2 * z**3 + 3*x*y*z**2 + 6
# i3 = Integral(f3, 3)
# i2 = Integral(f2, 2)
# i1 = Integral(f1, 1)

# print("\nResults for number of points = 3 (default):")
# print(i1.calculate())
# print(i1.calculate((2, 3)))
# print(i2.calculate())
# print(i2.calculate((2, 3)))
# print(i3.calculate((2, 3)))
# print(i3.calculate((-2, 10)))

# print("\nResults for number of points = 2:")
# Integral.set_number_of_points(2)
# print(i1.calculate())
# print(i1.calculate((2, 3)))
# print(i2.calculate())
# print(i3.calculate((-2, 10)))
