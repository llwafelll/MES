import numpy as np

class ArrayCreator:
    def __init__(self):
        a1 = np.arange(1, 5)
        self.arr = a1[3]*a1[:3][:,np.newaxis] + a1
        self.arr = np.vstack((a1, self.arr))
    
    def get_arr(self):
        return self.arr

    def get_arr_view(self):
        return self.arr.view()
    
    def get_tl_corner2x2(self):
        return self.arr[:2, :2]
    
class DiagonalManager:
    def __init__(self, arg_arr):
        self.diagonal = arg_arr.get_arr().diagonal()
        self.diagonal_len = len(self.diagonal)
    
    def get_diagonal(self):
        return self.diagonal
    
    def set_diagonal(self, arg_diagonal):
        if len(arg_diagonal) == self.diagonal_len:
            self.diagonal = arg_diagonal
    
    def set_sqr(self, arg_sqr):
        self.arg_arr[:2, :2] = arg_sqr
        
        
M = ArrayCreator()
print(M.get_arr())

DM = DiagonalManager(M)
DM.set_diagonal(np.array((11, 22, 33, 44)))
print(M.get_arr())
