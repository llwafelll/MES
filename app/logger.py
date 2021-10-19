from __future__ import annotations
from abc import ABC, abstractmethod
from itertools import cycle, zip_longest
from numpy import array, newaxis

class GridPrinter(ABC):
    def __init__(self) -> None:
        self.current_logger: GridLogger = None
    
    def set_logger(self, logger: GridLogger):
        self.current_logger: GridLogger = logger
    
    def log(self, log, mode={'id': 'ne', 'coor': 'ne', 'nofe': '1'}):
        '''id: n - print ids of all nodes
        id: e - print ids of all edges
        coor: n - print coordinates of all nodes
        coor: e - print edges and coordinates
        nofe: 'n' - print nodes surrounding edge which id is provided'''
        self.current_logger.log(log, mode)
        

class GridLogger(ABC):
    def __init__(self) -> None:
        self.grid = None
        self.STR_LEN = 17
    
    @abstractmethod
    def log(cls, grid, mode):
        pass

    @classmethod
    def _print_elements(cls, nodes, elements, strlen):
        initial = True
        for Ncol, Ecol in zip_longest(nodes, elements, fillvalue=[]):
            d: str = strlen - 2
            for node in Ncol:
                print(f"id:{node._id:0>2d}", end='')
                print(f"x:{node.x:0=2.2f}", end='')
                print(f"y:{node.y:0=2.2f} | ", end='')
                
            print()
            print('-' * (d+1), end='') if Ecol != array([]) or initial else None

            for edge in Ecol:
                print(f"id:{edge._id:0>2d}", end='-' * d)
                
            initial = False
            print()

class GridConsoleLogger(GridLogger):
    def __init__(self) -> None:
        super().__init__()

    def log(self, grid, mode):
        self.grid = grid

        if 'n' in list(mode.get('id', [])):
            print("Nodes of the grid:")
            self.grid.NODES.print_nodes()
            print()

        if 'e' in list(mode.get('id', [])):
            print("Elements of the grid:")
            self.grid.ELEMENTS.print_elements()
            print()

        if 'n' in list(mode.get('coor', [])):
            print("Printing all nodes with coordinates:")
            self.grid.NODES.print_all_data()
            print()

        if 'e' in list(mode.get('coor', [])):
            print("Printing all elements with all nodes with coordinates:")
            self._print_elements(self.grid.NODES, self.grid.ELEMENTS,
                                 self.STR_LEN)
            # the biggest grid to print
            # line_len: int = (self.STR_LEN + 3) * self.grid.N_NODES_HORIZONTAL - 1

        
        if 'nofe' in mode:
            id: int = int(mode['nofe'])
            element = array([self.grid.ELEMENTS.get_by_id(id)])[newaxis]
            surr_nodes = element[0][0].surr_nodes
            self._print_elements(surr_nodes, element, self.STR_LEN)

class GridTxtLogger(GridLogger):
    def log(cls, grid):
        pass

class GridJsonLogger(GridLogger):
    def log(cls, grid):
        pass


printer = GridPrinter()
printer.set_logger(GridConsoleLogger())