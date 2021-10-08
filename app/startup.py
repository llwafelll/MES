# TODO: find metaclass that support struct-like classes in python
# https://www.youtube.com/watch?v=vBH6GRJ1REM
# TODO: matplotlib to visualize

# TODO: introduce NodeList?
import numpy as np


class Node:
    _counter: int = 0
    
    def __init__(self, arg_x: float, arg_y: float) -> None:
        self.x: float = arg_x
        self.y: float = arg_y
        
        self._id: int = Node._counter
        Node._counter += 1
    
    def get_position(self):
        return (self.x, self.y)
    

class NodesContainer:
    def __init__(self, size: tuple, nodes: tuple) -> None:
        self._array = np.empty(nodes, dtype=Node)
        for col in reversed(range(nodes[1])):
            for row in range(nodes[0]):
                self._array[row, col] = Node(size[0], size[1])
    
        self.print_array()
        
    def get_surrounding_nodes(self) -> np.array:
        _from, _to = Element._counter, Element._counter + 1
        return self._array[_from:_to, _from:_to]
    
    def print_array(self):
        for i in self._array:
            for j in i:
                print(j._id, end=' ')
            print()


# TODO: introduce ElementList?
class Element:
    _counter: int = 0
    
    def __init__(self, arg_nodes: NodesContainer) -> None:
        self.surr_nodes: NodesContainer = arg_nodes.get_surrounding_nodes()
        self._id: int = Element._counter
        Element._counter += 1


class ElementsContainer:
    def __init__(self, size: tuple, nodes: NodesContainer) -> None:
        self._array = np.empty(size, dtype=Element)
        for row in range(size[1]):
            for col in range(size[0]):
                self._array[row, col] = Element(nodes)
    

class Grid:
    def __init__(self,
                 height: float, width: float,
                 nodes_vertiacl: int, nodes_horizontal: int) -> None:

        # height and width for whole square
        self.HEIGHT: float = height
        self.WIDTH: float = width

        # number of nodes vertically and horizontally
        self.N_NODES_VERTICAL: int = nodes_vertiacl
        self.N_NODES_HORIZONTAL: int = nodes_horizontal

        # total number of nodes and elements
        self.N_NODES_TOTAL: int = self.N_NODES_HORIZONTAL * self.N_NODES_VERTICAL
        self.N_ELEMENTS_TOTAL: int = \
            (self.N_NODES_HORIZONTAL - 1) * (self.N_NODES_VERTICAL - 1)

        # TODO: initialize following attributes
        # references to the objects?
        self.NODES: NodesContainer = NodesContainer(self.get_size(), self.get_n_nodes())
        self.ELEMENTS: ElementsContainer = \
            ElementsContainer(self.get_n_elements(), self.NODES)

        
    def get_size(self):
        return (self.HEIGHT, self.WIDTH)
    
    def get_n_nodes(self):
        return (self.N_NODES_VERTICAL, self.N_NODES_HORIZONTAL)
    
    def get_n_elements(self):
        return (self.N_NODES_VERTICAL - 1, self.N_NODES_HORIZONTAL - 1)


g = Grid(height=3.3, width=4.2, nodes_vertiacl=3, nodes_horizontal=4)
print()