# TODO: find metaclass that support struct-like classes in python
# https://www.youtube.com/watch?v=vBH6GRJ1REM
# TODO: matplotlib to visualize

import numpy as np

# TODO: Think about changing datatype of surr_nodes to NodesContainer
# remember to keep nodes as references


class Node:
    _counter: int = 1
    
    def __init__(self, arg_x: float, arg_y: float) -> None:
        self.x: float = arg_x
        self.y: float = arg_y
        
        self._id: int = Node._counter
        Node._counter += 1
    
    def get_position(self):
        return (self.x, self.y)
    
    def get_id(self):
        return self._id
    

class NodesContainer:
    def __init__(self, size: tuple, n_nodes: tuple) -> None:
        # n_nodes: tuple -> N_NODES_VERTICAL, N_NODES_HORIZONTAL
        self._array: np.ndarray = np.empty(n_nodes, dtype=Node)
        dh: float = size[0]/n_nodes[0]
        dw: float = size[1]/n_nodes[1]
        
        for col in range(n_nodes[1]):
            for row in reversed(range(n_nodes[0])):
                pos_x: float = dw * col
                pos_y: float = dh * (n_nodes[0] - 1 - row)

                self._array[row, col] = Node(pos_x, pos_y)
        
        self.shape = self._array.shape
    
        self.print_array()
        
    def get_nodes_surrouding_element(self, arg_id: int) -> np.ndarray:
        _from, _to = arg_id - 1, arg_id + 1
        N: int = len(self._array)

        return self._array[_from:_to, N - _to:N - _from]
        # return self._array[N - _to:N - _from, _from:_to]

    def get_by_id(self, id: int) -> np.ndarray:
        x = (id - 1) // (self.shape[0])
        y = (id - 1) % self.shape[0]
        y = ((self.shape[0] - 1) - y)

        return self[y, x]
    
    def print_array(self):
        for i in self._array:
            for j in i:
                print(j._id, end=' ')
            print()
    
    def __getitem__(self, pos: tuple):
        i, j = pos
        return self._array[i, j]
        


# TODO: introduce ElementList?
class Element:
    _counter: int = 1
    
    def __init__(self, arg_nodes: NodesContainer) -> None:
        # shape = arg_nodes._array.shape
        self._id: int = Element._counter
        self.surr_nodes: np.ndarray = arg_nodes.get_nodes_surrouding_element(self._id)

        Element._counter += 1


class ElementsContainer:
    def __init__(self, size: tuple, nodes: NodesContainer) -> None:
        self._array: np.ndarray = np.empty(size, dtype=Element)
        
        for col in range(size[1]):
            for row in reversed(range(size[0])):
                self._array[row, col] = Element(nodes)
        
        self.shape = self._array.shape
    
        self.print_elements()
    
    def get_by_id(self, id: int) -> np.ndarray:
        x = (id - 1) // (self.shape[0])
        y = (id - 1) % self.shape[0]
        y = ((self.shape[0] - 1) - y)

        return self[y, x]


    def print_elements(self):
        for i in self._array:
            for j in i:
                print(j._id, end=' ')
            
            print()

    def __getitem__(self, pos: tuple) -> np.ndarray:
        i, j = pos
        return self._array[i, j]

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
        '''Returns size of the metal element.'''

        return (self.HEIGHT, self.WIDTH)
    
    def get_n_nodes(self):
        '''Returns the number of the nodes that construct the grid in a tuple.'''

        return (self.N_NODES_VERTICAL, self.N_NODES_HORIZONTAL)
    
    def get_n_elements(self):
        '''Returns the number of the elements that construct the grid in a tuple.'''

        return (self.N_NODES_VERTICAL - 1, self.N_NODES_HORIZONTAL - 1)
    
    def get_element_by_id(self, arg_id: int) -> np.ndarray:
        return self.ELEMENTS.get_by_id(arg_id)
    
    def get_node_by_id(self, arg_id: int) -> np.ndarray:
        return self.NODES.get_by_id(arg_id)
    
    def get_nodes_surrouding_element(self, element_id: int) -> np.ndarray:
        return self.NODES.get_nodes_surrouding_element(element_id)
    

if __name__ == "__main__":
    g = Grid(height=3.3, width=4.2, nodes_vertiacl=3, nodes_horizontal=4)
    print(g.ELEMENTS.get_by_id(2))

    # Testing
    for i in g.NODES._array:
        for j in i:
            print(j._id, end=' ')
        print()

    for i in g.ELEMENTS._array[0, 0].surr_nodes:
        for j in i:
            print(j._id, end=' ')
        print()

    print()

    g.ELEMENTS._array[0, 0].surr_nodes[0, 0].x = 100
    g.ELEMENTS._array[0, 0].surr_nodes[0, 0].y = 101
    for i in g.NODES._array:
        for j in i:
            print(j._id, end=' ')
        print()

    for i in g.ELEMENTS._array[0, 0].surr_nodes:
        for j in i:
            print(j._id, end=' ')
        print()

    print()