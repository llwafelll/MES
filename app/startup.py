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
    def __init__(self, n_nodes: tuple, size: tuple = None,
                 arg_nodes: np.ndarray = np.array([])) -> None:
                
        # Creates NodesContainer whose _array keep references certain nodes
        if arg_nodes.size:
            self._array: np.array = np.empty(n_nodes, dtype=Node)

            for col in range(n_nodes[1]):
                for row in reversed(range(n_nodes[0])):
                    self._array[row, col] = arg_nodes[row, col]
        
        # Creates entirely new nodes (usend in initialization)
        elif size:
            # n_nodes: tuple -> N_NODES_VERTICAL, N_NODES_HORIZONTAL
            self._array: np.ndarray = np.empty(n_nodes, dtype=Node)
            dh: float = size[0]/(n_nodes[0] - 1)
            dw: float = size[1]/(n_nodes[1] - 1)
            
            for col in range(n_nodes[1]):
                for row in reversed(range(n_nodes[0])):
                    pos_x: float = dw * col
                    pos_y: float = dh * (n_nodes[0] - 1 - row)

                    self._array[row, col] = Node(pos_x, pos_y)
            
            self.shape = self._array.shape
        
            self.print_array()
        
    def get_nodes_surrouding_element(self, arg_id: int) -> np.ndarray:
        # arg_id is element id
        # self.shape returns nodesContainer shape

        # el_x, el_y = Grid.convert_id_to_coord(arg_id, self.shape[0])
        # node_x, node_y = Grid.convert_id_to_coord(arg_id, self.shape[0] + 1)
        x, y = Grid.convert_id_to_coord(arg_id, self.shape[0] - 1)

        # node left down corner id:
        node_id = arg_id + x
        node_x, node_y = Grid.convert_id_to_coord(node_id, self.shape[0])
        v_from, v_to = node_y - 1, node_y + 1
        h_from, h_to = node_x, node_x + 2

        # v_from, v_to = self.shape[0] - y, self.shape[0] - node_id + 1
        # h_from, h_to = x, node_id + 1

        # return self._array[_from:_to, N - _to:N - _from]
        # return self._array[N - _to:N - _from, _from:_to] # <- right
        return self._array[v_from:v_to, h_from:h_to]

    def get_by_id(self, id: int) -> np.ndarray:
        x, y = Grid.convert_id_to_coord(id, self.shape[0])
        # x = (id - 1) // (self.shape[0])
        # y = (id - 1) % self.shape[0]
        # y = ((self.shape[0] - 1) - y)
        # x = self._calc_x()
        # y = self._calc_y()

        return self[y, x]
    
    def _calc_x(self, id: int) -> int:
        return (id - 1) // (self.shape[0])
    
    def _calc_y(self, id: int) -> int:
        y = (id - 1) % self.shape[0]
        return ((self.shape[0] - 1) - y)
    
    def print_array(self):
        for i in self._array:
            for j in i:
                print(j._id, end=' ')
            print()
    
    def print_all_data(self):
        for i in self._array:
            for j in i:
                print(f"id:{j._id:0>2d}", end='')
                print(f"x:{j.x:0=2.2f}", end='')
                print(f"y:{j.y:0=2.2f} | ", end='')
                
            print()
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
        # self.surr_nodes: np.ndarray = arg_nodes.get_nodes_surrouding_element(self._id)
        self.surr_nodes: NodesContainer = \
            NodesContainer(
                n_nodes=(2, 2),
                arg_nodes=arg_nodes.get_nodes_surrouding_element(self._id)
                )

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
        x, y = Grid.convert_id_to_coord(id, self.shape[0])
        # x = (id - 1) // (self.shape[0])
        # y = (id - 1) % self.shape[0]
        # y = ((self.shape[0] - 1) - y)

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
        self.NODES: NodesContainer = NodesContainer(self.get_n_nodes(), size=self.get_size())
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
    
    @staticmethod
    def convert_id_to_coord(arg_id: int, height: int):
        x = (arg_id - 1) // height
        y = (arg_id - 1) % height
        y = ((height - 1) - y)

        return x, y

    

if __name__ == "__main__":
    g = Grid(height=3.3, width=4.2, nodes_vertiacl=3, nodes_horizontal=4)
    g.NODES.print_all_data()
    # g.ELEMENTS.print_elements()
    g.ELEMENTS.get_by_id(6).surr_nodes.print_all_data()
    # print(g.ELEMENTS.get_by_id(2))

    # Testing
    # for i in g.NODES._array:
    #     for j in i:
    #         print(j._id, end=' ')
    #     print()

    # for i in g.ELEMENTS._array[0, 0].surr_nodes:
    #     for j in i:
    #         print(j._id, end=' ')
    #     print()

    # print()

    # g.ELEMENTS._array[0, 0].surr_nodes[0, 0].x = 100
    # g.ELEMENTS._array[0, 0].surr_nodes[0, 0].y = 101
    # for i in g.NODES._array:
    #     for j in i:
    #         print(j._id, end=' ')
    #     print()

    # for i in g.ELEMENTS._array[0, 0].surr_nodes:
    #     for j in i:
    #         print(j._id, end=' ')
    #     print()

    # print()