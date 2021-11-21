# TODO: find metaclass that support struct-like classes in python
# https://www.youtube.com/watch?v=vBH6GRJ1REM
# TODO: matplotlib to visualize

from __future__ import annotations
import numpy as np
from logger import *
from numpy.linalg import inv, det
from matrix_partial_copy import Element4p_2D
from termcolor import colored
from typing import Type, Union
from enum import Enum, auto
import itertools as it

# TODO: Think about changing datatype of surr_nodes to NodesContainer
# remember to keep nodes as references

K = 30
ALPHA = 25

class Direction(Enum): 
    LEFT = auto()
    RIGHT = auto()
    TOP = auto()
    BOTTOM = auto()

class Node:
    _counter: int = 1
    
    def __init__(self, arg_x: float, arg_y: float) -> None:
        self.x: float = arg_x
        self.y: float = arg_y

        self.edge = {
            "is_left": False,
            "is_right": False,
            "is_top": False,
            "is_bottom": False
        }
        
        self._id: int = Node._counter
        Node._counter += 1
    
    def get_position(self):
        return (self.x, self.y)
    
    def get_id(self):
        return self._id
    
    def set_edge(self, *args, **kwargs):
        if kwargs:
            for k, v in kwargs.items():
                self.edge[k] = v
            
        elif args:
            for k, v in zip(self.edge.keys(), args):
                self.edge[k] = v

    def update_edge(self, *args, **kwargs):
        if kwargs:
            for k, v in kwargs.items():
                self.edge[k] += v
            
        elif args:
            for k, v in zip(self.edge.keys(), args):
                self.edge[k] += v
    
    def is_edge(self):
        return any(self.edge.values())


# TODO: NodesContainer should receive NodesContainer instead of np.ndarray
class NodesContainer:
    def __init__(self, n_nodes: tuple, size: tuple = None,
                 arg_nodes: np.ndarray = np.array([])) -> None:
        '''
        ARGS:
        n_nodes - number of nodes, respectively, in y and x axis
        n_nodes: tuple -> N_NODES_VERTICAL, N_NODES_HORIZONTAL
        size - real lenght (in [cm]), respectively, in y and x axis 
        arg_nodes - two dimensional array with refs to existing nodes
        
        DESC:
        The constructor for NodesContainer works in two modes.
        MODE_1: privide arg_nodes numpy array and n_nodes to create 
                new NodeContaier which holds references to the arg_nodes objects. 
        MDOE_2: provide size and n_nodes to create new array of the nodes
                from scratch.
        '''

        # Crucial attribute of the class, while initialized should contain nodes
        self._array = np.empty(n_nodes, dtype=Node);

        # Declaration of arrays with edge nodes
        self.left_nodes = None
        self.right_nodes = None
        self.top_nodes = None
        self.bottom_nodes = None
        self.edge_nodes = None
                
        # CREATE NodeContainer for _surr_nodes
        # If arg_nodes has been provided (size is not empty) and it's
        # dimension is equal to 2 (required) then
        # create NodesContainer whose _array keep references certain nodes.
        # Probably runs while creating _surr_nodes
        if arg_nodes.size and arg_nodes.ndim == 2:
            # TODEL
            # self._array: np.array = np.empty(n_nodes, dtype=Node)

            # Initialize _array using provided array "arg_nodes"
            for row in range(n_nodes[0]):
                for col in range(n_nodes[1]):
                    self._array[row, col] = arg_nodes[row, col]

                    if self._array[row, col].edge["is_left"]:
                        self.left_nodes = self._array[:, 0]
                    if self._array[row, col].edge["is_right"]:
                        self.right_nodes = self._array[:, -1]
                    if self._array[row, col].edge["is_top"]:
                        self.top_nodes = self._array[0, :]
                    if self._array[row, col].edge["is_bottom"]:
                        self.bottom_nodes = self._array[-1, :]

                    self.edge_nodes = [self.left_nodes, self.right_nodes,
                                    self.top_nodes, self.bottom_nodes]

            # TODEL
            # for col in range(n_nodes[1]):
            #     for row in reversed(range(n_nodes[0])):
            #         self._array[row, col] = arg_nodes[row, col]
        
        # GENERATE THE GRID:
        # Creates entirely new array of nodes (from scratch).
        # The width and height is divided by number of nodes then each node
        # received it's coordinates.
        # This procedure should be executed only once while initializing
        # a new grid.
        elif size:
            # TODEL
            # self._array: np.ndarray = np.empty(n_nodes, dtype=Node)

            # dh - delta height, dw - delta width
            dh: float = size[0]/(n_nodes[0] - 1)
            dw: float = size[1]/(n_nodes[1] - 1)
            
            # Loop that calculates coordinates and uses them to construct
            # new Node objects.
            # First loop col then reversed rows in order to set ids in proper
            # order
            for col in range(n_nodes[1]):
                for row in reversed(range(n_nodes[0])):
                    pos_x: float = dw * col 
                    pos_y: float = dh * (n_nodes[1] - row - 1)
                    
                    # Initialize node and put in right position in array
                    self._array[row, col] = Node(pos_x, pos_y)

            # Fill edge_nodes
            self.left_nodes = self._array[:, 0]
            self.right_nodes = self._array[:, -1]
            self.top_nodes = self._array[0, :]
            self.bottom_nodes = self._array[-1, :]
            self.edge_nodes = [self.left_nodes, self.right_nodes,
                               self.top_nodes, self.bottom_nodes]
            
            # Update information in each Node
            # set_edte accept args and interprests it as follows
            # [left, right, top, bottom]
            for en_i, en in enumerate(self.edge_nodes):
                    value = [i == en_i for i in range(4)]
                    for node in en:
                        node.update_edge(*value)
                
        self.shape = self._array.shape

    def get_nodes_surrounding_element(self, element_id: int) -> np.ndarray:
        '''This method returns elements that are neighbours of the
        element which id is passed as argument.'''

        # Get coordinates for elementContainer array based on element id
        # note: self.shape is shape for nodeContainer
        el_x, el_y = Grid.convert_id_to_coord(element_id, self.shape[0] - 1)

        # Calculate node left down corner id:
        node_id = element_id + el_x

        # Get coordinated for nodeContainer array based on node id
        node_x, node_y = Grid.convert_id_to_coord(node_id, self.shape[0])

        # Create helper variables that indicated the interval to slice
        # from nodeContainer
        v_from, v_to = node_y - 1, node_y + 1
        h_from, h_to = node_x, node_x + 2

        return self._array[v_from:v_to, h_from:h_to]

    def get_by_id(self, id: int) -> np.ndarray:
        x, y = Grid.convert_id_to_coord(id, self.shape[0])

        return self[y, x]
    
    def print_nodes(self):
        '''This method prints out id for each node in proper format.'''

        for i in self._array:
            for j in i:
                print(j._id, end=' ')
            print()
    
    def print_all_data(self):
        '''Prints out id, x and y coordinates in proper format.'''

        for i in self._array:
            for j in i:
                print(f"id:{j._id:0>2d}", end='')
                print(f"x:{j.x:0=2.2f}", end='')
                print(f"y:{j.y:0=2.2f} | ", end='')
                
            print()
            print()
    
    def __getitem__(self, pos: tuple):
        if isinstance(pos, int):
            return self._array[pos, :]

        else:
            i, j = pos
            return self._array[i, j]
        

class Element:
    _counter: int = 1
    
    def __init__(self, arg_nodes: NodesContainer) -> None:
        # shape = arg_nodes._array.shape
        self._id: int = Element._counter
        # self.surr_nodes: np.ndarray = arg_nodes.get_nodes_surrouding_element(self._id)
        self.surr_nodes: NodesContainer = \
            NodesContainer(
                n_nodes=(2, 2),
                arg_nodes=arg_nodes.get_nodes_surrounding_element(self._id)
                )
        # [pc1, pc2, pc3, pc4]
        self.H: np.ndarray = np.empty((4, 4, 4))
        self.Hbc: np.ndarray = np.empty((4, 4, 4))

        # Print coordinates which are on edge for each element 
        # print(colored(f"Element {self._id}", 'red', attrs=('bold', )))
        # try:
        #     print('left', [(n.x, n.y) for n in self.surr_nodes.left_nodes])
        # except TypeError:
        #     print('none type, skipping')

        # try:
        #     print('right', [(n.x, n.y) for n in self.surr_nodes.right_nodes])
        # except TypeError:
        #     print('none type, skipping')

        # try:
        #     print('up', [(n.x, n.y) for n in self.surr_nodes.up_nodes])
        # except TypeError:
        #     print('none type, skipping')

        # try:
        #     print('down', [(n.x, n.y) for n in self.surr_nodes.down_nodes])
        # except TypeError:
        #     print('none type, skipping')

        Element._counter += 1
    
    def get_H(self) -> np.ndarray:
        result = np.zeros((4, 4));
        for i in self.H:
            result += i
        
        return result


class ElementsContainer:
    def __init__(self, size: tuple, nodes: NodesContainer) -> None:
        self._array: np.ndarray = np.empty(size, dtype=Element)
        
        for col in range(size[1]):
            for row in reversed(range(size[0])):
                self._array[row, col] = Element(nodes)
        
        self.shape = self._array.shape
    
    def get_by_id(self, id: int) -> np.ndarray:
        x, y = Grid.convert_id_to_coord(id, self.shape[0])

        return self[y, x]

    def get_obj_by_id(self, id: int) -> Element:
        x, y = Grid.convert_id_to_coord(id, self.shape[0])

        return self._array[y, x]


    def print_elements(self):
        '''This method prints out id for each node in proper format.'''

        for i in self._array:
            for j in i:
                print(j._id, end=' ')
            
            print()

    def __getitem__(self, pos: tuple) -> np.ndarray:
        if isinstance(pos, int):
            return self._array[pos, :]

        else:
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


        # Initialization of the nodes
        self.NODES: NodesContainer = NodesContainer(self.get_n_nodes(),
                                                    size=self.get_size())

        # Initialization of the elements. NodesContainer has to be initialized
        # first
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
    
    def get_element_by_id(self, element_id: int) -> np.ndarray:
        return self.ELEMENTS.get_by_id(element_id)
    
    def get_node_by_id(self, node_id: int) -> np.ndarray:
        return self.NODES.get_by_id(node_id)
    
    def get_nodes_surrouding_element(self, element_id: int) -> np.ndarray:
        return self.NODES.get_nodes_surrounding_element(element_id)
    
    def print_nodes(self) -> None:
        self.NODES.print_nodes()

    def print_elements(self) -> None:
        self.ELEMENTS.print_elements()
    
    def get_size_of_element(self):
        w = self.WIDTH / (self.N_NODES_HORIZONTAL - 1)
        h = self.HEIGHT / (self.N_NODES_VERTICAL - 1)

        return h, w
    
    @staticmethod
    def convert_id_to_coord(arg_id: int, height: int):
        x = (arg_id - 1) // height
        y = (arg_id - 1) % height
        y = ((height - 1) - y)

        return x, y

    @staticmethod
    def jakobian(element_id: Union[int, Element], row, J, Jinv,
                 e: Element4p_2D, grid: Union[Grid, None] = None):

        part_N_by_eta = e.get_part_N_by_eta()
        part_N_by_ksi = e.get_part_N_by_ksi()

        NOD: np.ndarray
        if isinstance(element_id, int) and grid:
            NOD = grid.get_element_by_id(element_id).surr_nodes._array
        elif isinstance(element_id, Element):
            NOD = element_id.surr_nodes
        else:
            raise Exception("Invalid argument type for 'element_id'.")

        X, Y = zip(*((v.x, v.y) for v in NOD[[1, 1, 0, 0], [0, 1, 1 ,0]]))
        # X = np.array([NOD[1, 0].x, NOD[1, 1].x, NOD[0, 1].x, NOD[0, 0].x])
        # Y = np.array([0, 0, .025, .025])
        # Y = np.array([NOD[1, 0].y, NOD[1, 1].y, NOD[0, 1].y, NOD[0, 0].y])

        # j = punkt calkowania, czy kolejnosc w part_N_by... jest dobra?
        dxdksi = np.sum(part_N_by_ksi[row] * X)
        dydeta = np.sum(part_N_by_eta[row] * Y)

        dxdeta = np.sum(part_N_by_eta[row] * X)
        dydksi = np.sum(part_N_by_ksi[row] * Y)

        # creatge 2x2 matrix with diagonal created based on the provided list
        J[:, :] = np.diag((dxdksi, dydeta))
        np.fill_diagonal(J[:, ::-1], [-dxdeta, -dydksi])
        Jinv[:, :] = inv(J)
        # print("\nMacierz z x, y na przekatnej")
        # print(M := np.diag((x, y)))

        # print("\nMarcierz Jakobiego:")
        # print(J := inv(M))

        # print("\n1 / det(J) = ")
        # print(1 / det(J))
        # print()

class Mode(Enum):
    ALL = auto()
    OPTION1 = auto()
    OPTION2 = auto()
    OPTION3 = auto()


if __name__ == "__main__":
    
    mode = Mode.OPTION2
    g = Grid(height=18, width=9, nodes_vertiacl=4, nodes_horizontal=4)
    # g = Grid(height=10, width=5, nodes_vertiacl=7, nodes_horizontal=7)

    # print("Printing all nodes ids:")
    # g.NODES.print_nodes()
    # print("\nPrinting all elements ids:")
    # g.ELEMENTS.print_elements()
    if mode == Mode.OPTION1:
        printer.log(g, mode={'id': 'ne', 'coor': 'en', 'nofe': '1'})

    # TODO: refector this to make this OO
    # Vars that will be overriden each iteration
    if mode == Mode.OPTION2:
        printer.log(g, mode={'coor': 'e'})
        Jak = np.empty((2, 2))
        Jak_inv = np.empty((2, 2))
        e1:Element4p_2D =  Element4p_2D(g.get_size_of_element())
        
        # part_N_by_x = np.empty((4, 4))
        # part_N_by_y = np.empty((4, 4))

        for element_id in range(g.N_ELEMENTS_TOTAL):
            # j liczba punktow calkowania
            for j in range(4):
                # Initialize part_N_by_x and part_N_by_y in Element4p_2D object
                # calculate jakobian for each element
                e1.calc_derivatives_global_coordinates(element_id, j, g)
            
            element: Element = g.ELEMENTS.get_obj_by_id(element_id)
            for j in range(4):
                # Calculate matrix matrix H for each element
                integral_function = \
                    lambda: (e1.get_part_N_x()[j][:, np.newaxis] * e1.get_part_N_x()[j] +
                            e1.get_part_N_y()[j][:, np.newaxis] * e1.get_part_N_y()[j])

                g.ELEMENTS.get_obj_by_id(element_id).H[j] = \
                    (K * integral_function() * det(e1.J))
                
                # FIXME: This is bad:
                # Calculate matrix Hbc for each element
                size = it.cycle(g.get_size_of_element()) # returns (height, width)
                g.ELEMENTS.get_obj_by_id(element_id).Hbc[j] = \
                    e1._Hbc[j] * (next(size) / 2 * ALPHA)
            
                
            # g.ELEMENTS.get_obj_by_id(i).Hbc[0] = \
            #     e1._H_left * g.HEIGHT/(g.N_NODES_VERTICAL - 1)/2
            # g.ELEMENTS.get_obj_by_id(i).Hbc[1] = \
            #     e1._H_bottom * g.WIDTH/(g.N_NODES_HORIZONTAL - 1)/2
            # g.ELEMENTS.get_obj_by_id(i).Hbc[2] = \
            #     e1._H_right * g.HEIGHT/(g.N_NODES_VERTICAL - 1)/2
            # g.ELEMENTS.get_obj_by_id(i).Hbc[3] = \
            #     e1._H_up * g.WIDTH/(g.N_NODES_HORIZONTAL - 1)/2

            element: Element = g.get_element_by_id(element_id)
            if isinstance(element.surr_nodes.left_nodes, np.ndarray):
                ksi = -1, -1
                eta = Element4p_2D._L[0][::-1]
                N_pc1 = Element4p_2D.N[0](ksi[0], eta[0]), Element4p_2D.N[3](ksi[0], eta[0])
                N_pc2 = Element4p_2D.N[0](ksi[1], eta[1]), Element4p_2D.N[3](ksi[1], eta[1])
                print(f"Left edge: N1_pc1: {N_pc1}, N4_pc2: {N_pc2}")
                

            print(colored(f"Elements no. {element_id+1} H value:",
                          "red", attrs=('bold', 'underline', )))
            print(g.ELEMENTS.get_obj_by_id(element_id).get_H())

        # print Hbc for each element
        dirs = ('left', 'right', 'up', 'down')
        print()
        for c, _Hbc in enumerate(element.Hbc):
            print(colored(f"Element no. {c+1} {dirs[c]} Hbc matrix:", 'red'))
            print(_Hbc)
            print()
        
        print()

        # print(e1._H)

        # e1.show_results()
                # print(part_N_by_x)
                # print(part_N_by_y)
                # print(w[j][np.newaxis])
                # print(Jak)
                # print((1/det(Jak) * Jak_inv)@w[j])
        # print(part_N_by_ksi)
        # print(part_N_by_eta)
        # print(part_N_by_x)
        # print(part_N_by_y)


    # print("\nPrinting all nodes with coordinates:")
    # g.NODES.print_all_data()

    # print(f"Printing neighbour nodes for element no.: {(e := 5)}:")
    # g.ELEMENTS.get_by_id(e).surr_nodes.print_all_data()

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

    if mode == Mode.OPTION3:
        X = np.array([0, .025, 0, 0.025])
        Y = np.array([.025, .025, 0, 0])

        nodes: np.ndarray = np.array([Node(arg_x=x, arg_y=y) for x, y in zip(X, Y)]).reshape((2, 2))
        nodes_container = NodesContainer(n_nodes=(2, 2), arg_nodes=nodes)    
        nodes_container.print_all_data()
        element = Element(nodes_container)
        element.surr_nodes = nodes

        e1 = Element4p_2D()
        for row in range(4):
            Grid.jakobian(element, row, e1.J, e1.Jinv, e1)
        e1.show_results()