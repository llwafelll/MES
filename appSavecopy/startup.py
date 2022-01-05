# TODO: find metaclass that support struct-like classes in python
# https://www.youtube.com/watch?v=vBH6GRJ1REM
# TODO: matplotlib to visualize

from __future__ import annotations

import itertools as it
from enum import Enum, auto
from typing import Type, Union

import numpy as np
from numpy.linalg import det, inv
from termcolor import colored

from constants import *
from custor_print_colored import print_H1, print_H2, print_H3
from logger import *
from matrix_partial_copy import Element4p_2D

# TODO: Think about changing datatype of surr_nodes to NodesContainer
# remember to keep nodes as references

# K = 30
# ALPHA = 25
# T_o = 1200

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
        self.t_0 = t_0

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
        """Overrides values in edge dict. For internal usage only"""
        if kwargs:
            for k, v in kwargs.items():
                self.edge[k] = v
            
        elif args:
            for k, v in zip(self.edge.keys(), args):
                self.edge[k] = v

    def update_edge(self, *args, **kwargs):
        """For internal usage only"""
        if kwargs:
            for k, v in kwargs.items():
                self.edge[k] += v
            
        elif args:
            for k, v in zip(self.edge.keys(), args):
                self.edge[k] += v
    
    def is_edge(self):
        return any(self.edge.values())
    

class NodesContainer:
    def __init__(self, n_nodes: tuple, size: tuple = None,
                 arg_nodes: np.ndarray = np.array([])) -> None:
        '''
        ARGS:
        n_nodes - number of nodes, respectively, in y and x axis
                  tuple -> N_NODES_VERTICAL, N_NODES_HORIZONTAL
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

        # GENERATE THE GRID:
        # Creates entirely new array of nodes (from scratch).
        # The width and height is divided by number of nodes then each node
        # received it's coordinates.
        # This procedure should be executed only once while initializing
        # a new grid.
        elif size:

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

    def _get_nodes_surrounding_element(self, element_id: int) -> np.ndarray:
        '''This method returns elements that are neighbours of the
        element which id is passed as argument.
        This method is used by NodesContainer constuctor'''

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

    def get_by_id(self, id: int) -> Node:
        """Returns np.ndarray of Node elements."""
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

    @staticmethod
    def calc_dist(vector: np.ndarray) -> float:
        '''Calculate distance between two nodes passed as np.ndarray (2, )'''
        return np.sqrt((vector[0].x - vector[1].x)**2 +\
                    (vector[0].y - vector[1].y)**2)
    
    def __getitem__(self, pos: tuple) -> Union[np.ndarray, Node]:
        if isinstance(pos, int):
            return self._array[pos, :]

        else:
            i, j = pos
            return self._array[i, j]
        

class Element:
    _counter: int = 1
    mask = None # static field for elements mask
    # agregation_matrix = np.zeros((4, 4, 2)) # FIXME: Delete?
    
    def __init__(self, arg_nodes: NodesContainer) -> None:
        '''
        ARGS:
        arg_nodes - NodesContainer which holds all nodes in the grid
        '''

        if not Element.mask:
            raise Exception("Element.mask cannot be NoneType.")

        self._id: int = Element._counter
        self.is_edge = False

        # Initialize surr_nodes which holds nodes surrouding element with
        # given id. n_nodes in NodesContainer is at fixed size (2, 2)
        _surr_nodes = arg_nodes._get_nodes_surrounding_element(self._id)
        self.surr_nodes: NodesContainer = \
            NodesContainer(n_nodes=(2, 2), arg_nodes=_surr_nodes)
        self._surr_nodes_as_list = \
            np.hstack([self.surr_nodes[1, :], self.surr_nodes[0, ::-1]])

        self.jacobians = self.calc_jacobians()

        # Initialize H matrix (4x4 matrix for each integration point)
        # [pc1, pc2, pc3, pc4]
        self.Hpc = np.zeros((4, 4, 4))
        # self._calc_H_matrix()

        self.C_matrices = np.zeros((4, 4, 4))
        # self._calc_C_matrices()

        # Initialize Hbc (H on boundaries) matrix (4x4 matrix
        # for each integration point)
        self.Hbc: np.ndarray = np.zeros((4, 4, 4))
        # self._calc_Hbc_matrix()
        
        self.Pvectors = np.zeros((4, 4))
        # self._calc_P_vector()

        Element._counter += 1
    
    def calculate(self):
        self.Hpc = np.zeros((4, 4, 4))
        self.C_matrices = np.zeros((4, 4, 4))
        self.Hbc = np.zeros((4, 4, 4))
        self.Pvectors = np.zeros((4, 4))

        self._calc_H_matrix()
        self._calc_C_matrices()

        # Boudary condition
        self._calc_Hbc_matrix()
        self._calc_P_vector()
    
    # FIXME: Delete?
    # def _update_local_agregation_matrix(self):
    #     element_ids = np.array([n._id for n in self._surr_nodes_as_list])
    #     a = np.tile(element_ids, (4, 1))
    #     b = np.tile(element_ids[:, np.newaxis], (1, 4))
        
    #     matrix = np.stack((b, a), axis=2)
    #     Element.agregation_matrix = matrix
    
    # def equ(self):
    #     T_1 = 0
    #     T_0 = 1
    #     self.get_H() * T_1 + self.get_C_matrix() * (T_1 - T_0)/(delta_T) + self.get_P_vector()

    def _calc_distances(self):
        '''Return array of distances between nodes on located on the same edge,
        such that [left, right, top, bottom] edge.'''

        # NodeContainer.calc_dist accept as argument np.ndarray as argument
        # The array consists of 2 elements both of them are Node objects
        return np.array([NodesContainer.calc_dist(self.surr_nodes[:, 0]),
                         NodesContainer.calc_dist(self.surr_nodes[:, 1]),
                         NodesContainer.calc_dist(self.surr_nodes[0, :]),
                         NodesContainer.calc_dist(self.surr_nodes[1, :]),
                         ])

    def _calc_H_matrix(self):
        """Updates H matrix in the element object. H object is NxN matrix where
        each column is integration point and column represents each shape funcion"""

        mat = np.transpose(np.squeeze(self._calc_gradN(), axis=3), (2, 0, 1))
        # np.transpose(np.squeeze(self._calc_gradN(), axis=3), (0, 2, 1))

        mat = np.stack((mat[0], mat[1]), axis=1)
        
        # for i, 
        mat = \
            mat[:, 0][:, :, np.newaxis] * mat[:, 0][:, np.newaxis] + \
            mat[:, 1][:, :, np.newaxis] * mat[:, 1][:, np.newaxis]

        self.Hpc = K * mat * det(self.jacobians)

        # for i, (gradN, j) in enumerate(zip(self._calc_gradN(),
        #                                    self.calc_jacobians())):
        #     gradN = gradN.T.reshape((2, 4))
        #     self.Hpc[i] = (det(j) * K * \
        #                 (gradN[0][:, np.newaxis] * gradN[0] + \
        #                  gradN[1][:, np.newaxis] * gradN[1])
        #                 )

    def get_Hpc(self) -> np.ndarray:
        """Sum all Hpcx and return H matrix."""
        return self.Hpc.sum(axis=0)

    def _calc_C_matrices(self):
        for pc, J in enumerate(self.jacobians):
            # self.C_matrices[pc] = Element4p_2D.N_matrix[pc][:, np.newaxis] * Element4p_2D.N_matrix[pc] * det(J) * C_p * rho
            # self.C_matrices[pc] = Element4p_2D._C_matrix[pc] * det(J)
            self.C_matrices[pc] = Element4p_2D.N_matrix[pc, :, np.newaxis] * \
                                  Element4p_2D.N_matrix[pc, :] * C_p * rho * det(J)
        # for i, J in enumerate(self.jacobians):
        #     self.C_matrices[i] = Element4p_2D.pre_C_matrix * det(J)
    
    def get_C_matrix(self):
        return np.sum(self.C_matrices, axis=0)
    
    def _calc_Hbc_matrix(self):
        nc: NodesContainer
        distances = self._calc_distances()
        nodes_pairs = self.surr_nodes.edge_nodes

        for i, surface in enumerate(Element4p_2D.N_surf):
            if isinstance(nodes_pairs[i], np.ndarray):
                for row, wage in zip(surface, Element4p_2D.ws):
                    self.Hbc[i] += wage * row[:, np.newaxis] * row

                self.Hbc[i] = self.Hbc[i] * ALPHA * distances[i]/2

        # print()
        # for i, (Nsurf, dist) in enumerate(zip(Element4p_2D.N_surf, self._calc_distances())):
        #     for row, wage in zip(Nsurf, Element4p_2D.ws):
        #         self.Hbc[i] += wage * row[:, np.newaxis] * row
            
        #     self.Hbc[i] = self.Hbc[i] * ALPHA * dist/2
       

        # for i, dist in enumerate(self._calc_distances()):
        #     self.Hbc[i] = ALPHA * dist/2 * Element4p_2D._Hbc[i]

    def get_Hbc(self) -> np.ndarray:
        """Sum all pbcx and return H matrix."""
        # _sum = np.zeros((4, 4))
        # for i, edge in enumerate(self.surr_nodes.edge_nodes):
        #     if isinstance(edge, np.ndarray):
        #         _sum += self.Hbc[i]

        # return _sum
        return self.Hbc.sum(axis=0)
    
    def get_H(self) -> np.ndarray:
        return self.Hbc() + self.Hpc

    
    def _calc_P_vector(self):

        distances = self._calc_distances()
        nodes_pairs = self.surr_nodes.edge_nodes

        for i, surface in enumerate(Element4p_2D.N_surf):
            if isinstance(nodes_pairs[i], np.ndarray):
                for row, wage in zip(surface, Element4p_2D.ws):
                    self.Pvectors[i] += wage * row

                self.Pvectors[i] = self.Pvectors[i] * ALPHA * T_o * distances[i]/2

        # print()

        # for i, (Nsurf, dist) in enumerate(zip(Element4p_2D.N_surf, self._calc_distances())):
        #     for row, wage in zip(Nsurf, Element4p_2D.ws):
        #         self.Pvectors[i] += wage * row * T_o
            
        #     self.Pvectors[i] = self.Pvectors[i] * ALPHA * dist/2

        # for i, edge in enumerate(self.surr_nodes.edge_nodes):
        #     if isinstance(edge, np.ndarray):

        #         dist = NodesContainer.calc_dist(edge)
        #         self.Pvectors[i] += dist / 2 * T_o * ALPHA * Element4p_2D._Pvector[i]
            


        # print
        # print(colored(f"P vectors for {self._id}", 
        #               "white", "on_red", attrs=("bold", )))
        # for p in self.Pvectors:
        #     for v in p:
        #         print(colored(f"{v:7.3f}", "cyan"), end=" | ")
        #     print()
        # print()
        # print(self.get_P_vector())

                
        # for pc, dist in enumerate(self._calc_distances()):
        #     self.P[pc] = dist / 2 * ALPHA * Element.mask._Pvector[pc]
    
    def get_P_vector(self):
        '''P vector is sum of all P vectors (each P vector is for each edge)'''
        return np.sum(self.Pvectors, axis=0)
        # _sum = np.zeros((4, 4))
        # for i, edge in enumerate(self.surr_nodes.edge_nodes):
        #     if isinstance(edge, np.ndarray):
        #         _sum[i] = self.Pvectors[i]

        # return _sum
        
    def _calc_gradN(self):
        """Returns 4 element array (gradN) containing Nx2 arrays (each Nx2 array
        is dedicated for integration point). Each Nx2 is array such that:
        N1
        [[[dN1/dx], [[dN2/dx], [[dN3/dx],
          [dN1/dy]], [dN2/dy]], [dN3/dy]], ...]
        """

        gradN = np.empty((4, 4, 2, 1))

        for pc, jac in zip(range(Element4p_2D.Npcs), self.jacobians):

            # [[dNi/deta],  this is loc_gradN
            #  [dNi/dksi]]
            loc_gradN = np.stack(
                                 (Element4p_2D.dNdksi[pc],
                                  Element4p_2D.dNdeta[pc]), axis=1
                                 )[:, :, np.newaxis]

            gradN[pc] = inv(jac) @ loc_gradN

        return gradN


    
    def calc_jacobian(self, pc: int) -> np.ndarray:
        '''Calculate Jacobian for given integration point - pc for this element.
        '''

        # x, y = self._surr_nodes_as_list[pc].get_position()
        x, y = zip(*(n.get_position() for n in self._surr_nodes_as_list))
        
        # dxdksi = np.sum(Element.mask.part_N_by_ksi[pc] * x)
        # dydeta = np.sum(Element.mask.part_N_by_eta[pc] * y)

        # Ze wzoru na interpolacje
        # dx/dKsi = dN1/dKsi * x1 + dN2/dKsi * x2 + ...
        dxdksi = np.sum(Element4p_2D.dNdksi[pc] * x)
        dydeta = np.sum(Element4p_2D.dNdeta[pc] * y)
        
        # dxdeta = np.sum(Element.mask.part_N_by_eta[pc] * x)
        # dydksi = np.sum(Element.mask.part_N_by_ksi[pc] * y)
        dxdeta = np.sum(Element4p_2D.dNdeta[pc] * x)
        dydksi = np.sum(Element4p_2D.dNdksi[pc] * y)
        
        return np.array([[dxdksi, dydksi],
                         [dxdeta, dydeta]])

    def calc_jacobians(self):
        '''Calculate Jacobian for all integration points in a element'''

        J = []

        for i in range(Element4p_2D.Npcs):
            J.append(self.calc_jacobian(i))
        
        return J
    
    
    
    def get_surr_nodes_ids(self):
        return [n._id - 1 for n in self._surr_nodes_as_list]
    
    @classmethod
    def set_mask(cls, mask: Element4p_2D):
        Element.mask = mask


class ElementsContainer:
    def __init__(self, n_elements: tuple, nodes: NodesContainer) -> None:
        '''
        ARGS:
        n_elements - number of elements, respectively, in y and x axis
                   - equivalent to nodes._array.shape - 1
        nodes - all nodes of the grid
        '''

        # Crucial attribute of the class, while initialized should contain elements
        self._array: np.ndarray = np.empty(n_elements, dtype=Element)
        
        # Initialization fo the self._array
        for col in range(n_elements[1]):
            for row in reversed(range(n_elements[0])):
                
                # Elements constructor creates proper surr_nodes based on
                # elements id
                self._array[row, col] = Element(nodes)

        
        self.shape = self._array.shape
    
    def get_by_id(self, id: int) -> np.ndarray:
        '''Return element for element with given id.'''

        x, y = Grid.convert_id_to_coord(id, self.shape[0])

        return self[y, x]

    def print_elements(self):
        '''This method prints out id for each node in proper format.'''

        for i in self._array:
            for j in i:
                print(j._id, end=' ')
            
            print()

    def __getitem__(self, pos: tuple) -> Union[np.ndarray, Element]:
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


        ny, nx = self.get_n_nodes()
        self.H_AGREGATION_MATRIX = np.zeros((ny * nx, ny * nx))
        self.Hbc_AGREGATION_MATRIX = np.zeros((ny * nx, ny * nx))
        self.C_AGREGATION_MATRIX = self.H_AGREGATION_MATRIX.copy()
        self.P_AGREGATION_MATRIX = np.zeros((ny * nx))

        # Initialization of the nodes
        self.NODES: NodesContainer = NodesContainer(self.get_n_nodes(),
                                                    size=self.get_size())

        # Initialization of the elements. NodesContainer has to be initialized
        # first
        self.ELEMENTS: ElementsContainer = \
            ElementsContainer(self.get_n_elements(), self.NODES)
        
        
    def fill_agregation_matrixes(self):
        self.H_AGREGATION_MATRIX = np.zeros((ny * nx, ny * nx))
        self.Hbc_AGREGATION_MATRIX = np.zeros((ny * nx, ny * nx))
        self.C_AGREGATION_MATRIX = self.H_AGREGATION_MATRIX.copy()
        self.P_AGREGATION_MATRIX = np.zeros((ny * nx))
        # Fill global agregate matrix with data
        for element_id in range(1, self.N_ELEMENTS_TOTAL + 1):
            element = self.get_element_by_id(element_id)
            element.calculate()

            # Agregation matrix
            local_agregate = it.product(element.get_surr_nodes_ids(), repeat=2)
            H_indices = it.product(range(4), repeat=2)

            for (i, j), (ax, ay) in zip(H_indices, local_agregate):
                self.H_AGREGATION_MATRIX[ax, ay] += element.get_Hpc()[i, j]

            local_agregate = it.product(element.get_surr_nodes_ids(), repeat=2)
            H_indices = it.product(range(4), repeat=2)

            for (i, j), (ax, ay) in zip(H_indices, local_agregate):
                self.Hbc_AGREGATION_MATRIX[ax, ay] += element.get_Hbc()[i, j]
            
            local_agregate = it.product(element.get_surr_nodes_ids(), repeat=2)
            C_indices = it.product(range(4), repeat=2)
            for (i, j), (ax, ay) in zip(C_indices, local_agregate):
                self.C_AGREGATION_MATRIX[ax, ay] += element.get_C_matrix()[i, j]
            
            # P agregation vector
            for i, a in enumerate(element.get_surr_nodes_ids()):
                self.P_AGREGATION_MATRIX[a] += element.get_P_vector()[i]
        
    # Getters
    def get_size(self):
        '''Returns size of the metal element.'''

        return (self.HEIGHT, self.WIDTH)
    
    def get_n_nodes(self):
        '''Returns the number of the nodes that construct the grid in a tuple.'''

        return (self.N_NODES_VERTICAL, self.N_NODES_HORIZONTAL)
    
    def get_n_elements(self):
        '''Returns the number of the elements that construct the grid in a tuple.'''

        return (self.N_NODES_VERTICAL - 1, self.N_NODES_HORIZONTAL - 1)

    def get_size_of_element(self):
        '''Returns real size of a single element in metres.'''
        w = self.WIDTH / (self.N_NODES_HORIZONTAL - 1)
        h = self.HEIGHT / (self.N_NODES_VERTICAL - 1)

        return h, w
    
    # Shortcuts 
    def get_element_by_id(self, element_id: int) -> Element:
        return self.ELEMENTS.get_by_id(element_id)
    
    def get_node_by_id(self, node_id: int) -> Node:
        return self.NODES.get_by_id(node_id)
    
    # Printing methods
    def print_nodes(self) -> None:
        self.NODES.print_nodes()

    def print_elements(self) -> None:
        self.ELEMENTS.print_elements()
    
    def print_agregation_matrix(self, matrix) -> None:
        """Print agregation matrix"""

        prec = 2
        n = len(str(int(matrix.max()))) + prec + 1 # add one because of "."
        for row in matrix:
            for col in row:
                print(f"{col:0{n}.{prec}f}", end="  ")
            print()
        
    
    @staticmethod
    def convert_id_to_coord(arg_id: int, height: int):
        '''
        Returns x and y coordinates based on id of element/node

        ARGS:
        arg_id - id of element/node
        height - number of elements in y axis

        EXAMPLE:
        self.convert_id_to_coord(5, 3) -> (1, 1)
        self.convert_id_to_coord(5, 4) -> (1, 3)
        '''
        
        x = (arg_id - 1) // height
        y = (arg_id - 1) % height
        y = ((height - 1) - y)

        return x, y

    @staticmethod
    def jakobian(element_id: Union[int, Element], row, J, Jinv,
                 e: Element4p_2D, grid: Union[Grid, None] = None):

        # part_N_by_eta = e.get_part_N_by_eta()
        # part_N_by_ksi = e.get_part_N_by_ksi()
        part_N_by_eta = Element4p_2D.dNdksi
        part_N_by_ksi = Element4p_2D.dNdeta

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
    OPTION4 = auto()


if __name__ == "__main__":
    
    # Mode selection
    mode = Mode.OPTION4

    # Create universal element
    e1: Element4p_2D = Element4p_2D()
    Element.set_mask(e1)

    # Grid initialization
    g = Grid(height=H, width=B, nodes_vertiacl=N_H, nodes_horizontal=N_B)
    # g = Grid(height=10, width=5, nodes_vertiacl=7, nodes_horizontal=7)

    # ==== THE PROGRAM ====
    if mode == Mode.OPTION1:
        printer.log(g, mode={'id': 'ne', 'coor': 'en', 'nofe': '1'})
    

    if mode == Mode.OPTION4:
        print(); print_H1("Grid:")
        printer.log(g, mode={'coor': 'e'})

        ny, nx = g.get_n_nodes()
        # TU MA BYC TA PETLA!!!!!!!!
        for iter_no in range(10):
            print(f"\nITERATION {iter_no}\n")
            g.fill_agregation_matrixes()

        # temps = np.full((ny * nx, ), t_0)

            print_H1("H+C/dT")
            M = g.H_AGREGATION_MATRIX + g.Hbc_AGREGATION_MATRIX + g.C_AGREGATION_MATRIX/dT
            # g.print_agregation_matrix(M)

            b = np.zeros((ny * ny, ))
            print_H1("P+C/dT * t_0")
            temperatures = np.zeros((ny * ny,))
            for i in range(ny * ny):
                temperatures[i] = g.get_node_by_id(i).t_0

            # for i in range(ny * ny):
                # b[i] = g.P_AGREGATION_MATRIX[i] + \
                #             np.sum(g.C_AGREGATION_MATRIX[i] / dT) * g.get_node_by_id(i).t_0
            b = g.P_AGREGATION_MATRIX + np.squeeze((g.C_AGREGATION_MATRIX/dT) @ temperatures[:, np.newaxis])
            # b = g.P_AGREGATION_MATRIX + \
            #                           g.C_AGREGATION_MATRIX/dT @ temps
            # print(b)

            print_H1("equation solution")
            x = np.linalg.solve(M, b)
            # print(x)
            print(f"min = {x.min()}")
            print(f"max = {x.max()}")
            for i in range(ny * ny):
                g.get_node_by_id(i).t_0 = x[i]


            # print_H1("N")
            # g.print_agregation_matrix(Element4p_2D.N_matrix)

            # print_H1("dNdksi")
            # g.print_agregation_matrix(Element4p_2D.dNdksi)

            # print_H1("dNdeta")
            # g.print_agregation_matrix(Element4p_2D.dNdeta)

            # print_H1("H agregation matrix:")
            # g.print_agregation_matrix(g.H_AGREGATION_MATRIX)

            # print_H1("Hbc agregation matrix:")
            # g.print_agregation_matrix(g.Hbc_AGREGATION_MATRIX)

            # print_H1("C agregation matrix:")
            # g.print_agregation_matrix(g.C_AGREGATION_MATRIX)

            # print_H1("P vector agregation matrix")
            # print(g.P_AGREGATION_MATRIX)

        assert False
        print()

        print_H1("Values for each element")
        # Iterate over each element in the grid
        for element_id in range(1, g.N_ELEMENTS_TOTAL + 1):
            element: Element = g.ELEMENTS.get_by_id(element_id)

            # print(colored(f"Element id: {element_id}", 'red', attrs=('bold', )))
            print_H2(f"Element id: {element_id}")

            print_H3(f"Hpc matrix for element:")
            print(element.Hpc)

            print_H3(f"H matrix for element")
            print(element.get_H())

            print_H3(f"Hbc matrix for element")
            print(element.Hbc)

            print_H3(f"P vector")
            print(element.Pvectors)

            print_H3(f"C matrices for each Hpc")
            print(element.C_matrices)

            print_H3("C matrix")
            print(element.get_C_matrix())
            
            # Print out jacobian for each node of the element
            print_H3(f"Jacobians")
            for c, j in enumerate(element.calc_jacobians()): # returns array of 4 elements each of them is jacobian
                print(colored(f"pc no. {c + 1}", 'red'))
                print(inv(j))
                # print(inv(j))

            # print(element.get_Hbc())
        