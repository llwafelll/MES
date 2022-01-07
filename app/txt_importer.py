import os
import re
import numpy as np


class DataLoader:
    """Provides method 'load_data' which initializes 'self.data' dictionary.
    'self.data' dictionary contais all data stored in provided file."""
    BASEDIR = os.path.dirname(__file__)
    RELATIVE_PATH = os.path.join(".", "tests")
    
    def __init__(self, filename=None):
        if filename:
            self.change_file(filename)
        else:
            self.filename = None
            self.abs_path = None
            self.data = None
        # self.filename = filename
        # self.abs_path = os.path.join(self.BASEDIR, self.RELATIVE_PATH, filename)

        # self.data = dict()
    
    def change_file(self, new_filename):
        self.filename = new_filename
        self.abs_path = os.path.join(self.BASEDIR, self.RELATIVE_PATH, new_filename)

        self.data = dict()

    def load_data(self):
        regex_consts = r"^((?:[a-zA-z]+\s)+)(\d+)"
        regex_nodes = r"(\d+),\s*?(\-?\d\.\d*),\s*?.(\-?\d\.\d*)"
        regex_elements = r"^\s(\d+),\s+((?:\d+(?:,\s+|\d+\s)+)+)"
        regex_boundary_cond = r"^\b(?: ?\d+,?)+"

        with open(self.abs_path) as file:
            lines = file.readlines() # get list of strings of each line

            # Load constants from file
            for line in lines:
                if (find := re.search(regex_consts, line)):
                    self.data.update({find.group(1).strip(): float(find.group(2).strip())})
                else:
                    break

            # Declare numpy arrays for nodes and elements
            nodes_num = int(self.data["Nodes number"])
            nodes = np.empty((nodes_num, 2))
            self.data.update({"Nodes": nodes})

            temp = np.empty((nodes_num, 2)) # todel
            self.data.update({"Temp": temp}) # todel
            
            elem_num = int(self.data["Elements number"])
            elements = np.empty((elem_num, 4))
            self.data.update({"Elements": elements})
            
            # Initialize nodes and elements numpy arrays
            for line in lines:
                if (find := re.search(regex_nodes, line)):
                    row = int(find.group(1)) - 1
                    self.data["Nodes"][row][0] = float(find.group(2))
                    self.data["Nodes"][row][1] = float(find.group(3))
                    self.data["Temp"][row][0] = int(find.group(1)) # todel
                    self.data["Temp"][row][1] = int(find.group(1)) # todel
                
                if (find := re.search(regex_elements, line)):
                    row = int(find.group(1)) - 1
                    rows_elements = [int(i) for i in find.group(2).split(",")]
                    self.data["Elements"][row] = rows_elements
                
                if (find := re.search(regex_boundary_cond, line)):
                    bc = np.array([int(i) for i in find.group(0).split(",")])
                    self.data.update({"BC": bc})
            
        pass

dl = DataLoader()

if __name__ == '__main__':
    from startup import NodesContainer, Node

    # Initialize & load data
    dl = DataLoader("Test1_4_4.txt")
    dl.load_data()

    # Change matrix view to convert numbering convention from applien in read
    # file to one used in this program
    nodes = dl.data["Nodes"] = dl.data["Nodes"] \
                                .reshape((4, 4, 2)) \
                                .transpose((1, 0, 2))[::-1, ::-1, :]

    # Declaer matrix to store Node class instances and pass them to NodesContainer
    nodesObjs = np.empty((nodes.shape[0], nodes.shape[1]), dtype=Node)
    
    # Initialize matrix by creating Node cls instances and assigning them
    for i, row in enumerate(nodes):
        for j, col in enumerate(row):
            nodesObjs[len(row) - j-1, i] = Node(col[0], col[1])
    
    edge_noes = [
        nodesObjs[:, 0], nodesObjs[:, -1], # left, right
        nodesObjs[0, :], nodesObjs[-1, :], # bottom, top
    ]
    keys = ("is_left", "is_right", "is_top", "is_bottom") # you can use node object to get the data

    for es, k in zip(edge_noes, keys):
        for e in es:
            e.update_edge(**{k: True})

    # Update information about node's edges
    # for _node in nodesObjs[0, :]:
    #     _node.update_edge(False, False, True, False)
    # for _node in nodesObjs[-1, :]:
    #     _node.update_edge(False, False, False, True)
    # for _node in nodesObjs[:, 0]:
    #     _node.update_edge(True, False, False, False)
    # for _node in nodesObjs[:, -1]:
    #     _node.update_edge(False, True, False, False)

    # Pass created matrix with nodes to NodesContainer contructor
    nc = NodesContainer((4, 4), arg_surr_nodes=nodesObjs)

    # Printing result
    print(dl.data["Nodes"])
    pass
    
