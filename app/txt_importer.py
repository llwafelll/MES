import os
import re
import numpy as np

class DataLoader:
    BASEDIR = os.path.dirname(__file__)
    
    def __init__(self, filename):
        self.filename = filename
        self.abspath = os.path.join(self.BASEDIR, filename)

        self.number_of_nodes = None
        self.data = dict()

    def _load_nodes(self):
        regex_consts = r"^((?:[a-zA-z]+\s)+)(\d+)"
        regex_nodes = r"(\d+),\s*(\d\.\d*),\s*.(\d\.\d*)"
        regex_elements = r"^\s(\d+),\s+((?:\d+(?:,\s+|\d+\s)+)+)"

        with open(self.abspath) as file:
            lines = file.readlines()
            for line in lines:
                if (find := re.search(regex_consts, line)):
                    self.data.update({find.group(1).strip(): find.group(2).strip()})
                else:
                    break

            nodes_num = int(self.data["Nodes number"])
            nodes = np.empty((nodes_num, 2))
            self.data.update({"Nodes": nodes})
            
            elem_num = int(self.data["Elements number"])
            elements = np.empty((elem_num, int(np.sqrt(nodes_num))))
            self.data.update({"Elements": elements})
            
            for line in lines:
                # self.data.update({"Elements": np.empty((self.data["Elements number"], 2))})

                if (find := re.search(regex_nodes, line)):
                    row = int(find.group(1)) - 1
                    self.data["Nodes"][row][0] = float(find.group(2))
                    self.data["Nodes"][row][1] = float(find.group(3))
                
                if (find := re.search(regex_elements, line)):
                    row = int(find.group(1)) - 1
                    rows_elements = [int(i) for i in find.group(2).split(",")]
                    self.data["Elements"][row] = rows_elements
        pass

if __name__ == '__main__':
    dl = DataLoader("Test1_4_4.txt")
    dl._load_nodes()