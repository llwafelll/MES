# TODO: find metaclass that support struct-like classes in python
# TODO: matplotlib to visualize

# TODO: introduce NodeList?
class Node:
    x: float = None
    y: float = None

# TODO: introduce ElementList?
class Element:
    id: list[int] = list()

class Grid:
    # height and width for whole square
    HEIGHT: float = None
    WIDTH: float = None

    # number of nodes vertically and horizontally
    N_NODES_HEIGHT: int = None
    N_NODES_WIDTH: int = None

    # total number of nodes and elements
    N_NODES_TOTAL: int = None
    N_ELEMENTS_TOTAL: int = None

    # references to the objects?
    NODES: list = None
    ELEMENTS: list = None