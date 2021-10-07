# TODO: find metaclass that support struct-like classes in python
# https://www.youtube.com/watch?v=vBH6GRJ1REM
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
    N_NODES_VERTIACAL: int = None
    N_NODES_HORIZONTAL: int = None

    # total number of nodes and elements
    N_NODES_TOTAL: int = None
    N_ELEMENTS_TOTAL: int = None

    # references to the objects?
    NODES: list = None
    ELEMENTS: list = None