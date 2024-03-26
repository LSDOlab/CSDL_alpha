from dataclasses import dataclass
from csdl_alpha.src.recorder import Namespace

class Node:
    """
    Represents a node in a graph.

    Attributes:
        namespace (Namespace): The namespace of the node.
    """
    namespace: Namespace = None

    # def __eq__(self, other):
    #     return self is other
    # def __hash__(self):
    #     return id(self)