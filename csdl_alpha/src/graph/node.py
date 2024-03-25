from dataclasses import dataclass
from csdl_alpha.src.recorder import Namespace

@dataclass
class Node:
    """
    Represents a node in a graph.

    Attributes:
        namespace (Namespace): The namespace of the node.
    """
    namespace: Namespace = None