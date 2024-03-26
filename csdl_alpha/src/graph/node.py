from dataclasses import dataclass
from csdl_alpha.src.recorder import Namespace

class Node(object):
    """
    Represents a node in a graph.

    Attributes:
        namespace (Namespace): The namespace of the node.
    """
    namespace: Namespace = None

    def __init__(self) -> None:
        # recorder object
        import csdl_alpha
        self.recorder = csdl_alpha.get_current_recorder()
        self.recorder._set_namespace(self)

        

    # def __eq__(self, other):
    #     return self is other
    # def __hash__(self):
    #     return id(self)