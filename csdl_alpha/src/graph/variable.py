from csdl_alpha.src.graph.node import Node
import numpy as np


class Variable(Node):
    is_input = True
    is_implicit = False

    def __init__(self, shape: tuple, 
                 name: str = None, 
                 value: np.ndarray = None,  
                 tags: list[str] = [], 
                 hierarchy: int = None):
        from csdl_alpha.api import manager
        recorder = manager.active_recorder
        recorder._add_node(self) # sets namespace and index
        
        self.shape = shape
        self.name = name
        self.value = value
        self.tags = tags
        self.hierarchy = hierarchy

