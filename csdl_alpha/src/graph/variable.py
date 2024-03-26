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
        
        super().__init__()
        self.recorder._add_node(self) # sets namespace and index
        
        self.shape = shape
        self.name = name
        self.value = value
        self.tags = tags
        self.hierarchy = hierarchy


    def __add__(self, other):
        from csdl_alpha.src.operations.add import add
        return add(self,other)
    
    def __mul__(self, other):
        from csdl_alpha.src.operations.mult import mult
        return mult(self,other)

    def __neg__(self):
        from csdl_alpha.src.operations.neg import neg
        return neg(self)
    
    def __sub__(self, other):
        from csdl_alpha.src.operations.sub import sub
        return sub(self, other)