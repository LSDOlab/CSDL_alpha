from csdl_alpha.src.graph.node import Node
import numpy as np
from typing import Union
from csdl_alpha.utils.inputs import ingest_value, check_shape

class Variable(Node):
    is_input = True
    is_implicit = False


    def __init__(self, shape: tuple = None, 
                 *, 
                 name: str = None, 
                 value: Union[np.ndarray, float, int] = None,  
                 tags: list[str] = [], 
                 hierarchy: int = None):
        
        super().__init__()
        self.recorder._add_node(self)

        value = ingest_value(value)
        check_shape(shape, value)
        
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