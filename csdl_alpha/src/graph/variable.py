from csdl_alpha.src.graph.node import Node
import numpy as np
from typing import Union

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

        if isinstance(value, (float, int)):
            value = np.array([value])
        elif not isinstance(value, np.ndarray) and value is not None:
            raise ValueError("Value must be a numpy array, float or int")
        
        if shape is None:
            if value is not None:
                shape = value.shape
            else:
                raise ValueError("Shape or value must be provided")
        else:
            if value is not None:
                if shape != value.shape:
                    raise ValueError("Shape and value shape must match")
        
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