from csdl_alpha.src.operations.operation_subclasses import ElementwiseOperation
from csdl_alpha.src.graph.operation import set_properties 

@set_properties(linear=True)
class Neg(ElementwiseOperation):

    def __init__(self,x):
        super().__init__(x)
        self.name = 'neg'

    def compute_inline(self, x):
        return -x
    
    def evaluate_jacobian(self, x):
        return csdl.Constant(x.size, val = -1)

    def evaluate_jvp(self, x, vx):
        return -vx

    def evaluate_vjp(self, x, vout):
        return -vout

def neg(x):
    """
    doc strings
    """
    return Neg(x).finalize_and_return_outputs()