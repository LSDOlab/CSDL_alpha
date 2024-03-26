from csdl_alpha.src.graph.operation import ElementwiseOperation

class Neg(ElementwiseOperation):

    def __init__(self,x):
        super().__init__(x)
        self.name = 'neg'

    def compute_inline(self, x):
        return -x.value

    def evaluate_diagonal_jacobian(self, x):
        return csdl.Constant(x.size, val = -1)

    def evaluate_jvp(self, x, vx):
        return -vx

    def evaluate_vjp(self, x, vout):
        return -vout

def neg(x):
    """
    doc strings
    """

    op = Neg(x)
    return op.get_outputs()