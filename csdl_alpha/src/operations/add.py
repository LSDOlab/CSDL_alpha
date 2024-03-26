from csdl_alpha.src.graph.operation import ElementwiseOperation

class Add(ElementwiseOperation):

    def __init__(self,x,y):
        super().__init__(x,y)
        self.name = 'add'

    def compute_inline(self, x, y):
        return x.value + y.value

    def evaluate_diagonal_jacobian(self, x, y):
        return csdl.Constant(x.size, val = 1), csdl.Constant(y.size, val = 1)

    def evaluate_jvp(self, x,y, vx, vy):
        return add(vx.flatten(), vy.flatten())

    def evaluate_vjp(self, x, y, vout):
        return vout.flatten(), vout.flatten()

def add(x,y):
    """
    doc strings
    """

    op = Add(x,y)
    return op.get_outputs()