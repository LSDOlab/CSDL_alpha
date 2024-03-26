from csdl_alpha.src.graph.operation import ElementwiseOperation

class Mult(ElementwiseOperation):

    def __init__(self,x,y):
        super().__init__(x,y)
        self.name = 'mult'

    def compute_inline(self, x, y):
        return x.value*y.value

    def evaluate_diagonal_jacobian(self, x, y):
        return y, x

    def evaluate_jvp(self, x,y, vx, vy):
        return y.flatten()*vx + x.flatten()*vy

    def evaluate_vjp(self, x, y, vout):
        return x.flatten()*vout, y.flatten()*vout

def mult(x,y):
    """
    doc strings
    """

    op = Mult(x,y)
    return op.get_outputs()