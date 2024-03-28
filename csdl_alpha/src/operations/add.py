from csdl_alpha.src.graph.operation import ElementwiseOperation

class Add(ElementwiseOperation):

    def __init__(self,x,y):
        super().__init__(x,y)
        self.name = 'add'
    
    # TODO: no more x.value --> x
    # TODO: pass in "np" into the function
    # TODO: ? Rename to compute_numpy?
    # TODO: make a wrapper in operation base class
    # TODO: inline = 'jax' option in recorder
    # TODO: "properties" dictionary for operations such as linear, diagonal jacobian, etc.
    # TODO: no more evaluate_diagonal_jacobian. Just evaluate_jacobian.
    #       -- If self.diagonal_jacobian = True, then return diagonal jacobian
    # TODO: 
         
    def compute_inline(self, x, y):
        return x.value + y.value

    def evaluate_jacobian(self, x, y):
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