from csdl_alpha.src.graph.operation_old import ElementwiseOperation

class Add(ElementwiseOperation):

    def __init__(self,x,y):
        super().__init__(x,y)
        self.name = 'add'
    
    # TODO: no more x.value --> x: done
    # TODO: pass in "np" into the function: no
    # TODO: ? Rename to compute_numpy?: no
    # TODO: make a wrapper in operation base class:  just make another jax specific method
    # TODO: inline = 'jax' option in recorder: just make another jax specific method
        
    # TODO: "properties" dictionary for operations such as linear, diagonal jacobian, etc.: done
        
    # TODO: no more evaluate_diagonal_jacobian. Just evaluate_jacobian.
    #       -- If self.diagonal_jacobian = True, then return diagonal jacobian
        
    # TODO: We need new broadcasting classes (but no broadcasting base class): ok
        
    # TODO: No Sparse/Dense addition elementwise, always make the sparse to dense.
         
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
    if x.shape == y.shape:
        op = Add(x,y)
    elif x.shape == (1,):
        op = BroadcastAdd(x,y)
    elif y.shape == (1,):
        op = BroadcastAdd(y,x)
    else:
        raise ValueError('Shapes do not match')
    return op.get_outputs()