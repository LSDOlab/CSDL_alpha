from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.src.operations.custom.custom import CustomExplicitOperation

class CustomDistribute(CustomExplicitOperation):
    def __init__(self, comm):
        super().__init__()
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    def evaluate(self, distributed_var):
        # Gather all local chunks into a global array
        self.declare_input('in_var', distributed_var)
        f_global = self.create_output('out_var',  distributed_var.shape)
        return f_global

    def compute(self, inputs, outputs):
        outputs['out_var'] = inputs['in_var']

    def compute_jacvec_product(self, input_vals, outputs_vals, d_inputs, d_outputs, mode):
        from mpi4py import MPI
        if mode == 'rev':
            # sum up the contributions from all ranks
            d_inputs['in_var'] = self.comm.allreduce(d_outputs['out_var'], op=MPI.SUM)

def sync_variable(x:Variable, comm):
    """
    Synchronizes a variables for reverse-mode derivatives across ranks.
    """
    return CustomDistribute(comm).evaluate(x)
