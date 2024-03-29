
from csdl_alpha.src.graph.operation import Operation
# TODO: make variable and operation easier to import

class ImplicitOperation(Operation):

    def __init__(self, *args, name = 'nl_op', **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.name = name
        self.nonlinear_solver = self.metadata['nonlinear_solver']

    def compute_inline(self, *args):

        print(f'COMPUTING NLSOLVER INLINE: {self.nonlinear_solver.name}')
        self.nonlinear_solver.solve_implicit_inline(*args)
        
        print(f'UPDATING DOWNSTREAM:')
        import csdl_alpha as csdl
        current_graph = csdl.get_current_recorder().active_graph
        current_graph.update_downstream(self)
        
        return [output.value for output in self.outputs]