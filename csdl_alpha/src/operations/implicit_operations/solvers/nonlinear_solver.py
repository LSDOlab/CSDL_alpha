from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.src.operations.implicit_operations.implicit_operation import ImplicitOperation

class NonlinearSolver(object):
    def __init__(self, name = 'nlsolver'):
        self.name = name

        self.metadata = {}
        
        # state variable -> residual variable
        self.state_to_residual_map = {}
        
        # residual variable -> state variable
        self.residual_to_state_map = {}
        
        # state variable -> any other information about the state such as initial value, tolerance, etc.
        self.state_metadata = {}
        
        # CSDL variables that are "constants" from the perspective of the solver everytime it is ran
        # for example:
        # - initial conditions,
        # - brackets
        # - tolerances
        # Question: should we allow these to not be "constants"? IE, can we change the tolerance at every iteration?
        # If not, we can throw an error.
        self.meta_variables = set()

        # variables to build the implicit operation graph 
        self._intersection_sources = set()
        self._intersection_targets = set()


    def add_state_residual_pair(
            self, 
            state: Variable,
            residual: Variable):
        
        """
        state,residual: pair of residual and state variables to be solved for.

        Initializes mappings between states and residuals, and stores metadata about the state.
        """
        self.state_to_residual_map[state] = residual
        self.residual_to_state_map[residual] = state
        self.state_metadata[state] = {}

        self.add_intersection_source(state)
        self.add_intersection_target(residual)

    def add_intersection_source(self, source:Variable):
        self._intersection_sources.add(source)

    def add_intersection_target(self, target:Variable):
        self._intersection_targets.add(target)

    def run(self):
        """
        Creates the implicit operation graph and runs the solver if inline
        """

        # Steps:
        # G is the current graph we are in
        # S(G) is the subgraph of the implicit operation

        # 1. Perform the graph transformation:
        #   a) Identify subgraph S(G) between all sources (state) and sinks (residual/new_state) 
        #   b) Delete all operations in S(G) in G
        #   c) Enter a new subgraph of G and set S(G) as the graph
        #   d) Add the implicit_operation operation to G
        #        a) inputs: parameters, meta variables (fake edges)
        #           - All inputs should already exist in G
        #        b) the operation object
        #           - The operation object should be a subclass of ComposedOperation (??? would be crazy if this just works)
        #        c) outputs: new state variables, EVERY intermediate variable ... 
        #           - All outputs should already exist in G
        #   e) draw the edges: inputs --> implicit_operation --> outputs

        import csdl_alpha as csdl
        recorder = csdl.get_current_recorder()
        recorder.active_graph.visualize(f'top_level_{self.name}_before')

        # 1.a/b
        G = recorder.active_graph
        S, S_inputs, S_outputs = G.extract_subgraph(            
            sources = self._intersection_sources,
            targets = self._intersection_targets,
            keep_variables = True,    
        )

        # 1.c
        recorder._enter_subgraph()
        recorder.active_graph.replace(S)
        recorder.active_graph.visualize(f'implicit_function_{self.name}')
        self.update_residual = recorder.active_graph.execute_inline
        recorder._exit_subgraph()


        # 1.d/e
        state_variables = set(self.state_to_residual_map.keys())
        input_variables_set = self.meta_variables.union(S_inputs.symmetric_difference(state_variables))
        output_variables_set = state_variables.union(S_outputs)
        
        operation_metadata = {'nonlinear_solver': self}
        implicit_operation = ImplicitOperation(
            *list(input_variables_set),
            metadata = operation_metadata,
            name = f'implicit_{self.name}'
        )
        implicit_operation.outputs = list(output_variables_set)
        implicit_operation.get_outputs()

        recorder.active_graph.visualize(f'top_level_{self.name}_after')
        

    def solve(self, *args):
        """
        Solves the nonlinear system of equations.
        """
        
        for arg in args:
            print(arg)

        self.update_residual()