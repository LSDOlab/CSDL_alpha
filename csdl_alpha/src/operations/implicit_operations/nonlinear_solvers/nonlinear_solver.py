from csdl_alpha.src.graph.variable import ImplicitVariable, Variable
from csdl_alpha.src.operations.implicit_operations.implicit_operation import ImplicitOperation
from csdl_alpha.utils.inputs import scalarize, ingest_value
import csdl_alpha.utils.error_utils as error_utils
from csdl_alpha.utils.error_utils import GraphError

class NonlinearSolver(object):
    def __init__(
            self,
            name = 'nlsolver', 
            print_status = True,
            tolerance=1e-10,
            max_iter=100,
        ):
        self.name = name
        self.print_status = print_status

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
        self.meta_input_variables = set()

        # variables to build the implicit operation graph 
        self._intersection_sources = set()
        self._intersection_targets = set()

        # add metadata that is common to all solvers
        self.add_metadata('tolerance', tolerance)
        self.add_metadata('max_iter', max_iter)

        # block nonlinear solver from running more than once
        self.locked = False

    def add_metadata(self, key, datum, is_input=True):
        if isinstance(datum, Variable) and is_input:
            self.meta_input_variables.add(datum)
        self.metadata[key] = datum

    def add_state_metadata(self, state:ImplicitVariable, key, datum, is_input=True):
        if isinstance(datum, Variable) and is_input:
            self.meta_input_variables.add(datum)
        self.state_metadata[state][key] = datum

    def add_state_residual_pair(
            self, 
            state: ImplicitVariable,
            residual: Variable):
        
        """
        state,residual: pair of residual and state variables to be solved for.

        Initializes mappings between states and residuals, and stores metadata about the state.
        """
        if self.locked:
            raise ValueError("Nonlinear solver has already been run. Cannot add more state-residual pairs.")
        if not isinstance(state, ImplicitVariable):
            raise TypeError(f"State must be an ImplicitVariable. {state} given")
        else:
            if state.in_solver:
                raise ValueError(f"Implicit variable {state.name} has already been previously added to a solver.")
            state.in_solver = True

        if not isinstance(residual, Variable):
            raise TypeError(f"Residual must be a Variable. {residual} given")
        
        if state.shape != residual.shape:
            raise ValueError(error_utils.get_check_shape_mismatch_string(state, residual, 'state', 'residual'))

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
        Creates the implicit operation graph and runs the solver if inline is True
        """
        if len(self.state_to_residual_map) == 0:
            raise ValueError("No state-residual pairs added to the solver")
        
        self.locked = True

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
        # recorder.active_graph.visualize(f'top_level_{self.name}_before')

        # 1.a/b
        G = recorder.active_graph
        try:
            S, S_inputs, S_outputs = G.extract_subgraph(            
                sources = self._intersection_sources,
                targets = self._intersection_targets,
                keep_variables = True,    
            )
        except Exception as e:
            raise ValueError(f"Error extracting non-linear solver residual function subgraph: {e.message}")
        
        # print(S.node_table, S_inputs, S_outputs)

        # 1.c
        recorder._enter_subgraph(name = self.name)
        recorder.active_graph.replace(S)
        self.update_residual = recorder.active_graph.execute_inline
        self.residual_graph = recorder.active_graph
        recorder._exit_subgraph()

        # 1.d/e
        state_variables = set(self.state_to_residual_map.keys())
        input_variables_set = self.meta_input_variables.union(S_inputs.symmetric_difference(state_variables))
        output_variables_set = state_variables.union(S_outputs)
        
        operation_metadata = {'nonlinear_solver': self}
        implicit_operation = ImplicitOperation(
            *list(input_variables_set),
            metadata = operation_metadata,
            name = f'implicit_{self.name}'
        )
        implicit_operation.outputs = list(output_variables_set)
        implicit_operation.assign_subgraph(self.residual_graph)
        
        # TODO: only perform these checks in debug mode?
        state_keys = set(self.state_to_residual_map.keys())
        if state_keys.symmetric_difference(set(self.state_metadata.keys())):
            raise ValueError("State variables do not match metadate state keys")

        implicit_operation.finalize_and_return_outputs()

        # print(f'UPDATING DOWNSTREAM:  {self.name}')
        G.update_downstream(implicit_operation)

        # recorder.active_graph.visualize(f'top_level_{self.name}_after')
        

    def _inline_solve_(self):
        raise NotImplementedError("Solve method must be implemented by subclass")

    def _inline_set_initial_values(self):
        """
        Set initial values for the state variables.
        'initial_value' must be a metadata key where value is a number or variable
        """
        for state in self.state_metadata:
            if not isinstance(self.state_metadata[state]['initial_value'], Variable):
                state.value = ingest_value(self.state_metadata[state]['initial_value'])
            else:
                state.value = self.state_metadata[state]['initial_value'].value
                
    def _inline_print_nl_status(self, iter_num, did_converge):
        """
        Print the status of the nonlinear solver.
        """
        if did_converge:
            return f'nonlinear solver: {self.name} converged in {iter_num} iterations.'
        else:
            main_str = f'\nnonlinear solver: {self.name} did not converge in {iter_num} iterations.\n'
            for state,residual in self.state_to_residual_map.items():
                state_str = f'\t{state}\n'
                state_str += f'\t\tname:  {state.name}\n'
                state_str += f'\t\tval:    {state.value}\n'
                state_str += f'\t\tres:    {residual.value}\n'
                main_str += state_str
            return main_str

    def solve_implicit_inline(self, *args):
        """
        Solves the nonlinear system of equations.
        """
        
        # for arg in args:
        #     print(arg)

        self._inline_solve_()

        