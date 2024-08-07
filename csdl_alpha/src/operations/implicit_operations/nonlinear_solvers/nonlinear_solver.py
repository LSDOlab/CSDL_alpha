from csdl_alpha.src.graph.variable import ImplicitVariable, Variable
from csdl_alpha.src.graph.operation import Operation
from csdl_alpha.src.operations.implicit_operations.implicit_operation import ImplicitOperation
from csdl_alpha.utils.inputs import scalarize, ingest_value, validate_and_variablize, get_type_string
import csdl_alpha.utils.error_utils as error_utils
from csdl_alpha.utils.error_utils import GraphError
import numpy as np

from csdl_alpha.src.operations.derivatives.bookkeeping import VarTangents

from csdl_alpha.utils.typing import VariableLike
from typing import Union, Any, Callable

class NonlinearSolver(object):
    def __init__(
            self,
            name = 'nlsolver', 
            print_status:bool = True,
            tolerance:float=1e-10,
            max_iter:int=100,
            elementwise_states:bool=False,
            residual_jac_kwargs:dict = None,
        ):
        if elementwise_states is True:
            raise ValueError("Elementwise_states argument is deprecated. Use keyword argument: residual_jac_kwargs = {'elementwise':True} instead")
        self.name = name
        self.print_status = print_status
        self.implicit_operation:ImplicitOperation = None

        self.metadata = {}
        
        # state variable -> residual variable
        self.state_to_residual_map:dict[Variable:Variable] = {}
        
        # residual variable -> state variable
        self.residual_to_state_map:dict[Variable:Variable] = {}
        
        # state variable -> any other information about the state such as initial value, tolerance, etc.
        self.state_metadata:dict[Variable:Any] = {}
        
        # CSDL variables that are "constants" from the perspective of the solver everytime it is ran
        # for example:
        # - initial conditions,
        # - brackets
        # - tolerances
        # Question: should we allow these to not be "constants"? IE, can we change the tolerance at every iteration?
        # If not, we can throw an error.
        self.meta_input_variables = set()
        self.meta_input_var_list = []

        # variables to build the implicit operation graph 
        self._intersection_sources = set()
        self._intersection_targets = set()

        # add metadata that is common to all solvers
        self.add_metadata('tolerance', tolerance)
        self.add_metadata('max_iter', max_iter)

        # block nonlinear solver from running more than once
        self.locked = False

        # Attributes for derivatives:
        self.full_residual_jacobian = None
        self.total_state_size = 0
        if residual_jac_kwargs is None:
            residual_jac_kwargs = {}
        elif not isinstance(residual_jac_kwargs, dict):
            raise TypeError(f"residual_jac_kwargs must be a dictionary. {get_type_string(residual_jac_kwargs)} given.")
        self.residual_jac_kwargs = residual_jac_kwargs

    def add_metadata(self, key, datum, is_input=True):
        if isinstance(datum, Variable) and is_input:
            self.meta_input_variables.add(datum)
            if datum not in self.meta_input_var_list:
                self.meta_input_var_list.append(datum)
        self.metadata[key] = datum

    def add_state_metadata(self, state:ImplicitVariable, key, datum, is_input=True):
        if isinstance(datum, Variable) and is_input:
            self.meta_input_variables.add(datum)
            if datum not in self.meta_input_var_list:
                self.meta_input_var_list.append(datum)
        self.state_metadata[state][key] = datum

    def add_tolerance(
            self,
            state:Variable,
            tolerance:Union[VariableLike, None],
            ):
        """Add a tolerance to state metadata. If tolerance is None, the default tolerance is used.
        If a CSDL variable, reshaped to state shape.
        If a scalar, it is broadcasted to state shape.

        Parameters
        ----------
        state : Variable
        tolerance : Union[VariableLike, None]
        """
        if tolerance is None:
            self.add_state_metadata(state, 'tolerance', self.metadata['tolerance'])
        else:
            if isinstance(tolerance, Variable):
                try:
                    tolerance = check_variable_shape_compatibility(tolerance, state)
                except Exception as e:
                    raise ValueError(f"Error with tolerance argument: {e}")
            else:
                tolerance = ingest_value(tolerance)
                if tolerance.size == 1:
                    tolerance = np.ones(state.shape) * tolerance.flatten()
                elif tolerance.shape != state.shape:
                    raise ValueError(f"Tolerance shape must match state shape. {tolerance.shape} given, {state.shape} expected.")
            self.add_state_metadata(state, 'tolerance', tolerance)

    def add_state_residual_pair(
            self, 
            state: ImplicitVariable,
            residual: Variable):
        
        """
        state,residual: pair of residual and state variables to be solved for.

        Initializes mappings between states and residuals, and stores metadata about the state.
        """
        import csdl_alpha as csdl
        current_graph = csdl.get_current_recorder().active_graph
        if self.locked:
            raise RuntimeError("Nonlinear solver has already been run. Cannot add more state-residual pairs.")
        
        if not isinstance(state, Variable):
            raise TypeError(f"State must be a Variable. {get_type_string(state)} given")
        elif current_graph.in_degree(state) != 0:
            raise TypeError(f"State must not be computed from another operation.")
        else:
            if state._check_nlsolver_conflict():
                raise ValueError(f"State with name {state.name} has already been previously added to a solver.")

        if not isinstance(residual, Variable):
            residual = validate_and_variablize(residual)
            raise TypeError(f"Residual must be a Variable. {residual} given")
        
        if state.shape != residual.shape:
            raise ValueError(error_utils.get_check_shape_mismatch_string(state, residual, 'state', 'residual'))

        if residual._check_nlsolver_conflict():
            raise ValueError(f"Residual with name {residual.name} has already been previously added to a solver.")

        self.state_to_residual_map[state] = residual
        self.residual_to_state_map[residual] = state
        self.state_metadata[state] = {}

        self.add_intersection_source(state)
        self.add_intersection_target(residual)

    def add_intersection_source(self, source:Variable):
        self._intersection_sources.add(source)

    def add_intersection_target(self, target:Variable):
        self._intersection_targets.add(target)

    def _preprocess_run(self):
        """
        Preprocesses the solver before running it.
        """
        pass

    def run(self):
        """
        Creates the implicit operation graph and runs the solver if inline is True
        """
        if len(self.state_to_residual_map) == 0:
            raise ValueError("No state-residual pairs added to the solver")
        
        if self.locked:
            raise RuntimeError("Nonlinear solver has already been run. Cannot run again.")

        self.locked = True

        self._preprocess_run()

        for residual in self.residual_to_state_map:
            state = self.residual_to_state_map[residual]

            # Keep track of indices for each state
            self.add_state_metadata(state, 'index_lower', self.total_state_size)
            self.total_state_size += state.size
            self.add_state_metadata(state, 'index_upper', self.total_state_size)

        # Steps:
        # G is the current graph we are in
        # S(G) is the subgraph of the implicit operation

        # 1. Perform the graph transformation:
        #   alpha) get order of inputs and outputs of G
        #   a) Identify subgraph S(G) between all sources (state) and sinks (residual/new_state) 
        #   b) Delete all operations in S(G) in G
        #   beta) reset order of operations in G.node table
        #   c) Enter a new subgraph of G and set S(G) as the graph
        #   d) Add the implicit_operation operation to G
        #        a) inputs: parameters, meta variables (fake edges)
        #           - All inputs should already exist in G
        #           - All inputs should be in the same order as the original graph
        #        b) the operation object
        #           - The operation object should be a subclass of ComposedOperation (??? would be crazy if this just works)
        #        c) outputs: new state variables, EVERY intermediate variable ... 
        #           - All outputs should already exist in G
        #           - All outputs should be in the same order as the original graph
        #   e) draw the edges: inputs --> implicit_operation --> outputs

        import csdl_alpha as csdl
        recorder = csdl.get_current_recorder()
        # recorder.active_graph.visualize(f'top_level_{self.name}_before')

        # 1.alpha
        # this order should be the same across loop iterations
        G = recorder.active_graph
        G_inputs = []
        G_outputs = []
        G_ops = []
        for node in G.node_table:
            if isinstance(node, Operation):
                G_ops.append(node)
                G_inputs.extend(node.inputs)
                G_outputs.extend(node.outputs)

        # 1.a/b
        try:
            S, S_inputs, S_outputs = G.extract_subgraph(            
                sources = self._intersection_sources,
                targets = self._intersection_targets,
                keep_variables = True,
                check_targets = list(self.state_to_residual_map.values()),
            )
        except Exception as e:
            raise ValueError(f"Error extracting non-linear solver residual function subgraph: {e.message}")
        
        # 1.beta
        for node in G_ops:
            if node in G.node_table:
                G.node_table[node] = G.node_table.pop(node)

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
        
        # >reorder inputs<

        # put meta variables first
        input_vars = self.meta_input_var_list
        # put state variables first
        output_vars = list(self.state_to_residual_map)
        # append vars that are inputs/outputs of operations
        for var in G_inputs:
            if var in input_variables_set and var not in input_vars:
                input_vars.append(var)
        for var in G_outputs:
            if var in output_variables_set and var not in output_vars:
                output_vars.append(var)
        # append any additional vars (unordered, should be ok?)
        # TODO: revisit this
        # eg, newton_nlsolver_jac
        for var in input_variables_set:
            if var not in input_vars:
                input_vars.append(var)

        operation_metadata = {'nonlinear_solver': self}
        implicit_operation = ImplicitOperation(
            *input_vars,
            metadata = operation_metadata,
            name = f'implicit_{self.name}'
        )
        implicit_operation.outputs = output_vars
        implicit_operation.assign_subgraph(self.residual_graph)
        
        # TODO: only perform these checks in debug mode?
        state_keys = set(self.state_to_residual_map.keys())
        if state_keys.symmetric_difference(set(self.state_metadata.keys())):
            raise ValueError("State variables do not match metadate state keys")

        implicit_operation.finalize_and_return_outputs()
        self.implicit_operation = implicit_operation

        # print(f'UPDATING DOWNSTREAM:  {self.name}')
        recorder = csdl.get_current_recorder()
        if recorder.inline:
            G.update_downstream(implicit_operation)
        
        # For debugging:
        # self.residual_graph.visualize(f'inner_graph_{self.name}')
        # recorder.active_graph.visualize(f'top_level_{self.name}_after')
    
    def prep_vjp(self):
        """
        Prepare the nonlinear solver for reverse mode differentiation.
        """
        self.get_full_residual_jacobian(for_deriv=True)

    def accumulate_cotangents(
            self,
            cotangents: VarTangents,
            outputs_with_cotangents: list[Variable],
            inputs_to_accumulate: list[Variable]
            ):
        # Steps
        # 1) Preprocess cotangents of the states by accumulating them with exposed output cotangents
        # 2) Solve adjoint system
        # 3) Compute VJP of the adjoint residuals
        # 4) Accumulate exposed output cotangents with the adjoint residuals
        import csdl_alpha as csdl
        recorder = csdl.get_current_recorder()

        # Step 1
        seeds:list[tuple[Variable, Variable]] = []
        for exposed_outer_variable in outputs_with_cotangents:
            # if exposed_outer_variable in self.state_to_residual_map:
            #     continue
            if exposed_outer_variable in self.state_to_residual_map:
                continue
            seeds.append((exposed_outer_variable, cotangents[exposed_outer_variable]))
        
        wrts = []
        wrts_set = set()
        for input_variable in inputs_to_accumulate:
            
            if input_variable in self.meta_input_variables:
                continue

            wrts.append(input_variable)
            wrts_set.add(input_variable)
        for state in self.state_to_residual_map:
            if state not in wrts_set:
                wrts.append(state)

        from csdl_alpha.src.operations.derivatives.reverse import vjp
        
        # TODO: Should VJP function should be INSIDE or OUTSIDE the nonlinear solver?
        # Compute vector-Jacobian product of the residuals in the residual graph
        # recorder._enter_subgraph(graph = self.residual_graph)
        # residual_graph = self.residual_graph
        # for seed in seeds:
        #     residual_graph.add_node(seed[0])
        # recorder.visualize_graph('pre_step1_residual_graph')
        # recorder.visualize_graph('post_step1_residual_graph')
        # recorder._exit_subgraph()

        # recorder.visualize_graph('pre_step1')
        exposed_vjps = vjp(seeds, wrts, self.residual_graph)
        for state in self.state_to_residual_map:
            cotangents.initialize(state)
            if exposed_vjps[state] is not None:
                cotangents.accumulate(state, exposed_vjps[state])
        # recorder.visualize_graph('post_step1')
        
        # step 2
        # full_residual_jacobian_T = self.get_full_residual_jacobian(for_deriv=True).T()
        full_residual_jacobian_T = self.get_full_residual_jacobian_for_deriv().T()

        residual_vector = csdl.Variable(name = 'residual_vector', value = np.zeros((self.total_state_size,)))
        for current_state in self.state_to_residual_map.keys():
            il = self.state_metadata[current_state]['index_lower']
            iu = self.state_metadata[current_state]['index_upper']
            if cotangents[current_state] is not None:
                residual_vector = residual_vector.set(csdl.slice[il:iu], cotangents[current_state].flatten())
        psi = csdl.solve_linear(full_residual_jacobian_T, residual_vector)
        # recorder.visualize_graph('post_step2')

        # step 3
        seeds:list[tuple[Variable, Variable]] = []
        for current_state, current_residual in self.state_to_residual_map.items():
            il = self.state_metadata[current_state]['index_lower']
            iu = self.state_metadata[current_state]['index_upper']
            seeds.append((current_residual, psi[il:iu].reshape(current_residual.shape)))
        psi_vjps = vjp(seeds, list(wrts_set), self.residual_graph)
        # recorder.visualize_graph('post_step3')

        # step 4
        for input_variable in inputs_to_accumulate:
            # cotangents.accumulate(input_variable,  csdl.Variable(value = 0.01+np.zeros(input_variable.shape)))
            if input_variable not in self.meta_input_variables:
                if psi_vjps[input_variable] is not None:
                    cotangents.accumulate(input_variable, -psi_vjps[input_variable])
                if exposed_vjps[input_variable] is not None:
                    cotangents.accumulate(input_variable, exposed_vjps[input_variable])
            if cotangents[input_variable] is None:
                cotangents.accumulate(input_variable, csdl.Variable(value = np.zeros(input_variable.shape)))
        # recorder.visualize_graph('post_step4')
        
        pass

    def get_full_residual_jacobian(
            self, 
            for_deriv:bool = False,    
        )-> Variable:
        if self.full_residual_jacobian is None:
            self.full_residual_jacobian = self._build_full_residual_jacobian(for_deriv=for_deriv)
            return self.full_residual_jacobian
        else:
            return self.full_residual_jacobian

    def _build_full_residual_jacobian(
            self,
            for_deriv:bool,
        )-> Variable:
        import csdl_alpha as csdl
        states_list = list(self.state_to_residual_map.keys())
        residuals_list = list(self.state_to_residual_map.values())
        self.residual_jac_kwargs['as_block'] = True
        if for_deriv:
            self.residual_jac_kwargs['graph'] = self.residual_graph
        jac = csdl.derivative(residuals_list, states_list, **self.residual_jac_kwargs)
        jac.add_name(f'{self.name}_jac')
        return jac

    def get_full_residual_jacobian_for_deriv(self)->Variable:
        if self.implicit_operation.vjp_prep_enabled:
            return self.get_full_residual_jacobian(for_deriv=True)
        else:
            return self._build_full_residual_jacobian(for_deriv=True)

    def _inline_solve_(self):
        raise NotImplementedError("Solve method must be implemented by subclass")
    
    def _jax_solve_(
            self,
            jax_residual_function:Callable,
            jax_intermediate_function:Callable,
            input_dict:dict,
        ):
        """Solves the nonlinear equation using JAX.

        This method is responsible for solving the nonlinear equation using the JAX library.
        It takes a JAX residual function and input variables as arguments.

        Parameters
        ----------
        jax_residual_function : function
            The JAX residual function that represents the nonlinear equation. 
            Takes in a list of states and outputs a list of residuals, in order of state_to_residual_map.
        input_dict : dict
            A dictionary mapping CSDL variable inputs to their jax values.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by the subclass.

        """
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
                
    def _inline_print_nl_status(self, iter_num:int, did_converge:bool)->None:
        """
        Print the status of the nonlinear solver.
        """
        if did_converge:
            return f'nonlinear solver: {self.name} converged in {iter_num} iterations.'
        else:
            main_str = f'\nnonlinear solver: {self.name} did not converge in {iter_num} iterations.\n'
            for i, (state,residual) in enumerate(self.state_to_residual_map.items()):
                state_str  = f'    state {i}\n'
                state_str += f'        name:     {state.name}\n'
                state_str += f'        value:    {state.value}\n'
                state_str += f'        residual: {residual.value}\n'
                main_str += state_str
            return main_str

    def _inline_check_converged(self,):
        converged = True
        for current_state, current_residual in self.state_to_residual_map.items():

            # get current residual and error
            current_residual_value = current_residual.value

            # Uncomment to print iteration:
            # error = np.linalg.norm(current_residual_value.flatten())
            # print(f'iteration {iter}, {current_residual} error: {error}')

            # compute tolerance:
            tol = self.state_metadata[current_state]['tolerance']
            if isinstance(tol, Variable):
                tol = tol.value

            if np.any(np.isnan(current_residual_value)):
                raise ValueError(f'Residual is NaN for state {current_state.name} with residual {current_residual.name}')
            
            # if current_residual_value > tol:
            # if any of the residuals do not meet tolerance, no need to compute errors for other residuals
            if not check_run_time_tolerance(current_residual_value, tol):
                converged = False
                break
        return converged
    
    def _jax_check_converged(self, residuals, iter, input_dict):
        import jax.numpy as jnp
        converged = jnp.less_equal(0, 1)
        for i, state in enumerate(self.state_to_residual_map.keys()):
            residual = residuals[i]
            tol = self.state_metadata[state]['tolerance']
            if isinstance(tol, Variable):
                tol = input_dict[tol]
            
            converged = converged & jnp.all(jnp.less_equal(jnp.abs(residual), tol))

        converged = converged | jnp.greater_equal(iter, self.metadata['max_iter'])
        
        return converged

    def solve_implicit_inline(self, *args):
        """
        Solves the nonlinear system of equations.
        """
        
        # for arg in args:
        #     print(arg)

        self._inline_solve_()

    def solve_implicit_jax(self, *args, **kwargs):
        """
        Solves the nonlinear system of equations.
        """
        return self._jax_solve_(*args, **kwargs)

def check_variable_shape_compatibility(
        var_to_check:Variable, 
        var_reference:Variable,
    )->Variable:
    """_summary_

    Parameters
    ----------
    var_to_check : Variable
        the variable to check
    
    var_reference : Variable
        the variable to compare against

    Returns
    -------
    Variable
        the variable to check (expanded if necessary)
    """
    if var_to_check.size == 1:
        from csdl_alpha.src.operations import expand
        var_to_check = expand(var_to_check.flatten(), var_reference.shape)
    elif var_to_check.shape != var_reference.shape:
        raise ValueError(f"Variable shapes are incompatible. {var_to_check.shape} given, {var_reference.shape} expected.")
    return var_to_check

def check_run_time_tolerance(residual:np.ndarray, tol:np.ndarray):
    """Given a residual and a tolerance array, check if all the residuals are within tolerance.

    Parameters
    ----------
    residual : np.ndarray
    tol : np.ndarray

    Returns
    -------
    bool
        True if all residuals are within tolerance, False otherwise.
    """
    return np.all(np.abs(residual) <= tol)
