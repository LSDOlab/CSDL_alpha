from csdl_alpha.src.operations.implicit_operations.nonlinear_solvers.nonlinear_solver import NonlinearSolver, check_variable_shape_compatibility, check_run_time_tolerance
from csdl_alpha.src.operations.implicit_operations.nonlinear_solvers.gauss_seidel import GaussSeidel
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.utils.inputs import scalarize, ingest_value
import numpy as np

from typing import Union, Callable


class Jacobi(GaussSeidel):
    
    def __init__(
            self,
            name = 'jacobi_nlsolver',
            print_status:bool = True,
            tolerance:float=1e-10,
            max_iter:int=100,
            elementwise_states:bool=False,
            residual_jac_kwargs:dict = None,
        ):
        """
        A nonlinear block Jacobi solver

        The states are updated all at once:
        x0_new = x0_state_update(x0_old, x1_old, ... xn_old)
        x1_new = x1_state_update(x0_old, x1_old, ... xn_old)
        ...
        xn_new = xn_state_update(x0_old, x1_old, ... xn_old)
        """
        super().__init__(
            name = name,
            print_status = print_status,
            tolerance = tolerance,
            max_iter = max_iter,
            elementwise_states = elementwise_states,
            residual_jac_kwargs = residual_jac_kwargs,
        )

    def _inline_update_states(self):
        # while not converged:
        #    x0_new = x0_state_update(x0_old, x1_old, ... xn_old)
        #    x1_new = x1_state_update(x0_old, x1_old, ... xn_old)
        #    ...
        #    xn_new = xn_state_update(x0_old, x1_old, ... xn_old)
        
        # compute residuals once before hand
        for current_state, current_residual in self.state_to_residual_map.items():
            # get current state value and residual value
            current_state_value = current_state.value
            current_residual_value = current_residual.value

            # update current state value
            if self.state_metadata[current_state]['state_update'] is None:
                current_state.value = current_state_value - current_residual_value
            else:
                current_state.value = self.state_metadata[current_state]['state_update'].value

    def _jax_update_states(
            self,
            jax_residual_function:Callable,
            jax_intermediate_function:Callable,
            val,
            input_var_dict):
        from csdl_alpha.backends.jax.graph_to_jax import create_jax_function
        # while not converged:
        #    x0_new = x0_state_update(x0_old, x1_old, ... xn_old)
        #    x1_new = x1_state_update(x0_old, x1_old, ... xn_old)
        #    ...
        #    xn_new = xn_state_update(x0_old, x1_old, ... xn_old)
        
        # compute residuals once before hand
        states = val[0]
        residual = val[1]
        graph_input_dict = {key: value for key, value in input_var_dict.items() if key in self.residual_graph.node_table}
        update_outputs = [self.state_metadata[state]['state_update'] for state in self.state_to_residual_map if self.state_metadata[state]['state_update'] is not None]
        j = 0
        new_states = []
        for i, current_state_var in enumerate(self.state_to_residual_map.keys()):
            # update current state value
            if self.state_metadata[current_state_var]['state_update'] is None:
                new_states.append(states[i] - residual[i])
            else:
                jax_update_fn = create_jax_function(self.residual_graph,
                                            update_outputs,
                                            [key for key in graph_input_dict]+list(self.state_to_residual_map))
                state_updates = jax_update_fn(*([val for val in graph_input_dict.values()]+states))
                new_states.append(state_updates[j])
                j += 1
        return new_states
