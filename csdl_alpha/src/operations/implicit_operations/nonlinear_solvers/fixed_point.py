from csdl_alpha.src.operations.implicit_operations.nonlinear_solvers.nonlinear_solver import NonlinearSolver, check_variable_shape_compatibility, check_run_time_tolerance
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.utils.inputs import scalarize, ingest_value
import numpy as np

from typing import Union

class FixedPoint(NonlinearSolver):
    
    def __init__(
            self,
            name,
            print_status,
            tolerance,
            max_iter,
            elementwise_states,
            residual_jac_kwargs,
        ):
        """
        Base class for standard fixed point iteration solvers. Not to be used directly.
        """
        # well now this just seems redundant
        super().__init__(
            name = name,
            print_status = print_status,
            tolerance = tolerance,
            max_iter = max_iter,
            elementwise_states = elementwise_states,
            residual_jac_kwargs = residual_jac_kwargs,
        )

    def add_state(*args, **kwargs):
        """
        Not implemented
        """
        raise NotImplementedError("add_state method not implemented for this solver")
    
    def _inline_update_states(self):
        """
        Not implemented
        """
        raise NotImplementedError("_inline_update_states method not implemented for this solver")
    
    def _jax_update_states(self, jax_residual_function, val, input_var_dict):
        """
        Not implemented
        """
        raise NotImplementedError("_jax_update_states method not implemented for this solver")

    def _inline_solve_(self):
        """
        Solves the implicit operation graph using Nonlinear Gauss-Seidel
        """

        iter = 0

        self._inline_set_initial_values()

        while True:
            # update all states
            self._inline_update_states()
            
            # update residuals to check
            self.update_residual()

            # check convergence
            converged = self._inline_check_converged()

            # if solved or maxiter, end loop
            if converged:
                break
            iter += 1
            if iter >= self.metadata['max_iter']:
                break
        # print status
        if self.print_status:
            print(self._inline_print_nl_status(iter, converged))

    def _jax_solve_(self, jax_residual_function, input_vars, *inputs):
        """
        Solves the implicit operation graph using Nonlinear Gauss-Seidel
        """
        import jax.numpy as jnp
        import jax.lax as lax
        input_var_dict = {input_var:input for input_var, input in zip(input_vars, inputs)}

        def loop_body(val): # (states, residuals, iter)
            # update all states
            states = self._jax_update_states(jax_residual_function, val, input_var_dict)
            
            # compute residuals
            residuals = jax_residual_function(states)

            # update iter number
            iter = val[2] + 1
            
            return (states, residuals, iter)
        
        def loop_condition(val):
            residuals = val[1]
            iter = val[2]
            return ~self._jax_check_converged(residuals, iter, input_var_dict)
        
        # set initial values
        states = []
        residuals = []
        for state in self.state_to_residual_map.keys():
            residuals.append(jnp.ones(state.shape))
            if not isinstance(self.state_metadata[state]['initial_value'], Variable):
                value = ingest_value(self.state_metadata[state]['initial_value'])
                print(value.shape)
                states.append(jnp.array(value))
            else:
                states.append(input_var_dict[self.state_metadata[state]['initial_value']])
        val = (states, residuals, 0)

        # loop
        states, residuals, iter = lax.while_loop(loop_condition, loop_body, val)
        return states


        
        